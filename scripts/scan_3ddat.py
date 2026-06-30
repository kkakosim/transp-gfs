"""Scan a 3D.DAT file for cells whose values would crash CALMET.

Looks for:
  - surface mslp <= 0
  - surface t2 <= 0
  - surface q2 <= 0
  - vertical record temp <= 0
  - vertical record vapmr == 0.00 (written that way in F5.2)
  - vertical record rh == 0   (written that way in i3)

Usage:
    python scripts/scan_3ddat.py out/ERA5_2022070100.3D.DAT
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Surface line: starts with 10 digits (YYYYMMDDHH), then 2 ix/jx fields.
_SURF_RE = re.compile(
    r"^(?P<stamp>\d{10})"
    r"(?P<ix>.{3})(?P<jx>.{3})"
    r"(?P<pres>.{7})(?P<rain>.{5})(?P<sc>.{2})"
    r"(?P<radsw>.{8})(?P<radlw>.{8})(?P<t2>.{8})"
    r"(?P<q2>.{8})(?P<wd10>.{8})(?P<ws10>.{8})(?P<sst>.{8})"
)


def _f(s: str) -> float:
    return float(s.strip())


_PRES_LEVELS = {1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
                750, 725, 700, 675, 650, 625, 600, 575, 550, 525,
                500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50}


def main(path: str) -> int:
    p = Path(path)
    print(f"scanning {p}", flush=True)
    issues: list[str] = []
    n_surf = 0
    n_vert = 0
    current_stamp = "?"
    current_ix = current_jx = "?"
    in_data = False           # only inspect verticals after first surface line

    with open(p, "r", encoding="ascii", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            m = _SURF_RE.match(line)
            if m:
                in_data = True
                n_surf += 1
                current_stamp = m.group("stamp")
                current_ix = m.group("ix").strip()
                current_jx = m.group("jx").strip()
                pres = _f(m.group("pres"))
                t2 = _f(m.group("t2"))
                q2 = _f(m.group("q2"))
                if pres <= 0:
                    issues.append(
                        f"L{lineno} {current_stamp} ix={current_ix} jx={current_jx}: pres={pres}")
                if t2 <= 0:
                    issues.append(
                        f"L{lineno} {current_stamp} ix={current_ix} jx={current_jx}: t2={t2}")
                if q2 <= 0:
                    issues.append(
                        f"L{lineno} {current_stamp} ix={current_ix} jx={current_jx}: q2={q2}")
                continue

            if not in_data:
                # Still inside the header block (grid points etc.).  Skip.
                continue

            # Vertical record: i4 i6 f6.1 i4 f5.1 [i3 f5.2]
            if len(line) < 25:
                continue
            try:
                pres_v = int(line[0:4])
            except ValueError:
                continue
            # Header grid-point lines have iindex(i4)+jindex(i4) starting
            # at column 0; the first int there is in [1, NX] not a real
            # pressure level.  Restrict to recognised pressure levels.
            if pres_v not in _PRES_LEVELS:
                continue
            try:
                z_v = int(line[4:10])
                tempk = float(line[10:16])
            except ValueError:
                continue
            n_vert += 1
            if z_v <= 0:
                issues.append(
                    f"L{lineno} {current_stamp} ix={current_ix} jx={current_jx} "
                    f"pres={pres_v}: z={z_v}")
            if tempk <= 0:
                issues.append(
                    f"L{lineno} {current_stamp} ix={current_ix} jx={current_jx} "
                    f"pres={pres_v}: tempk={tempk}")
            # rh and vapmr only present when ioutq=1 (line length >= 33).
            if len(line) >= 33:
                try:
                    rh = int(line[25:28])
                    vapmr = float(line[28:33])
                except ValueError:
                    continue
                if rh <= 0:
                    issues.append(
                        f"L{lineno} {current_stamp} ix={current_ix} jx={current_jx} "
                        f"pres={pres_v}: rh={rh}")
                if vapmr <= 0:
                    issues.append(
                        f"L{lineno} {current_stamp} ix={current_ix} jx={current_jx} "
                        f"pres={pres_v}: vapmr={vapmr}")

    print(f"scanned {n_surf} surface + {n_vert} vertical records", flush=True)
    if not issues:
        print("no zero/negative values found.", flush=True)
        return 0
    print(f"{len(issues)} issue(s):", flush=True)
    for s in issues[:50]:
        print("  " + s)
    if len(issues) > 50:
        print(f"  ... and {len(issues) - 50} more")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: scan_3ddat.py <path-to-3D.DAT>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
