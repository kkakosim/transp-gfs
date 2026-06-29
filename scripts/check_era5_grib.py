"""Smoke-test a single ERA5 GRIB file with pygrib.

Usage:
    python scripts/check_era5_grib.py data/era5/era5_pl_2022060100_2022063023.grib2

Prints per-message metadata for the first 20 messages, then counts
the total messages.  If pygrib segfaults, the last printed line tells
you which message it choked on.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(path: str) -> int:
    import pygrib                              # noqa: PLC0415

    p = Path(path)
    print(f"opening {p} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
    grbs = pygrib.open(str(p))
    n = 0
    try:
        for msg in grbs:
            n += 1
            if n <= 20 or n % 5000 == 0:
                print(
                    f"  msg {n:6d}: "
                    f"short={getattr(msg, 'shortName', '?'):8s} "
                    f"level={getattr(msg, 'level', '?'):6} "
                    f"tol={getattr(msg, 'typeOfLevel', '?'):20s} "
                    f"validDate={getattr(msg, 'validDate', '?')}",
                    flush=True,
                )
            if n <= 3:
                # Force lat/lon and values to verify reading the array works
                lats, lons = msg.latlons()
                vals = msg.values
                print(
                    f"           shape={vals.shape} "
                    f"range=[{vals.min():.2f}, {vals.max():.2f}]",
                    flush=True,
                )
    finally:
        grbs.close()
    print(f"done. total messages: {n}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_era5_grib.py <path-to-grib>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
