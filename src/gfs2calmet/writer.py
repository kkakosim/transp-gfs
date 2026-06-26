"""Write a CALMET 3D.DAT (v2.1) file from in-memory dataclasses.

Reference: CALPUFF v6 User Instructions, Section 7.7, Tables 7-32/7-33.

The file is text-only with one record per line. We use ``\\n`` line
endings (LF) regardless of host OS to match the Unix-style output of
the reference CALWRF/CALMM5 binaries — CALMET on Windows reads both
line endings fine.

The hydrometeor compression rule (per the spec footnote): if all
emitted mixing-ratio fields for a vertical record are zero, write a
single ``-N.000`` field (where N = number of emitted ratios) in place
of the N individual F6.3 fields.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

from gfs2calmet.dataset import (
    CellData,
    Frame,
    Frames,
    GridPoint,
    Header,
    OutputFlags,
    SurfaceRecord,
    VerticalRecord,
)
from gfs2calmet.fortran_format import fmt_a, fmt_f, fmt_i, fmt_x, to_ascii


_LINE_END = "\n"

# Hydrometeor compression threshold from CALWRF v2.0.3 (subroutine wrtcmp,
# parameter xzero). Magnitudes below this are treated as zero for the
# purpose of deciding whether to emit the compressed ``-N.000`` token.
_HYDRO_ZERO_THRESHOLD = 0.00049


# ---------------------------------------------------------------------------
# Header writers
# ---------------------------------------------------------------------------


def _write_header_record_1(out: IO[str], header: Header) -> None:
    """Format(2a16, a64) — dataset name, version, message.

    Text fields are transliterated to ASCII so the file is safe for
    CALMET's FORTRAN reader regardless of what the user pasted into
    the YAML config.
    """
    out.write(
        fmt_a(to_ascii(header.dataset_name), 16)
        + fmt_a(to_ascii(header.dataset_version), 16)
        + fmt_a(to_ascii(header.dataset_message), 64)
        + _LINE_END
    )


def _write_header_comments(out: IO[str], header: Header) -> None:
    """Header Record #2: NCOMM as (i4) (matches CALWRF v2.0.3 line 898).
    Header Records #3..NCOMM+2: (a132) per comment.

    Each comment is transliterated to ASCII before writing (em-dashes,
    smart quotes, etc. -> ASCII equivalents) so YAML configs that
    pasted typography don't break the 3D.DAT.
    """
    out.write(fmt_i(len(header.comments), 4) + _LINE_END)
    for c in header.comments:
        out.write(fmt_a(to_ascii(c.text), 132) + _LINE_END)


def _write_header_flags(out: IO[str], flags: OutputFlags) -> None:
    """Format(6i3) — IOUTW IOUTQ IOUTC IOUTI IOUTG IOSRF."""
    out.write(
        fmt_i(flags.ioutw, 3)
        + fmt_i(flags.ioutq, 3)
        + fmt_i(flags.ioutc, 3)
        + fmt_i(flags.iouti, 3)
        + fmt_i(flags.ioutg, 3)
        + fmt_i(flags.iosrf, 3)
        + _LINE_END
    )


def _write_header_projection(out: IO[str], header: Header) -> None:
    """Format(a4, f9.4, f10.4, 2f7.2, 2f10.3, f8.3, 2i4, i3)."""
    p = header.projection
    d = header.domain
    out.write(
        fmt_a(to_ascii(p.maptxt), 4)
        + fmt_f(p.rlatc, 9, 4)
        + fmt_f(p.rlonc, 10, 4)
        + fmt_f(p.truelat1, 7, 2)
        + fmt_f(p.truelat2, 7, 2)
        + fmt_f(p.x1dmn, 10, 3)
        + fmt_f(p.y1dmn, 10, 3)
        + fmt_f(p.dxy, 8, 3)
        + fmt_i(d.nx, 4)
        + fmt_i(d.ny, 4)
        + fmt_i(d.nz, 3)
        + _LINE_END
    )


def _write_header_model_options(out: IO[str], header: Header) -> None:
    """Format(30i3) — MM5 physics codes + FLAGS_2D[12] + NLAND.

    For non-MM5 sources every code can be 0; NLAND should still match
    the downstream CALMET landuse setup. We emit exactly 21 values
    (the documented variables); FORTRAN ``(30i3)`` happily reads fewer
    fields than the maximum.
    """
    m = header.model_options
    fields = [
        m.inhyd, m.imphys, m.icupa, m.ibltyp,
        m.ifrad, m.isoil, m.ifddan, m.ifddaob,
        *m.flags_2d,        # 12 entries
        m.nland,
    ]
    out.write("".join(fmt_i(v, 3) for v in fields) + _LINE_END)


def _write_header_time_window(out: IO[str], header: Header) -> None:
    """Format(i4, 3i2, i5, 3i4).

    Date components use i2.2 (zero-padded) so the leading column reads
    as the concatenated YYYYMMDDHH integer that CALMM5/CALWRF emit.
    """
    t = header.time_window
    out.write(
        fmt_i(t.ibyrm, 4, min_digits=4)
        + fmt_i(t.ibmom, 2, min_digits=2)
        + fmt_i(t.ibdym, 2, min_digits=2)
        + fmt_i(t.ibhrm, 2, min_digits=2)
        + fmt_i(t.nhrsmm5, 5)
        + fmt_i(t.nxp, 4)
        + fmt_i(t.nyp, 4)
        + fmt_i(t.nzp, 4)
        + _LINE_END
    )


def _write_header_extraction(out: IO[str], header: Header) -> None:
    """Format(6i4, 2f10.4, 2f9.4)."""
    e = header.extraction
    out.write(
        fmt_i(e.nx1, 4)
        + fmt_i(e.ny1, 4)
        + fmt_i(e.nx2, 4)
        + fmt_i(e.ny2, 4)
        + fmt_i(e.nz1, 4)
        + fmt_i(e.nz2, 4)
        + fmt_f(e.rxmin, 10, 4)
        + fmt_f(e.rxmax, 10, 4)
        + fmt_f(e.rymin, 9, 4)
        + fmt_f(e.rymax, 9, 4)
        + _LINE_END
    )


def _write_header_sigma_levels(out: IO[str], header: Header) -> None:
    """One Format(F6.3) record per sigma level (NZP records total)."""
    for sigma in header.sigma_levels:
        out.write(fmt_f(sigma, 6, 3) + _LINE_END)


def _write_header_grid_points(out: IO[str], header: Header) -> None:
    """Format(2i4, f9.4, f10.4, i5, i3, 1x, f9.4, f10.4, i5).
    NXP*NYP records total."""
    for g in header.grid_points:
        out.write(
            fmt_i(g.iindex, 4)
            + fmt_i(g.jindex, 4)
            + fmt_f(g.xlat_dot, 9, 4)
            + fmt_f(g.xlong_dot, 10, 4)
            + fmt_i(g.ielev_dot, 5)
            + fmt_i(g.iland, 3)
            + fmt_x(1)
            + fmt_f(g.xlat_crs, 9, 4)
            + fmt_f(g.xlong_crs, 10, 4)
            + fmt_i(g.ielev_crs, 5)
            + _LINE_END
        )


# ---------------------------------------------------------------------------
# Data record writers
# ---------------------------------------------------------------------------


def _write_surface_record(out: IO[str], s: SurfaceRecord) -> None:
    """Format(i4, 3i2, 2i3, f7.1, f5.2, i2, 3f8.1, f8.2, 3f8.1)."""
    out.write(
        fmt_i(s.year, 4, min_digits=4)
        + fmt_i(s.month, 2, min_digits=2)
        + fmt_i(s.day, 2, min_digits=2)
        + fmt_i(s.hour, 2, min_digits=2)
        + fmt_i(s.ix, 3)
        + fmt_i(s.jx, 3)
        + fmt_f(s.pres, 7, 1)
        + fmt_f(s.rain, 5, 2)
        + fmt_i(s.sc, 2)
        + fmt_f(s.radsw, 8, 1)
        + fmt_f(s.radlw, 8, 1)
        + fmt_f(s.t2, 8, 1)
        + fmt_f(s.q2, 8, 2)
        + fmt_f(s.wd10, 8, 1)
        + fmt_f(s.ws10, 8, 1)
        + fmt_f(s.sst, 8, 1)
        + _LINE_END
    )


def _vertical_hydrometeors(
    v: VerticalRecord, flags: OutputFlags
) -> tuple[float, ...]:
    """Collect the hydrometeor mixing ratios that should be written.

    Returns them in spec order: CLDMR, RAINMR, ICEMR, SNOWMR, GRPMR
    filtered by the IOUTC / IOUTI / IOUTG flags.
    """
    ratios: list[float] = []
    if flags.ioutc:
        ratios.extend([v.cldmr, v.rainmr])
    if flags.iouti:
        ratios.extend([v.icemr, v.snowmr])
    if flags.ioutg:
        ratios.append(v.grpmr)
    return tuple(ratios)


def _write_vertical_record(
    out: IO[str], v: VerticalRecord, flags: OutputFlags
) -> None:
    """Format(i4, i6, f6.1, i4, f5.1, f6.2, i3, f5.2, 5f6.3) with optional fields."""
    parts: list[str] = [
        fmt_i(v.pres, 4),
        fmt_i(v.z, 6),
        fmt_f(v.tempk, 6, 1),
        fmt_i(v.wd, 4),
        fmt_f(v.ws, 5, 1),
    ]
    if flags.ioutw:
        parts.append(fmt_f(v.w, 6, 2))
    if flags.ioutq:
        parts.append(fmt_i(v.rh, 3))
        parts.append(fmt_f(v.vapmr, 5, 2))

    ratios = _vertical_hydrometeors(v, flags)
    if ratios:
        # CALWRF v2.0.3 clamps negatives to zero before the compression
        # check and treats |x| < xzero (0.00049) as zero. Compression is
        # also skipped when only one ratio is emitted (a CALWRF edge case
        # that should not happen with our enforced flag dependencies, but
        # we defend against it for byte-identical output).
        clamped = tuple(max(r, 0.0) for r in ratios)
        all_zero = all(r < _HYDRO_ZERO_THRESHOLD for r in clamped)
        if all_zero and len(clamped) > 1:
            parts.append(fmt_f(-float(len(clamped)), 6, 3))
        else:
            parts.extend(fmt_f(r, 6, 3) for r in clamped)

    out.write("".join(parts) + _LINE_END)


def _write_frame(
    out: IO[str], frame: Frame, header: Header
) -> None:
    """Write one timestep: NXP*NYP surface records, each followed by NZP vertical records."""
    expected_cells = header.time_window.nxp * header.time_window.nyp
    if len(frame.cells) != expected_cells:
        raise ValueError(
            f"frame has {len(frame.cells)} cells; expected NXP*NYP={expected_cells}"
        )

    nzp = header.time_window.nzp
    for cell in frame.cells:
        if len(cell.levels) != nzp:
            raise ValueError(
                f"cell ({cell.surface.ix},{cell.surface.jx}) has "
                f"{len(cell.levels)} levels; expected NZP={nzp}"
            )
        _write_surface_record(out, cell.surface)
        for v in cell.levels:
            _write_vertical_record(out, v, header.flags)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def write_3ddat(
    path: str | os.PathLike[str],
    header: Header,
    frames: Frames,
) -> int:
    """Write a complete 3D.DAT v2.1 file. Returns the number of frames written.

    ``frames`` is iterated once; each Frame must contain exactly
    ``NXP*NYP`` cells in (JX-outer, IX-inner) order, and each cell must
    contain exactly ``NZP`` vertical records.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(p, "w", encoding="ascii", newline="") as out:
        _write_header_record_1(out, header)
        _write_header_comments(out, header)
        _write_header_flags(out, header.flags)
        _write_header_projection(out, header)
        _write_header_model_options(out, header)
        _write_header_time_window(out, header)
        _write_header_extraction(out, header)
        _write_header_sigma_levels(out, header)
        _write_header_grid_points(out, header)
        for frame in frames:
            _write_frame(out, frame, header)
            n_written += 1
    return n_written
