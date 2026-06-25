"""Round-trip test for the 3D.DAT writer.

Strategy: build a synthetic 5x5 grid, 1 timestep, 4 vertical levels
with deterministic but distinguishable values, write the file, then
parse it back by slicing each record into the exact column positions
implied by the spec's FORTRAN FORMAT strings. This both verifies the
writer's byte-level correctness and pins the format so any drift is
caught by CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gfs2calmet import (
    CellData,
    Comment,
    Extraction,
    Frame,
    GridDomain,
    GridPoint,
    Header,
    ModelOptions,
    OutputFlags,
    Projection,
    SurfaceRecord,
    TimeWindow,
    VerticalRecord,
    write_3ddat,
)


# ---------------------------------------------------------------------------
# Synthetic test domain
# ---------------------------------------------------------------------------


NXP, NYP, NZP = 5, 5, 4

# Pressure levels (hPa) for the 4 synthetic vertical levels.
LEVELS_HPA = (1000, 925, 850, 700)
# Approximate heights (m MSL) corresponding to those levels.
LEVELS_Z_M = (110, 760, 1500, 3000)


def _make_grid_points() -> list[GridPoint]:
    """Synthetic grid: 5x5 dot points at 0.04-deg spacing."""
    pts: list[GridPoint] = []
    for j in range(1, NYP + 1):
        for i in range(1, NXP + 1):
            lat = 24.40 + 0.04 * (j - 1)
            lon = 50.50 + 0.04 * (i - 1)
            pts.append(
                GridPoint(
                    iindex=i,
                    jindex=j,
                    xlat_dot=lat,
                    xlong_dot=lon,
                    ielev_dot=10 + i + j,
                    iland=16,                # USGS "water" category, arbitrary
                    xlat_crs=lat + 0.02,
                    xlong_crs=lon + 0.02,
                    ielev_crs=12 + i + j,
                )
            )
    return pts


def _make_header(
    flags: OutputFlags,
    sigma: tuple[float, ...] = (0.99, 0.95, 0.88, 0.75),
) -> Header:
    return Header(
        dataset_name="3D.DAT",
        dataset_version="2.1",
        dataset_message="Synthetic test fixture",
        comments=[
            Comment("Produced by gfs2calmet test suite"),
            Comment("Domain: 5x5 synthetic, 4 levels, 1 timestep"),
        ],
        flags=flags,
        projection=Projection(
            maptxt="MER",
            rlatc=25.30,
            rlonc=51.20,
            truelat1=0.0,
            truelat2=0.0,
            x1dmn=0.0,
            y1dmn=0.0,
            dxy=4.0,
        ),
        domain=GridDomain(nx=NXP, ny=NYP, nz=NZP),
        model_options=ModelOptions(nland=38),
        time_window=TimeWindow(
            ibyrm=2026, ibmom=1, ibdym=15, ibhrm=0,
            nhrsmm5=1, nxp=NXP, nyp=NYP, nzp=NZP,
        ),
        extraction=Extraction(
            nx1=1, ny1=1, nx2=NXP, ny2=NYP, nz1=1, nz2=NZP,
            rxmin=50.50, rxmax=50.66, rymin=24.40, rymax=24.56,
        ),
        sigma_levels=list(sigma),
        grid_points=_make_grid_points(),
    )


def _make_frame(
    year: int = 2026, month: int = 1, day: int = 15, hour: int = 0,
    nonzero_hydrometeors: bool = False,
) -> Frame:
    """Build one Frame with NXP*NYP cells in (JX-outer, IX-inner) order."""
    cells: list[CellData] = []
    for j in range(1, NYP + 1):
        for i in range(1, NXP + 1):
            surface = SurfaceRecord(
                year=year, month=month, day=day, hour=hour,
                ix=i, jx=j,
                pres=1012.3,
                rain=0.0,
                sc=0,
                radsw=250.0,
                radlw=400.0,
                t2=295.0 + 0.1 * i,
                q2=14.5,
                wd10=180.0,
                ws10=3.5,
                sst=297.0,
            )
            levels: list[VerticalRecord] = []
            for k, (p_hpa, z_m) in enumerate(zip(LEVELS_HPA, LEVELS_Z_M)):
                # Hydrometeors: zero everywhere (compression case) unless
                # the test explicitly opts in to nonzero values at level 0.
                cldmr = 0.05 if (nonzero_hydrometeors and k == 0) else 0.0
                levels.append(
                    VerticalRecord(
                        pres=p_hpa,
                        z=z_m,
                        tempk=295.0 - 5.0 * k,
                        wd=180 + k,
                        ws=3.0 + 0.5 * k,
                        w=-0.01,
                        rh=70 - 5 * k,
                        vapmr=14.5 - 1.0 * k,
                        cldmr=cldmr,
                    )
                )
            cells.append(CellData(surface=surface, levels=tuple(levels)))
    return Frame(cells=tuple(cells))


# ---------------------------------------------------------------------------
# Header parsing helpers (reverse of writer.py for round-trip verification)
# ---------------------------------------------------------------------------


def _parse_record_1(line: str) -> tuple[str, str, str]:
    """Format(2a16, a64)."""
    assert len(line) >= 32, line
    return line[0:16].rstrip(), line[16:32].rstrip(), line[32:96].rstrip()


def _parse_flags(line: str) -> tuple[int, int, int, int, int, int]:
    """Format(6i3)."""
    return tuple(int(line[i:i + 3]) for i in range(0, 18, 3))  # type: ignore[return-value]


def _parse_projection(line: str) -> dict:
    """Format(a4, f9.4, f10.4, 2f7.2, 2f10.3, f8.3, 2i4, i3)."""
    pos = 0
    out: dict = {}
    out["maptxt"] = line[pos:pos + 4].rstrip(); pos += 4
    out["rlatc"] = float(line[pos:pos + 9]); pos += 9
    out["rlonc"] = float(line[pos:pos + 10]); pos += 10
    out["truelat1"] = float(line[pos:pos + 7]); pos += 7
    out["truelat2"] = float(line[pos:pos + 7]); pos += 7
    out["x1dmn"] = float(line[pos:pos + 10]); pos += 10
    out["y1dmn"] = float(line[pos:pos + 10]); pos += 10
    out["dxy"] = float(line[pos:pos + 8]); pos += 8
    out["nx"] = int(line[pos:pos + 4]); pos += 4
    out["ny"] = int(line[pos:pos + 4]); pos += 4
    out["nz"] = int(line[pos:pos + 3]); pos += 3
    return out


def _parse_time_window(line: str) -> dict:
    """Format(i4, 3i2, i5, 3i4)."""
    return {
        "ibyrm": int(line[0:4]),
        "ibmom": int(line[4:6]),
        "ibdym": int(line[6:8]),
        "ibhrm": int(line[8:10]),
        "nhrsmm5": int(line[10:15]),
        "nxp": int(line[15:19]),
        "nyp": int(line[19:23]),
        "nzp": int(line[23:27]),
    }


def _parse_extraction(line: str) -> dict:
    """Format(6i4, 2f10.4, 2f9.4)."""
    return {
        "nx1": int(line[0:4]),
        "ny1": int(line[4:8]),
        "nx2": int(line[8:12]),
        "ny2": int(line[12:16]),
        "nz1": int(line[16:20]),
        "nz2": int(line[20:24]),
        "rxmin": float(line[24:34]),
        "rxmax": float(line[34:44]),
        "rymin": float(line[44:53]),
        "rymax": float(line[53:62]),
    }


def _parse_grid_point(line: str) -> dict:
    """Format(2i4, f9.4, f10.4, i5, i3, 1x, f9.4, f10.4, i5)."""
    return {
        "iindex": int(line[0:4]),
        "jindex": int(line[4:8]),
        "xlat_dot": float(line[8:17]),
        "xlong_dot": float(line[17:27]),
        "ielev_dot": int(line[27:32]),
        "iland": int(line[32:35]),
        # one space (1x) at position 35
        "xlat_crs": float(line[36:45]),
        "xlong_crs": float(line[45:55]),
        "ielev_crs": int(line[55:60]),
    }


def _parse_surface(line: str) -> dict:
    """Format(i4, 3i2, 2i3, f7.1, f5.2, i2, 3f8.1, f8.2, 3f8.1)."""
    return {
        "year": int(line[0:4]),
        "month": int(line[4:6]),
        "day": int(line[6:8]),
        "hour": int(line[8:10]),
        "ix": int(line[10:13]),
        "jx": int(line[13:16]),
        "pres": float(line[16:23]),
        "rain": float(line[23:28]),
        "sc": int(line[28:30]),
        "radsw": float(line[30:38]),
        "radlw": float(line[38:46]),
        "t2": float(line[46:54]),
        "q2": float(line[54:62]),
        "wd10": float(line[62:70]),
        "ws10": float(line[70:78]),
        "sst": float(line[78:86]),
    }


def _parse_vertical(line: str, flags: OutputFlags) -> dict:
    """Format(i4, i6, f6.1, i4, f5.1, [f6.2], [i3, f5.2], [hydrometeors])."""
    pos = 0
    out: dict = {
        "pres": int(line[pos:pos + 4]),
    }; pos += 4
    out["z"] = int(line[pos:pos + 6]); pos += 6
    out["tempk"] = float(line[pos:pos + 6]); pos += 6
    out["wd"] = int(line[pos:pos + 4]); pos += 4
    out["ws"] = float(line[pos:pos + 5]); pos += 5
    if flags.ioutw:
        out["w"] = float(line[pos:pos + 6]); pos += 6
    if flags.ioutq:
        out["rh"] = int(line[pos:pos + 3]); pos += 3
        out["vapmr"] = float(line[pos:pos + 5]); pos += 5

    n_h = flags.n_hydrometeors
    if n_h > 0:
        # Peek first F6.3: if it equals -float(n_h) exactly the rest of the
        # field is omitted (compression). Otherwise read n_h F6.3 ratios.
        first = float(line[pos:pos + 6])
        if first == -float(n_h):
            out["hydrometeors"] = (0.0,) * n_h
            out["_compressed"] = True
            pos += 6
        else:
            ratios = [first]
            pos += 6
            for _ in range(n_h - 1):
                ratios.append(float(line[pos:pos + 6]))
                pos += 6
            out["hydrometeors"] = tuple(ratios)
            out["_compressed"] = False
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_writes_header_record_1_with_exact_dataset_name_and_version(
    tmp_path: Path,
) -> None:
    flags = OutputFlags(ioutw=1, ioutq=1, ioutc=1, iouti=1, ioutg=0, iosrf=0)
    path = tmp_path / "synthetic.3D.DAT"

    n = write_3ddat(path, _make_header(flags), [_make_frame()])

    assert n == 1
    lines = path.read_text(encoding="ascii").splitlines()
    name, version, message = _parse_record_1(lines[0])
    assert name == "3D.DAT"
    assert version == "2.1"
    assert message == "Synthetic test fixture"


def test_full_header_round_trips(tmp_path: Path) -> None:
    flags = OutputFlags(ioutw=1, ioutq=1, ioutc=1, iouti=1, ioutg=0, iosrf=0)
    header = _make_header(flags)
    path = tmp_path / "synthetic.3D.DAT"
    write_3ddat(path, header, [_make_frame()])
    lines = path.read_text(encoding="ascii").splitlines()

    # Record 1
    name, version, _ = _parse_record_1(lines[0])
    assert name == header.dataset_name
    assert version == header.dataset_version

    # Comment count + comment lines
    ncomm = int(lines[1])
    assert ncomm == len(header.comments)
    for k, c in enumerate(header.comments):
        assert lines[2 + k].rstrip() == c.text

    # Output flags
    parsed_flags = _parse_flags(lines[2 + ncomm])
    assert parsed_flags == (1, 1, 1, 1, 0, 0)

    # Projection
    proj = _parse_projection(lines[3 + ncomm])
    assert proj["maptxt"] == "MER"
    assert proj["rlatc"] == pytest.approx(25.30)
    assert proj["rlonc"] == pytest.approx(51.20)
    assert proj["dxy"] == pytest.approx(4.0)
    assert proj["nx"] == NXP
    assert proj["ny"] == NYP
    assert proj["nz"] == NZP

    # Model options — first 8 zeros, 12 flag zeros, nland=38 → 21 i3 fields.
    opts_line = lines[4 + ncomm]
    opts = [int(opts_line[i:i + 3]) for i in range(0, 63, 3)]
    assert opts[:8] == [0] * 8
    assert opts[8:20] == [0] * 12
    assert opts[20] == 38

    # Time window
    tw = _parse_time_window(lines[5 + ncomm])
    assert tw == {
        "ibyrm": 2026, "ibmom": 1, "ibdym": 15, "ibhrm": 0,
        "nhrsmm5": 1, "nxp": NXP, "nyp": NYP, "nzp": NZP,
    }

    # Extraction
    ext = _parse_extraction(lines[6 + ncomm])
    assert ext["nx1"] == 1 and ext["nx2"] == NXP
    assert ext["ny1"] == 1 and ext["ny2"] == NYP
    assert ext["rxmin"] == pytest.approx(50.50)
    assert ext["rxmax"] == pytest.approx(50.66)

    # Sigma levels (NZP records of F6.3)
    sigma_start = 7 + ncomm
    for k, expected in enumerate(header.sigma_levels):
        assert float(lines[sigma_start + k]) == pytest.approx(expected)

    # Grid points (NXP*NYP records)
    gp_start = sigma_start + NZP
    for k, gp in enumerate(header.grid_points):
        parsed = _parse_grid_point(lines[gp_start + k])
        assert parsed["iindex"] == gp.iindex
        assert parsed["jindex"] == gp.jindex
        assert parsed["xlat_dot"] == pytest.approx(gp.xlat_dot)
        assert parsed["xlong_dot"] == pytest.approx(gp.xlong_dot)
        assert parsed["ielev_dot"] == gp.ielev_dot
        assert parsed["iland"] == gp.iland


def test_date_components_are_zero_padded_into_yyyymmddhh(tmp_path: Path) -> None:
    """Header time-window and surface records both encode the date as the
    concatenated YYYYMMDDHH integer using i2.2 for the components."""
    flags = OutputFlags(ioutw=0, ioutq=0, ioutc=0, iouti=0, ioutg=0, iosrf=0)
    path = tmp_path / "date.3D.DAT"
    write_3ddat(path, _make_header(flags), [_make_frame()])
    lines = path.read_text(encoding="ascii").splitlines()

    # Header layout with NCOMM=2 puts the time-window record at index 7:
    # 0 = record 1, 1 = NCOMM, 2..3 = comments, 4 = flags, 5 = projection,
    # 6 = model options, 7 = time window.
    tw_line = lines[7]
    assert tw_line.startswith("2026011500"), tw_line

    # First surface record. Surface format width is 86; the time-window
    # header line (also starts with "2026011500") is only 27 chars, so
    # we filter on length.
    surface_indices = [
        i for i, ln in enumerate(lines)
        if ln.startswith("2026011500") and len(ln) >= 80
    ]
    assert surface_indices, "no surface record found with concatenated date"
    first_surface = lines[surface_indices[0]]
    parsed = _parse_surface(first_surface)
    assert parsed["year"] == 2026
    assert parsed["month"] == 1
    assert parsed["day"] == 15
    assert parsed["hour"] == 0


def test_hydrometeor_compression_when_all_zero(tmp_path: Path) -> None:
    """All-zero ratios collapse to a single ``-N.000`` field."""
    flags = OutputFlags(ioutw=1, ioutq=1, ioutc=1, iouti=1, ioutg=0, iosrf=0)
    path = tmp_path / "compress.3D.DAT"
    write_3ddat(path, _make_header(flags), [_make_frame(nonzero_hydrometeors=False)])
    text = path.read_text(encoding="ascii")
    # n_h = 2 (cloud,rain) + 2 (ice,snow) = 4 → compression token "-4.000"
    assert flags.n_hydrometeors == 4
    assert "-4.000" in text
    # And the explicit non-compressed alternative must NOT appear.
    assert " 0.000 0.000 0.000 0.000" not in text


def test_hydrometeors_compress_with_subthreshold_floating_residuals(
    tmp_path: Path,
) -> None:
    """Per CALWRF v2.0.3 wrtcmp: |x| < 0.00049 counts as zero, so a
    grid of 1e-6 g/kg residuals still triggers compression."""
    flags = OutputFlags(ioutw=1, ioutq=1, ioutc=1, iouti=1, ioutg=0, iosrf=0)
    header = _make_header(flags)
    cells: list[CellData] = []
    for j in range(1, NYP + 1):
        for i in range(1, NXP + 1):
            s = SurfaceRecord(
                year=2026, month=1, day=15, hour=0, ix=i, jx=j,
                pres=1012.3, rain=0.0, sc=0, radsw=0.0, radlw=0.0,
                t2=0.0, q2=0.0, wd10=0.0, ws10=0.0, sst=0.0,
            )
            levels = tuple(
                VerticalRecord(
                    pres=LEVELS_HPA[k], z=LEVELS_Z_M[k],
                    tempk=290.0, wd=180, ws=2.0, w=0.0, rh=50, vapmr=5.0,
                    cldmr=1e-6, rainmr=-2e-7, icemr=0.0, snowmr=1e-5,
                )
                for k in range(NZP)
            )
            cells.append(CellData(surface=s, levels=levels))
    path = tmp_path / "subthresh.3D.DAT"
    write_3ddat(path, header, [Frame(cells=tuple(cells))])
    text = path.read_text(encoding="ascii")
    # Tiny residuals must still compress.
    assert "-4.000" in text


def test_hydrometeors_expanded_when_any_nonzero(tmp_path: Path) -> None:
    """Any nonzero ratio forces all N fields to be written explicitly."""
    flags = OutputFlags(ioutw=1, ioutq=1, ioutc=1, iouti=1, ioutg=0, iosrf=0)
    path = tmp_path / "expand.3D.DAT"
    write_3ddat(path, _make_header(flags), [_make_frame(nonzero_hydrometeors=True)])
    text = path.read_text(encoding="ascii")
    # Level 0 of every cell carries cldmr=0.05; the explicit form must appear.
    assert " 0.050 0.000 0.000 0.000" in text


def test_optional_fields_omitted_when_flags_clear(tmp_path: Path) -> None:
    """With every optional flag off, vertical records carry only the
    mandatory 5 fields → line length 4+6+6+4+5 = 25."""
    flags = OutputFlags(ioutw=0, ioutq=0, ioutc=0, iouti=0, ioutg=0, iosrf=0)
    path = tmp_path / "minimal.3D.DAT"
    write_3ddat(path, _make_header(flags), [_make_frame()])
    lines = path.read_text(encoding="ascii").splitlines()

    # Pick out a vertical line: it starts at a column where the first 4
    # chars parse as a pressure level (1000, 925, 850, or 700).
    vertical_lines = [ln for ln in lines if ln[:4].strip().isdigit()
                      and int(ln[:4]) in (1000, 925, 850, 700)
                      and len(ln) <= 30]
    assert vertical_lines, "no minimal vertical lines found"
    for ln in vertical_lines:
        assert len(ln) == 25, f"unexpected width {len(ln)}: {ln!r}"


def test_frame_with_wrong_cell_count_raises(tmp_path: Path) -> None:
    flags = OutputFlags(ioutw=1, ioutq=1, ioutc=0, iouti=0, ioutg=0, iosrf=0)
    header = _make_header(flags)
    short_frame = Frame(cells=tuple(_make_frame().cells[:5]))
    with pytest.raises(ValueError, match="expected NXP\\*NYP=25"):
        write_3ddat(tmp_path / "bad.dat", header, [short_frame])


def test_cell_with_wrong_level_count_raises(tmp_path: Path) -> None:
    flags = OutputFlags(ioutw=1, ioutq=1, ioutc=0, iouti=0, ioutg=0, iosrf=0)
    header = _make_header(flags)
    # Build a frame whose first cell only has 2 vertical levels.
    good = _make_frame()
    broken_cells = list(good.cells)
    broken_cells[0] = CellData(
        surface=good.cells[0].surface,
        levels=good.cells[0].levels[:2],
    )
    bad_frame = Frame(cells=tuple(broken_cells))
    with pytest.raises(ValueError, match="expected NZP=4"):
        write_3ddat(tmp_path / "bad.dat", header, [bad_frame])


def test_outputflags_enforces_dependencies() -> None:
    # IOUTI requires IOUTC
    with pytest.raises(ValueError, match="IOUTI=1 requires IOUTC=1"):
        OutputFlags(ioutw=0, ioutq=1, ioutc=0, iouti=1, ioutg=0, iosrf=0)
    # IOUTG requires IOUTI
    with pytest.raises(ValueError, match="IOUTG=1 requires IOUTI=1"):
        OutputFlags(ioutw=0, ioutq=1, ioutc=1, iouti=0, ioutg=1, iosrf=0)


def test_returns_number_of_frames_written(tmp_path: Path) -> None:
    flags = OutputFlags(ioutw=0, ioutq=0, ioutc=0, iouti=0, ioutg=0, iosrf=0)
    path = tmp_path / "multi.3D.DAT"
    frames = [
        _make_frame(hour=h)
        for h in (0, 1, 2)
    ]
    # Header still claims nhrsmm5=1; the writer doesn't enforce that
    # because the period length can lag the actual frame count in some
    # workflows. We only verify the count returned matches.
    header = _make_header(flags)
    n = write_3ddat(path, header, frames)
    assert n == 3
