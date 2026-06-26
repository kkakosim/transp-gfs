"""Tests for the CLI dispatch path.

We mock the pipeline stages (download / decode / regrid / write) to
assert the CLI:
    * loads the YAML config and feeds the right arguments to each stage,
    * streams one file at a time through the decode/regrid/frame-build
      stages (per the OOM fix — decoding all GFS files at once needs
      tens of GB of RAM),
    * skips the download when ``--skip-download`` is set,
    * returns exit code 0 on success and propagates errors otherwise.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from gfs2calmet import cli
from gfs2calmet.dataset import (
    Comment, Extraction, GridDomain, GridPoint, Header, ModelOptions,
    OutputFlags, Projection, TimeWindow,
)


_VALID_YAML = dedent("""\
    target_grid:
      crs: "+proj=utm +zone=39 +ellps=WGS84 +units=m"
      x0_km: 400.0
      y0_km: 2700.0
      dx_km: 4.0
      dy_km: 4.0
      nx: 5
      ny: 5

    gfs:
      cycle: "2026-01-15T00:00"
      forecast_hours: [0, 3, 6]
      pressure_levels: [1000, 850, 500]
      product: pgrb2.0p25
      model: gfs
      output_dir: ./data/grib

    output_flags:
      ioutw: 0
      ioutq: 1
      ioutc: 0
      iouti: 0
      ioutg: 0
      iosrf: 0

    header:
      maptxt: UTM
      nland: 38

    frame: {}

    output_path: ./out/test.3D.DAT
""")


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(_VALID_YAML, encoding="utf-8")
    return cfg_path


def _stub_header() -> Header:
    """Build a real (but trivial) Header so the CLI's
    ``dataclasses.replace(header.time_window, ...)`` patch works."""
    return Header(
        dataset_name="3D.DAT",
        dataset_version="2.1",
        dataset_message="test",
        comments=[Comment("c")],
        flags=OutputFlags(ioutw=0, ioutq=1, ioutc=0, iouti=0, ioutg=0, iosrf=0),
        projection=Projection(maptxt="UTM", rlatc=0.0, rlonc=0.0,
                              truelat1=0.0, truelat2=0.0,
                              x1dmn=0.0, y1dmn=0.0, dxy=4.0),
        domain=GridDomain(nx=1, ny=1, nz=1),
        model_options=ModelOptions(nland=38),
        time_window=TimeWindow(
            ibyrm=2026, ibmom=1, ibdym=15, ibhrm=0,
            nhrsmm5=1, nxp=1, nyp=1, nzp=1,
        ),
        extraction=Extraction(
            nx1=1, ny1=1, nx2=1, ny2=1, nz1=1, nz2=1,
            rxmin=50.0, rxmax=50.0, rymin=24.0, rymax=24.0,
        ),
        sigma_levels=[0.99],
        grid_points=[GridPoint(
            iindex=1, jindex=1, xlat_dot=24.0, xlong_dot=50.0,
            ielev_dot=0, iland=16, xlat_crs=-999.0, xlong_crs=-999.0,
            ielev_crs=-999,
        )],
    )


def _stub_regridded(levels=(1000, 850, 500)) -> xr.Dataset:
    """Tiny xarray Dataset matching the shape regrid_dataset returns.

    The CLI's reconciliation step looks at ``ds["level"].values``, so
    a real Dataset with a level coord is the minimum needed.
    """
    return xr.Dataset(
        coords={"level": ("level", np.array(list(levels), dtype=np.int32))},
    )


def _consume_generator_in_write(*args, **kwargs):
    """side_effect for write_3ddat that drains the frame generator so
    every per-file decode/regrid/build_frames call actually happens."""
    _path, _header, frames = args
    count = 0
    for _ in frames:
        count += 1
    return count


# ---------------------------------------------------------------------------
# Default run (download + streaming decode/regrid/build → write)
# ---------------------------------------------------------------------------


def test_cli_streams_decode_regrid_build_once_per_file(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)

    with patch.object(cli, "download_gfs_cycle") as mock_dl, \
         patch.object(cli, "read_gfs_to_dataset") as mock_read, \
         patch.object(cli, "regrid_dataset") as mock_regrid, \
         patch.object(cli, "build_header") as mock_header, \
         patch.object(cli, "build_frames") as mock_frames, \
         patch.object(cli, "write_3ddat",
                      side_effect=_consume_generator_in_write) as mock_write:

        mock_dl.return_value = [tmp_path / "f000.grib2",
                                tmp_path / "f003.grib2",
                                tmp_path / "f006.grib2"]
        mock_read.return_value = "src_ds"
        mock_regrid.return_value = _stub_regridded()
        mock_header.return_value = _stub_header()
        mock_frames.return_value = ["F"]

        rc = cli.main([str(cfg_path)])

    assert rc == 0
    mock_dl.assert_called_once()
    # Decode + regrid happen once per file (3 files in the config).
    assert mock_read.call_count == 3
    assert mock_regrid.call_count == 3
    # Header is built once, from the first file's regridded dataset.
    mock_header.assert_called_once()
    # build_frames is called once per file (the generator yields from each).
    assert mock_frames.call_count == 3
    mock_write.assert_called_once()


def test_header_nhrsmm5_patched_to_total_file_count(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    captured = {}

    def capture_write(path, header, frames):
        captured["header"] = header
        for _ in frames:
            pass
        return 0

    with patch.object(cli, "download_gfs_cycle",
                      return_value=[tmp_path / f"f{h:03d}.grib2"
                                    for h in (0, 3, 6)]), \
         patch.object(cli, "read_gfs_to_dataset", return_value="src"), \
         patch.object(cli, "regrid_dataset", return_value=_stub_regridded()), \
         patch.object(cli, "build_header", return_value=_stub_header()), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat", side_effect=capture_write):
        cli.main([str(cfg_path)])

    # 3 files → nhrsmm5 = max(3-1, 1) = 2.
    assert captured["header"].time_window.nhrsmm5 == 2


def test_download_called_with_cycle_and_fxx_from_config(
    tmp_path: Path,
) -> None:
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle") as mock_dl, \
         patch.object(cli, "read_gfs_to_dataset", return_value="src"), \
         patch.object(cli, "regrid_dataset", return_value=_stub_regridded()), \
         patch.object(cli, "build_header", return_value=_stub_header()), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat",
                      side_effect=_consume_generator_in_write):
        mock_dl.return_value = [tmp_path / "f.grib2"]
        cli.main([str(cfg_path)])

    kwargs = mock_dl.call_args.kwargs
    assert kwargs["cycle"] == datetime(2026, 1, 15, 0, 0)
    assert kwargs["fxx_hours"] == [0, 3, 6]
    assert kwargs["model"] == "gfs"
    assert kwargs["product"] == "pgrb2.0p25"
    assert "t_pl" in kwargs["roles"]
    assert "mslp" in kwargs["roles"]


def test_write_path_matches_config(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle",
                      return_value=[tmp_path / "f.grib2"]), \
         patch.object(cli, "read_gfs_to_dataset", return_value="src"), \
         patch.object(cli, "regrid_dataset", return_value=_stub_regridded()), \
         patch.object(cli, "build_header", return_value=_stub_header()), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat",
                      side_effect=_consume_generator_in_write) as mock_write:
        cli.main([str(cfg_path)])
    assert mock_write.call_args.args[0] == "./out/test.3D.DAT"


def test_unavailable_pressure_levels_are_dropped_with_warning(
    tmp_path: Path, caplog
) -> None:
    """The user's config asks for [1000, 850, 500] but the source file
    only carries [1000, 500] (no 850). The CLI should warn, drop 850
    from header + frame options, and continue."""
    cfg_path = _write_config(tmp_path)
    available_levels = (1000, 500)  # 850 missing

    captured = {}

    def capture_cfg(cfg, target, opts):
        captured["pressure_levels"] = list(opts.pressure_levels)
        return _stub_header()

    with patch.object(cli, "download_gfs_cycle",
                      return_value=[tmp_path / "f.grib2"]), \
         patch.object(cli, "read_gfs_to_dataset", return_value="src"), \
         patch.object(cli, "regrid_dataset",
                      return_value=_stub_regridded(levels=available_levels)), \
         patch.object(cli, "build_header", side_effect=capture_cfg), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat",
                      side_effect=_consume_generator_in_write):
        with caplog.at_level("WARNING", logger="gfs2calmet"):
            rc = cli.main([str(cfg_path)])

    assert rc == 0
    assert captured["pressure_levels"] == [1000, 500]
    assert any("850" in rec.message and "not in GFS" in rec.message
               for rec in caplog.records)


def test_all_pressure_levels_unavailable_raises(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle",
                      return_value=[tmp_path / "f.grib2"]), \
         patch.object(cli, "read_gfs_to_dataset", return_value="src"), \
         patch.object(cli, "regrid_dataset",
                      return_value=_stub_regridded(levels=(925, 700))), \
         patch.object(cli, "build_header", return_value=_stub_header()), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat",
                      side_effect=_consume_generator_in_write):
        # Config requests [1000, 850, 500]; available is [925, 700] → empty.
        with pytest.raises(FileNotFoundError, match="None of the requested"):
            cli.main([str(cfg_path)])


def test_empty_grib_paths_raises(tmp_path: Path) -> None:
    """If Herbie returns an empty list the CLI must error out before
    attempting to build a header from nothing."""
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle", return_value=[]):
        with pytest.raises(FileNotFoundError, match="no GRIB2 files"):
            cli.main([str(cfg_path)])


# ---------------------------------------------------------------------------
# --skip-download
# ---------------------------------------------------------------------------


def test_skip_download_reuses_existing_grib_files(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    grib_dir = tmp_path / "data" / "grib"
    grib_dir.mkdir(parents=True)
    (grib_dir / "a.grib2").write_bytes(b"")
    (grib_dir / "b.grib2").write_bytes(b"")

    cwd_was = Path.cwd()
    try:
        import os as _os
        _os.chdir(tmp_path)
        with patch.object(cli, "download_gfs_cycle") as mock_dl, \
             patch.object(cli, "read_gfs_to_dataset", return_value="src"), \
             patch.object(cli, "regrid_dataset", return_value=_stub_regridded()), \
             patch.object(cli, "build_header", return_value=_stub_header()), \
             patch.object(cli, "build_frames", return_value=[]), \
             patch.object(cli, "write_3ddat",
                          side_effect=_consume_generator_in_write):
            rc = cli.main([str(cfg_path), "--skip-download"])
    finally:
        import os as _os
        _os.chdir(str(cwd_was))

    assert rc == 0
    mock_dl.assert_not_called()


def test_skip_download_with_missing_dir_raises(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    cwd_was = Path.cwd()
    try:
        import os as _os
        _os.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            cli.main([str(cfg_path), "--skip-download"])
    finally:
        import os as _os
        _os.chdir(str(cwd_was))


# ---------------------------------------------------------------------------
# Argparse surface
# ---------------------------------------------------------------------------


def test_missing_config_argument_exits_with_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_nonexistent_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli.main([str(tmp_path / "does-not-exist.yaml")])


def test_verbose_flag_sets_logging(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle",
                      return_value=[tmp_path / "f.grib2"]), \
         patch.object(cli, "read_gfs_to_dataset", return_value="src"), \
         patch.object(cli, "regrid_dataset", return_value=_stub_regridded()), \
         patch.object(cli, "build_header", return_value=_stub_header()), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat",
                      side_effect=_consume_generator_in_write), \
         patch.object(cli, "_configure_logging") as mock_log:
        cli.main([str(cfg_path), "-vv"])
    mock_log.assert_called_once_with(2)
