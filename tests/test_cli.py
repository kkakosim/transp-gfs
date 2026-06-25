"""Tests for the CLI dispatch path.

We mock the four pipeline stages (download → decode → regrid → write)
to assert the CLI:
    * loads the YAML config and feeds the right arguments to each stage,
    * skips the download when ``--skip-download`` is set,
    * returns exit code 0 on success and propagates errors otherwise.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from gfs2calmet import cli


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


# ---------------------------------------------------------------------------
# Default run (download + decode + regrid + write)
# ---------------------------------------------------------------------------


def test_cli_dispatches_through_all_four_stages(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)

    with patch.object(cli, "download_gfs_cycle") as mock_dl, \
         patch.object(cli, "read_gfs_to_dataset") as mock_read, \
         patch.object(cli, "regrid_dataset") as mock_regrid, \
         patch.object(cli, "build_header") as mock_header, \
         patch.object(cli, "build_frames") as mock_frames, \
         patch.object(cli, "write_3ddat") as mock_write:

        # Make each mock return something sensible for the chain to flow.
        mock_dl.return_value = [tmp_path / "f000.grib2",
                                tmp_path / "f003.grib2",
                                tmp_path / "f006.grib2"]
        mock_read.return_value = "src_ds"  # opaque; cli only forwards it
        regridded = type("DS", (), {"sizes": {"time": 3}})()
        mock_regrid.return_value = regridded
        mock_header.return_value = "HEADER"
        mock_frames.return_value = ["F0", "F1", "F2"]
        mock_write.return_value = 3

        rc = cli.main([str(cfg_path)])

    assert rc == 0
    mock_dl.assert_called_once()
    mock_read.assert_called_once()
    mock_regrid.assert_called_once_with(
        "src_ds", mock_regrid.call_args.args[1]
    )
    mock_header.assert_called_once()
    mock_frames.assert_called_once()
    mock_write.assert_called_once()


def test_download_called_with_cycle_and_fxx_from_config(
    tmp_path: Path,
) -> None:
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle") as mock_dl, \
         patch.object(cli, "read_gfs_to_dataset"), \
         patch.object(cli, "regrid_dataset") as mock_regrid, \
         patch.object(cli, "build_header"), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat", return_value=0):
        mock_dl.return_value = [tmp_path / "f.grib2"]
        mock_regrid.return_value = type("DS", (), {"sizes": {"time": 0}})()
        cli.main([str(cfg_path)])

    kwargs = mock_dl.call_args.kwargs
    assert kwargs["cycle"] == datetime(2026, 1, 15, 0, 0)
    assert kwargs["fxx_hours"] == [0, 3, 6]
    assert kwargs["model"] == "gfs"
    assert kwargs["product"] == "pgrb2.0p25"
    # Roles cover the full DEFAULT_GFS_FIELDS catalog.
    assert "t_pl" in kwargs["roles"]
    assert "mslp" in kwargs["roles"]


def test_write_path_matches_config(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle", return_value=[]), \
         patch.object(cli, "read_gfs_to_dataset"), \
         patch.object(cli, "regrid_dataset",
                      return_value=type("DS", (), {"sizes": {"time": 0}})()), \
         patch.object(cli, "build_header"), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat", return_value=0) as mock_write:
        cli.main([str(cfg_path)])
    assert mock_write.call_args.args[0] == "./out/test.3D.DAT"


# ---------------------------------------------------------------------------
# --skip-download
# ---------------------------------------------------------------------------


def test_skip_download_reuses_existing_grib_files(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    # Place stub GRIB2 files into a directory matching gfs.output_dir.
    grib_dir = tmp_path / "data" / "grib"
    grib_dir.mkdir(parents=True)
    (grib_dir / "a.grib2").write_bytes(b"")
    (grib_dir / "b.grib2").write_bytes(b"")

    # Make the relative gfs.output_dir resolve to our tmp_path.
    cwd_was = Path.cwd()
    try:
        import os as _os
        _os.chdir(tmp_path)
        with patch.object(cli, "download_gfs_cycle") as mock_dl, \
             patch.object(cli, "read_gfs_to_dataset"), \
             patch.object(cli, "regrid_dataset",
                          return_value=type("DS", (), {"sizes": {"time": 0}})()), \
             patch.object(cli, "build_header"), \
             patch.object(cli, "build_frames", return_value=[]), \
             patch.object(cli, "write_3ddat", return_value=0):
            rc = cli.main([str(cfg_path), "--skip-download"])
    finally:
        import os as _os
        _os.chdir(str(cwd_was))

    assert rc == 0
    mock_dl.assert_not_called()


def test_skip_download_with_missing_dir_raises(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    # output_dir in the config is "./data/grib" — does not exist under tmp_path.
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
    # argparse exits with code 2 on usage errors.
    assert exc.value.code == 2


def test_nonexistent_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli.main([str(tmp_path / "does-not-exist.yaml")])


def test_verbose_flag_sets_logging(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    with patch.object(cli, "download_gfs_cycle", return_value=[]), \
         patch.object(cli, "read_gfs_to_dataset"), \
         patch.object(cli, "regrid_dataset",
                      return_value=type("DS", (), {"sizes": {"time": 0}})()), \
         patch.object(cli, "build_header"), \
         patch.object(cli, "build_frames", return_value=[]), \
         patch.object(cli, "write_3ddat", return_value=0), \
         patch.object(cli, "_configure_logging") as mock_log:
        cli.main([str(cfg_path), "-vv"])
    mock_log.assert_called_once_with(2)
