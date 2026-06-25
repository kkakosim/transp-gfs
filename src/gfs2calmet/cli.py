"""Command-line entry point: GFS forecast -> CALMET 3D.DAT.

Usage::

    python -m gfs2calmet path/to/run.yaml [--skip-download]

The CLI ties together the four pipeline stages:

    1. Herbie subset-download of GFS GRIB2 for the requested cycle.
    2. pygrib decode → xarray Dataset (in target units).
    3. Bilinear regrid onto the configured CALMET driver grid.
    4. Build Header + Frames and write the 3D.DAT.

``--skip-download`` reuses already-downloaded GRIB2 files in
``gfs.output_dir`` — useful when iterating on grid or output-flag
changes without paying the bandwidth cost each time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from gfs2calmet.config import RunConfig, load_config
from gfs2calmet.frames import build_frames, build_header
from gfs2calmet.gfs_fields import DEFAULT_GFS_FIELDS
from gfs2calmet.gfs_reader import download_gfs_cycle, read_gfs_to_dataset
from gfs2calmet.regrid import regrid_dataset
from gfs2calmet.writer import write_3ddat


_LOG = logging.getLogger("gfs2calmet")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gfs2calmet",
        description="GFS forecast -> CALMET 3D.DAT prognostic input",
    )
    p.add_argument(
        "config",
        type=Path,
        help="Path to a run YAML (see config_example.yaml)",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Herbie download; reuse existing GRIB2 files in gfs.output_dir",
    )
    p.add_argument(
        "-v", "--verbose",
        action="count", default=0,
        help="Increase log verbosity (-v info, -vv debug)",
    )
    return p.parse_args(argv)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _existing_grib_paths(output_dir: str | None) -> list[Path]:
    if output_dir is None:
        raise ValueError("--skip-download requires gfs.output_dir to be set")
    d = Path(output_dir)
    if not d.is_dir():
        raise FileNotFoundError(d)
    paths = sorted(d.glob("*.grib2")) + sorted(d.glob("*.grb2"))
    if not paths:
        raise FileNotFoundError(f"no GRIB2 files found in {d}")
    return paths


def _run(cfg: RunConfig, *, skip_download: bool) -> int:
    _LOG.info("Building 3D.DAT for cycle %s", cfg.gfs.cycle)

    if skip_download:
        grib_paths = _existing_grib_paths(cfg.gfs.output_dir)
        _LOG.info("Reusing %d existing GRIB2 files from %s",
                  len(grib_paths), cfg.gfs.output_dir)
    else:
        _LOG.info("Downloading %d forecast hours via Herbie",
                  len(cfg.gfs.forecast_hours))
        roles = [f.role for f in DEFAULT_GFS_FIELDS]
        grib_paths = download_gfs_cycle(
            cycle=cfg.gfs.cycle,
            fxx_hours=cfg.gfs.forecast_hours,
            roles=roles,
            output_dir=cfg.gfs.output_dir,
            model=cfg.gfs.model,
            product=cfg.gfs.product,
        )

    _LOG.info("Decoding GRIB2 (%d files) with pygrib", len(grib_paths))
    src_ds = read_gfs_to_dataset(
        grib_paths,
        fields=DEFAULT_GFS_FIELDS,
        levels=cfg.gfs.pressure_levels,
    )

    _LOG.info("Regridding to %d x %d cells in %s",
              cfg.target_grid.nx, cfg.target_grid.ny, cfg.target_grid.crs)
    tgt_ds = regrid_dataset(src_ds, cfg.target_grid)

    _LOG.info("Building header and %d frames", tgt_ds.sizes["time"])
    header = build_header(tgt_ds, cfg.target_grid, cfg.header)
    frames = build_frames(tgt_ds, cfg.frame)

    _LOG.info("Writing %s", cfg.output_path)
    n = write_3ddat(cfg.output_path, header, frames)
    _LOG.info("Done. Wrote %d frames.", n)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    cfg = load_config(args.config)
    return _run(cfg, skip_download=args.skip_download)


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
