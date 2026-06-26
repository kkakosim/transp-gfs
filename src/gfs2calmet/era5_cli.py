"""ERA5 reanalysis → CALMET 3D.DAT pipeline.

Usage::

    python -m gfs2calmet.era5 path/to/era5_run.yaml [--skip-download] [-v]

Mirrors the GFS pipeline (cli.py) but sources data from the Copernicus
CDS (ERA5) instead of NCEP/GFS.  The regrid → build_frames → write_3ddat
stages are shared and unchanged.

Prerequisites:
    pip install cdsapi
    A valid ~/.cdsapirc (API key from https://cds.climate.copernicus.eu).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Iterator, Sequence

from gfs2calmet.config import Era5RunConfig, load_era5_config
from gfs2calmet.dataset import Frame
from gfs2calmet.era5_fields import DEFAULT_ERA5_FIELDS
from gfs2calmet.era5_reader import download_era5_period, read_era5_to_dataset
from gfs2calmet.frames import build_frames, build_header
from gfs2calmet.regrid import regrid_dataset
from gfs2calmet.writer import write_3ddat


_LOG = logging.getLogger("gfs2calmet.era5")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gfs2calmet.era5",
        description="ERA5 reanalysis → CALMET 3D.DAT prognostic input",
    )
    p.add_argument("config", type=Path,
                   help="Path to an ERA5 run YAML (see config_era5_example.yaml)")
    p.add_argument("--skip-download", action="store_true",
                   help="Reuse existing GRIB files in era5.output_dir")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase log verbosity (-v info, -vv debug)")
    return p.parse_args(argv)


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _existing_grib_paths(output_dir: str | None) -> tuple[Path, Path]:
    if output_dir is None:
        raise ValueError("--skip-download requires era5.output_dir to be set")
    d = Path(output_dir)
    pl_files = sorted(d.glob("era5_pl_*.grib2"))
    sl_files = sorted(d.glob("era5_sl_*.grib2"))
    if not pl_files:
        raise FileNotFoundError(f"No era5_pl_*.grib2 files found in {d}")
    if not sl_files:
        raise FileNotFoundError(f"No era5_sl_*.grib2 files found in {d}")
    if len(pl_files) > 1 or len(sl_files) > 1:
        _LOG.warning(
            "Multiple ERA5 GRIB files found; using most recent: %s, %s",
            pl_files[-1].name, sl_files[-1].name,
        )
    return pl_files[-1], sl_files[-1]


def _run(cfg: Era5RunConfig, *, skip_download: bool) -> int:
    _LOG.info(
        "ERA5 pipeline: %s → %s",
        cfg.era5.start_date.isoformat(), cfg.era5.end_date.isoformat(),
    )

    if skip_download:
        pl_path, sl_path = _existing_grib_paths(cfg.era5.output_dir)
        _LOG.info("Reusing existing GRIB files: %s, %s", pl_path, sl_path)
    else:
        pl_path, sl_path = download_era5_period(
            start=cfg.era5.start_date,
            end=cfg.era5.end_date,
            pressure_levels=cfg.era5.pressure_levels,
            output_dir=cfg.era5.output_dir or "./data/era5",
            target=cfg.target_grid,
        )

    _LOG.info("Decoding ERA5 GRIB files")
    src_ds = read_era5_to_dataset(
        pl_path, sl_path,
        fields=DEFAULT_ERA5_FIELDS,
        levels=cfg.era5.pressure_levels,
    )

    _LOG.info("Regridding to CALMET driver grid")
    tgt_ds = regrid_dataset(src_ds, cfg.target_grid)

    # Reconcile available vs requested pressure levels (same as GFS pipeline).
    cfg = _reconcile_pressure_levels(cfg, tgt_ds)

    header = build_header(tgt_ds, cfg.target_grid, cfg.header)
    n_times = tgt_ds.sizes["time"]
    header.time_window = dataclasses.replace(
        header.time_window,
        nhrsmm5=max(n_times - 1, 1),
    )

    _LOG.info("Writing %s (%d frames)", cfg.output_path, n_times)
    n = write_3ddat(
        cfg.output_path,
        header,
        _iter_frames(tgt_ds, cfg),
    )
    _LOG.info("Done. Wrote %d frames.", n)
    return 0


def _reconcile_pressure_levels(cfg: Era5RunConfig, tgt_ds) -> Era5RunConfig:
    available = set(int(lv) for lv in tgt_ds["level"].values)
    requested = list(cfg.era5.pressure_levels)
    kept = [p for p in requested if p in available]
    missing = [p for p in requested if p not in available]
    if missing:
        _LOG.warning("Pressure levels not in ERA5 source (dropped): %s", missing)
    if not kept:
        raise FileNotFoundError(
            f"None of the requested pressure_levels {requested} are in the "
            f"ERA5 file (available: {sorted(available, reverse=True)})"
        )
    if kept == requested:
        return cfg
    new_era5 = dataclasses.replace(cfg.era5, pressure_levels=kept)
    new_header = dataclasses.replace(cfg.header, pressure_levels=kept)
    new_frame = dataclasses.replace(cfg.frame, pressure_levels=kept)
    return dataclasses.replace(cfg, era5=new_era5, header=new_header, frame=new_frame)


def _iter_frames(tgt_ds, cfg: Era5RunConfig) -> Iterator[Frame]:
    """Yield one Frame per time step from the regridded Dataset."""
    n = tgt_ds.sizes["time"]
    for ti in range(n):
        _LOG.info("Frame %d/%d", ti + 1, n)
        slice_ds = tgt_ds.isel(time=slice(ti, ti + 1))
        yield from build_frames(slice_ds, cfg.frame)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    cfg = load_era5_config(args.config)
    return _run(cfg, skip_download=args.skip_download)


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
