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
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from gfs2calmet.config import Era5RunConfig, load_era5_config
from gfs2calmet.dataset import Frame
from gfs2calmet.era5_fields import DEFAULT_ERA5_FIELDS
from gfs2calmet.era5_reader import (
    _split_into_monthly_chunks,
    download_era5_period,
    read_era5_to_dataset,
)
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


def _existing_grib_paths(output_dir: str | None) -> tuple[list[Path], list[Path]]:
    if output_dir is None:
        raise ValueError("--skip-download requires era5.output_dir to be set")
    d = Path(output_dir)
    pl_files = sorted(d.glob("era5_pl_*.grib2"))
    sl_files = sorted(d.glob("era5_sl_*.grib2"))
    if not pl_files:
        raise FileNotFoundError(f"No era5_pl_*.grib2 files found in {d}")
    if not sl_files:
        raise FileNotFoundError(f"No era5_sl_*.grib2 files found in {d}")
    return pl_files, sl_files


def _run(cfg: Era5RunConfig, *, skip_download: bool) -> int:
    _LOG.info(
        "ERA5 pipeline: %s → %s",
        cfg.era5.start_date.isoformat(), cfg.era5.end_date.isoformat(),
    )

    if skip_download:
        pl_paths, sl_paths = _existing_grib_paths(cfg.era5.output_dir)
        _LOG.info(
            "Reusing %d existing GRIB chunk(s) from %s",
            len(pl_paths), cfg.era5.output_dir,
        )
    else:
        pl_paths, sl_paths = download_era5_period(
            start=cfg.era5.start_date,
            end=cfg.era5.end_date,
            pressure_levels=cfg.era5.pressure_levels,
            output_dir=cfg.era5.output_dir or "./data/era5",
            target=cfg.target_grid,
        )

    _LOG.info("Decoding %d ERA5 GRIB chunk(s)", len(pl_paths))
    src_ds = read_era5_to_dataset(
        pl_paths, sl_paths,
        fields=DEFAULT_ERA5_FIELDS,
        levels=cfg.era5.pressure_levels,
    )

    _LOG.info("Regridding to CALMET driver grid")
    tgt_ds = regrid_dataset(src_ds, cfg.target_grid)

    # Reconcile available vs requested pressure levels (same as GFS pipeline).
    cfg = _reconcile_pressure_levels(cfg, tgt_ds)

    # Split the regridded Dataset into per-calendar-month 3D.DAT files
    # (mirrors the GRIB download chunking and matches CALMET's NM3D
    # multi-file ingest).  Single-month runs produce one file.
    chunks = _split_into_monthly_chunks(cfg.era5.start_date, cfg.era5.end_date)
    _LOG.info(
        "Writing %d 3D.DAT file(s) (monthly chunks)", len(chunks),
    )

    total_frames = 0
    for chunk_start, chunk_end in chunks:
        chunk_ds = _slice_dataset_to_range(tgt_ds, chunk_start, chunk_end)
        n_times = chunk_ds.sizes["time"]
        if n_times == 0:
            _LOG.warning(
                "Skipping empty chunk %s → %s",
                chunk_start.isoformat(), chunk_end.isoformat(),
            )
            continue

        header = build_header(chunk_ds, cfg.target_grid, cfg.header)
        header.time_window = dataclasses.replace(
            header.time_window,
            nhrsmm5=max(n_times - 1, 1),
        )

        output_path = _resolve_output_path(
            cfg.output_path, chunk_start, chunk_end,
        )
        _LOG.info("  %s (%d frames)", output_path, n_times)
        n = write_3ddat(
            output_path,
            header,
            _iter_frames(chunk_ds, cfg),
        )
        total_frames += n

    _LOG.info(
        "Done. Wrote %d frames across %d file(s).", total_frames, len(chunks),
    )
    return 0


def _slice_dataset_to_range(ds, start: datetime, end: datetime):
    """Return ds restricted to time ∈ [start, end] (inclusive).

    xarray's ``sel`` with a slice on a sorted time axis gives the closed
    interval we want.  Used by the monthly-chunked writer.
    """
    import numpy as np                                  # noqa: PLC0415
    return ds.sel(time=slice(
        np.datetime64(start, "s"), np.datetime64(end, "s"),
    ))


def _resolve_output_path(user_path: str, start, end) -> str:
    """Inject the actual date range into the output filename.

    Stops old 3D.DAT files from being reused (or pointed at by CALMET.INP)
    after the requested dates change.  If the user's filename already
    contains the start-date stamp ``YYYYMMDDHH``, we leave it alone; this
    lets callers script exact filenames when they want to.
    """
    import re                                          # noqa: PLC0415

    start_stamp = start.strftime("%Y%m%d%H")
    end_stamp = end.strftime("%Y%m%d%H")
    p = Path(user_path)

    # Treat the canonical 3D.DAT compound extension as a single suffix so
    # the date stamp lands before ".3D.DAT", not between ".3D" and ".DAT".
    name = p.name
    for compound in (".3D.DAT", ".3D.dat"):
        if name.lower().endswith(compound.lower()):
            stem = name[: -len(compound)]
            ext = name[-len(compound):]
            break
    else:
        stem = p.stem
        ext = p.suffix

    if start_stamp in stem:
        return str(p)

    # Strip any pre-existing 10-digit date stamp(s) before re-stamping so
    # repeated edits don't accumulate ``..._old_new`` suffixes.
    cleaned = re.sub(r"_?\d{10}(?:_\d{10})?", "", stem).strip("_") or "ERA5"
    return str(p.with_name(f"{cleaned}_{start_stamp}_{end_stamp}{ext}"))


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
