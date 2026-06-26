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
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Iterator, Sequence

from gfs2calmet.config import RunConfig, load_config
from gfs2calmet.dataset import Frame
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

    if not grib_paths:
        raise FileNotFoundError("no GRIB2 files to process")

    # Stream one file at a time: decoding all 73 GFS files at once would
    # hold ~35 GB of native-grid arrays in memory (60+ messages per file
    # at 0.25 degree x 8 bytes = ~480 MB per file). Decoding per-file
    # and regridding to the small target grid before moving on caps
    # working memory at one file's worth.
    _LOG.info("Decoding/regridding %d GRIB2 files (streaming, per-file)",
              len(grib_paths))

    # Build the header from the first file's geometry, then patch the
    # period-length field to reflect the total file count (one valid
    # time per file).
    first_tgt = _decode_one(grib_paths[0], cfg)
    header = build_header(first_tgt, cfg.target_grid, cfg.header)
    header.time_window = dataclasses.replace(
        header.time_window,
        nhrsmm5=max(len(grib_paths) - 1, 1),
    )

    _LOG.info("Writing %s", cfg.output_path)
    n = write_3ddat(
        cfg.output_path,
        header,
        _stream_frames(grib_paths, cfg, first_tgt),
    )
    _LOG.info("Done. Wrote %d frames.", n)
    return 0


def _decode_one(grib_path: Path, cfg: RunConfig):
    """Decode one GRIB2 file and regrid it to the configured driver grid.

    Factored out so that the streaming generator can call it for every
    forecast hour without the CLI orchestration code growing a loop
    body that's harder to test in isolation.
    """
    src_ds = read_gfs_to_dataset(
        [grib_path],
        fields=DEFAULT_GFS_FIELDS,
        levels=cfg.gfs.pressure_levels,
    )
    return regrid_dataset(src_ds, cfg.target_grid)


def _stream_frames(
    grib_paths: list[Path],
    cfg: RunConfig,
    first_tgt,
) -> Iterator[Frame]:
    """Yield one Frame per forecast hour. The first file's regridded
    Dataset is reused (we already decoded it to build the header)."""
    _LOG.info("Frame 1/%d: %s", len(grib_paths), grib_paths[0].name)
    yield from build_frames(first_tgt, cfg.frame)
    for i, path in enumerate(grib_paths[1:], start=2):
        _LOG.info("Frame %d/%d: %s", i, len(grib_paths), path.name)
        tgt = _decode_one(path, cfg)
        yield from build_frames(tgt, cfg.frame)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    cfg = load_config(args.config)
    return _run(cfg, skip_download=args.skip_download)


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
