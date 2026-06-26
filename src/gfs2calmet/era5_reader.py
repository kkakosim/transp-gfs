"""Download ERA5 reanalysis via the Copernicus CDS API and decode to xarray.

Public surface:
    download_era5_period  — CDS download for a start/end date range.
    read_era5_to_dataset  — pygrib decode + radiation deaccumulation.

Prerequisites:
    pip install cdsapi pygrib xarray numpy
    A ~/.cdsapirc file (or CDS_API_KEY / CDS_API_URL env vars) with a
    valid Copernicus CDS API key — register at https://cds.climate.copernicus.eu.

CDS returns two GRIB files (pressure-level + single-level) that are merged
into one xarray Dataset with the same variable names and units as GFS so
the regrid → build_frames → write_3ddat pipeline is unchanged.

Radiation note:
    ERA5 ssrd/strd are accumulated J/m² since the start of the enclosing
    12-hour forecast window.  deaccumulate_radiation() converts them to
    hourly-mean W/m² by differencing consecutive steps and clamping
    negatives (which arise at reset points or at night).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from gfs2calmet.era5_fields import (
    CDS_PRESSURE_LEVEL_VARIABLES,
    CDS_SINGLE_LEVEL_VARIABLES,
    DEFAULT_ERA5_FIELDS,
)
from gfs2calmet.gfs_fields import GfsField
from gfs2calmet.gfs_reader import _extract_messages, _assemble_dataset


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_cdsapi() -> Any:
    import cdsapi                              # noqa: PLC0415
    return cdsapi


def _import_xarray() -> Any:
    import xarray                              # noqa: PLC0415
    return xarray


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _hourly_steps(start: datetime, end: datetime) -> list[datetime]:
    steps = []
    t = start
    while t <= end:
        steps.append(t)
        t += timedelta(hours=1)
    return steps


def _cds_date_args(steps: list[datetime]) -> dict[str, list[str]]:
    """Build year/month/day/time lists for a CDS API request.

    CDS evaluates the cross-product of year × month × day × time; we
    deduplicate each dimension to avoid requesting duplicate times while
    keeping the number of API parameters small.
    """
    return {
        "year":  sorted({str(s.year) for s in steps}),
        "month": sorted({f"{s.month:02d}" for s in steps}),
        "day":   sorted({f"{s.day:02d}" for s in steps}),
        "time":  sorted({f"{s.hour:02d}:00" for s in steps}),
    }


# ---------------------------------------------------------------------------
# Bounding box from TargetGrid
# ---------------------------------------------------------------------------


def _bbox_from_target(target: Any, buffer_deg: float = 2.0) -> list[float]:
    """Return [north, west, south, east] degrees from a TargetGrid + buffer.

    Clips to [-90,90] / [-180,180].  The buffer ensures that all target
    cells have at least one surrounding ERA5 grid point for bilinear
    interpolation.
    """
    _, _, lats, lons = target.cell_meshes()
    north = min(float(np.max(lats)) + buffer_deg, 90.0)
    south = max(float(np.min(lats)) - buffer_deg, -90.0)
    east  = min(float(np.max(lons)) + buffer_deg, 180.0)
    west  = max(float(np.min(lons)) - buffer_deg, -180.0)
    return [north, west, south, east]


# ---------------------------------------------------------------------------
# CDS download
# ---------------------------------------------------------------------------


def _split_into_monthly_chunks(
    start: datetime, end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into per-calendar-month (start, end) sub-ranges.

    CDS request URLs grow with the year × month × day × time cross product;
    splitting by calendar month keeps each request bounded to ~744 hours
    (31 days × 24 h) and lets users redo just one month if a job fails.
    """
    chunks: list[tuple[datetime, datetime]] = []
    cur_start = start
    while cur_start <= end:
        # Last hour of the current month.
        if cur_start.month == 12:
            next_month_first = datetime(cur_start.year + 1, 1, 1)
        else:
            next_month_first = datetime(cur_start.year, cur_start.month + 1, 1)
        chunk_end = min(next_month_first - timedelta(hours=1), end)
        chunks.append((cur_start, chunk_end))
        cur_start = chunk_end + timedelta(hours=1)
    return chunks


def download_era5_period(
    start: datetime,
    end: datetime,
    pressure_levels: Sequence[int],
    output_dir: str | os.PathLike[str],
    *,
    target: Any | None = None,
    buffer_deg: float = 2.0,
    cdsapi_module: Any | None = None,
) -> tuple[list[Path], list[Path]]:
    """Download ERA5 GRIB2 for *start* to *end* (inclusive, hourly).

    The request is split into per-calendar-month chunks so each CDS job
    stays bounded in size (≤ 744 hours).  Within a chunk, two files are
    downloaded — one for pressure-level fields, one for single-level —
    matching the CDS dataset split.  Already-present files are reused.

    Parameters
    ----------
    start, end
        Inclusive UTC date range.  Both must be on the hour.
    pressure_levels
        List of pressure levels in hPa (e.g. [1000, 925, 850, ...]).
    output_dir
        Directory to save the GRIB files.
    target
        Optional ``TargetGrid`` used to compute a spatial subset (area).
        Pass ``None`` to download the global ERA5 grid (large!).
    buffer_deg
        Degrees of padding around the target grid bbox.
    cdsapi_module
        Test injection hook; leave unset in production.

    Returns
    -------
    (pl_paths, sl_paths)
        Lists of pressure-level and single-level GRIB file paths, in
        chronological order (one entry per monthly chunk).
    """
    cdsapi = cdsapi_module or _import_cdsapi()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks = _split_into_monthly_chunks(start, end)
    if not chunks:
        raise ValueError(f"Empty date range: {start} → {end}")
    _LOG.info(
        "ERA5 request: %s → %s, %d pressure levels, %d monthly chunks",
        start.isoformat(), end.isoformat(), len(pressure_levels), len(chunks),
    )

    area = _bbox_from_target(target, buffer_deg) if target is not None else None
    client = cdsapi.Client()
    pl_paths: list[Path] = []
    sl_paths: list[Path] = []

    for chunk_start, chunk_end in chunks:
        steps = _hourly_steps(chunk_start, chunk_end)
        date_args = _cds_date_args(steps)
        common: dict[str, Any] = {
            "product_type": "reanalysis",
            "data_format": "grib",
            **date_args,
        }
        if area is not None:
            common["area"] = area

        stamp = f"{chunk_start:%Y%m%d%H}_{chunk_end:%Y%m%d%H}"
        _LOG.info(
            "  chunk %s → %s (%d hours)",
            chunk_start.isoformat(), chunk_end.isoformat(), len(steps),
        )

        # --- Pressure-level ---
        pl_path = out / f"era5_pl_{stamp}.grib2"
        if pl_path.exists():
            _LOG.info("    reusing %s", pl_path.name)
        else:
            _LOG.info("    downloading %s", pl_path.name)
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    **common,
                    "variable": CDS_PRESSURE_LEVEL_VARIABLES,
                    "pressure_level": [str(p) for p in pressure_levels],
                },
                str(pl_path),
            )
        pl_paths.append(pl_path)

        # --- Single-level ---
        sl_path = out / f"era5_sl_{stamp}.grib2"
        if sl_path.exists():
            _LOG.info("    reusing %s", sl_path.name)
        else:
            _LOG.info("    downloading %s", sl_path.name)
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    **common,
                    "variable": CDS_SINGLE_LEVEL_VARIABLES,
                },
                str(sl_path),
            )
        sl_paths.append(sl_path)

    return pl_paths, sl_paths


# ---------------------------------------------------------------------------
# Radiation deaccumulation
# ---------------------------------------------------------------------------


def deaccumulate_radiation(ds: Any) -> Any:
    """Convert accumulated ssrd/strd (J/m²) to hourly-mean flux (W/m²).

    ERA5 radiation fields are accumulated from the start of the enclosing
    forecast window (resets roughly every 12 h, detected by a drop in the
    cumulative value).  This function differences consecutive time steps and
    divides by 3600 s to produce a mean W/m² for each hour.  Negative
    values (from noise at reset points or at night) are clamped to zero.
    """
    xr = _import_xarray()
    for role in ("dswrf", "dlwrf"):
        if role not in ds:
            continue
        arr = ds[role].values.copy()   # shape (time, ...)
        result = np.empty_like(arr)
        result[0] = arr[0] / 3600.0
        # Element-wise: where the previous step was higher (accumulation
        # reset between this step and the prior one) treat the current
        # value as a fresh one-hour accumulation; otherwise use the diff.
        for ti in range(1, len(arr)):
            diff = arr[ti] - arr[ti - 1]
            result[ti] = np.where(diff >= 0.0, diff, arr[ti]) / 3600.0
        result = np.maximum(result, 0.0)
        ds = ds.assign(
            {role: xr.DataArray(result, dims=ds[role].dims, attrs={
                **ds[role].attrs, "units": "W/m2",
            })}
        )
    return ds


# ---------------------------------------------------------------------------
# Decode ERA5 GRIB to xarray Dataset
# ---------------------------------------------------------------------------


def read_era5_to_dataset(
    pl_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    sl_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    fields: Sequence[GfsField] = DEFAULT_ERA5_FIELDS,
    levels: Sequence[int] | None = None,
    *,
    pygrib_module: Any | None = None,
    xarray_module: Any | None = None,
) -> Any:
    """Decode pressure-level and single-level ERA5 GRIB files into one Dataset.

    Accepts either single paths or lists of paths (one per monthly chunk
    when the requested period spans multiple months).  All GRIBs are
    concatenated along the time axis.

    The returned Dataset has the same variable names and units as the GFS
    pipeline so ``regrid_dataset``, ``build_frames``, and ``write_3ddat``
    work without modification.

    Radiation fields (dswrf / dlwrf) are deaccumulated from J/m² to W/m²
    before returning.
    """
    pl_list = _as_path_list(pl_paths)
    sl_list = _as_path_list(sl_paths)
    messages = _extract_messages(
        [*pl_list, *sl_list], fields, levels,
        pygrib_module=pygrib_module,
        # ERA5 ships every requested hour inside a single GRIB; collapsing
        # to one time per file would keep only the last hour.
        one_time_per_file=False,
    )
    ds = _assemble_dataset(messages, fields, xarray_module=xarray_module)
    ds = deaccumulate_radiation(ds)
    return ds


def _as_path_list(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> list[str | os.PathLike[str]]:
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths)
