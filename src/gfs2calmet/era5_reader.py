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


def download_era5_period(
    start: datetime,
    end: datetime,
    pressure_levels: Sequence[int],
    output_dir: str | os.PathLike[str],
    *,
    target: Any | None = None,
    buffer_deg: float = 2.0,
    cdsapi_module: Any | None = None,
) -> tuple[Path, Path]:
    """Download ERA5 GRIB2 for *start* to *end* (inclusive, hourly).

    Two files are downloaded: one for pressure-level fields and one for
    single-level (surface) fields.  CDS queues the request server-side;
    typical wait time is 1–5 minutes for a multi-day period.

    Parameters
    ----------
    start, end
        Inclusive UTC date range.  Both must be on the hour.
    pressure_levels
        List of pressure levels in hPa (e.g. [1000, 925, 850, ...]).
    output_dir
        Directory to save the two GRIB files.
    target
        Optional ``TargetGrid`` used to compute a spatial subset (area).
        Pass ``None`` to download the global ERA5 grid (large!).
    buffer_deg
        Degrees of padding around the target grid bbox.
    cdsapi_module
        Test injection hook; leave unset in production.

    Returns
    -------
    (pl_path, sl_path)
        Paths to the pressure-level and single-level GRIB files.
    """
    cdsapi = cdsapi_module or _import_cdsapi()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = _hourly_steps(start, end)
    if not steps:
        raise ValueError(f"Empty date range: {start} → {end}")

    date_args = _cds_date_args(steps)
    _LOG.info(
        "ERA5 request: %d hours (%s → %s), %d pressure levels",
        len(steps), start.isoformat(), end.isoformat(), len(pressure_levels),
    )

    area = _bbox_from_target(target, buffer_deg) if target is not None else None

    common: dict[str, Any] = {
        "product_type": "reanalysis",
        "format": "grib",
        **date_args,
    }
    if area is not None:
        common["area"] = area

    client = cdsapi.Client()

    # --- Pressure-level download ---
    pl_path = out / f"era5_pl_{start:%Y%m%d%H}_{end:%Y%m%d%H}.grib2"
    if pl_path.exists():
        _LOG.info("Reusing existing pressure-level file: %s", pl_path)
    else:
        _LOG.info("Downloading pressure-level ERA5 → %s", pl_path)
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                **common,
                "variable": CDS_PRESSURE_LEVEL_VARIABLES,
                "pressure_level": [str(p) for p in pressure_levels],
            },
            str(pl_path),
        )

    # --- Single-level download ---
    sl_path = out / f"era5_sl_{start:%Y%m%d%H}_{end:%Y%m%d%H}.grib2"
    if sl_path.exists():
        _LOG.info("Reusing existing single-level file: %s", sl_path)
    else:
        _LOG.info("Downloading single-level ERA5 → %s", sl_path)
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                **common,
                "variable": CDS_SINGLE_LEVEL_VARIABLES,
            },
            str(sl_path),
        )

    return pl_path, sl_path


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
        for ti in range(1, len(arr)):
            diff = arr[ti] - arr[ti - 1]
            # Negative diff → accumulation reset; treat the current value
            # as a fresh accumulation over one hour.
            result[ti] = (diff if diff >= 0 else arr[ti]) / 3600.0
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
    pl_path: str | os.PathLike[str],
    sl_path: str | os.PathLike[str],
    fields: Sequence[GfsField] = DEFAULT_ERA5_FIELDS,
    levels: Sequence[int] | None = None,
    *,
    pygrib_module: Any | None = None,
    xarray_module: Any | None = None,
) -> Any:
    """Decode pressure-level and single-level ERA5 GRIB files into one Dataset.

    The returned Dataset has the same variable names and units as the GFS
    pipeline so ``regrid_dataset``, ``build_frames``, and ``write_3ddat``
    work without modification.

    Radiation fields (dswrf / dlwrf) are deaccumulated from J/m² to W/m²
    before returning.
    """
    messages = _extract_messages(
        [pl_path, sl_path], fields, levels, pygrib_module=pygrib_module,
    )
    ds = _assemble_dataset(messages, fields, xarray_module=xarray_module)
    ds = deaccumulate_radiation(ds)
    return ds
