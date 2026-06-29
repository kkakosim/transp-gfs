"""Download GFS GRIB2 with Herbie and decode it with pygrib.

The module exposes two boundaries:

    download_gfs_cycle  — Herbie wrapper; downloads the minimal subset
                          of GRIB2 messages for one model cycle.
    read_gfs_to_dataset — pygrib wrapper; decodes one or more GRIB2
                          files into an xarray.Dataset whose variables
                          carry documented target units (per GfsField).

Both functions lazy-import their heavy third-party dependency so that
the writer-only install path (``pip install numpy pytest``) keeps
working. Tests inject a fake module via the ``pygrib_module`` /
``herbie_module`` keyword to exercise the data plumbing offline.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from gfs2calmet.gfs_fields import (
    DEFAULT_GFS_FIELDS,
    GfsField,
    herbie_search_for,
)


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (kept here so tests can patch sys.modules cleanly)
# ---------------------------------------------------------------------------


def _import_pygrib() -> Any:
    import pygrib                              # noqa: PLC0415
    return pygrib


def _import_xarray() -> Any:
    import xarray                              # noqa: PLC0415
    return xarray


def _import_herbie() -> Any:
    from herbie import Herbie                  # noqa: PLC0415
    return Herbie


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedMessage:
    """One GRIB2 message after pygrib decode, with target-unit values."""
    role: str
    valid_time: np.datetime64
    level: int                  # hPa for pressure levels, otherwise the
                                # GfsField.level (e.g. 10, 2, 0)
    latitudes: np.ndarray       # 2D, shape (ny, nx) as pygrib returns
    longitudes: np.ndarray      # 2D, shape (ny, nx)
    values: np.ndarray          # 2D, shape (ny, nx), in target_units


def _select_messages(
    grb_iter: Any, field: GfsField, levels: Sequence[int] | None
) -> list[Any]:
    """Pull matching pygrib messages from one open file.

    Filters by ``shortName`` and ``typeOfLevel`` (always) and by
    ``level`` when the field pins one. For pressure-level fields where
    ``field.level is None``, the optional ``levels`` argument further
    restricts which pressure levels are kept.
    """
    matched: list[Any] = []
    want_shorts = field.short_names
    want_tol = field.type_of_level
    want_level = field.level
    for msg in grb_iter:
        if getattr(msg, "shortName", None) not in want_shorts:
            continue
        if getattr(msg, "typeOfLevel", None) != want_tol:
            continue
        if want_level is not None and getattr(msg, "level", None) != want_level:
            continue
        if want_level is None and levels is not None:
            if getattr(msg, "level", None) not in levels:
                continue
        matched.append(msg)
    return matched


def _extract_messages(
    grib_paths: Iterable[str | os.PathLike[str]],
    fields: Sequence[GfsField],
    levels: Sequence[int] | None = None,
    *,
    pygrib_module: Any | None = None,
    one_time_per_file: bool = True,
) -> list[ExtractedMessage]:
    """Open each path with pygrib, pull matching messages, convert units.

    Parameters
    ----------
    one_time_per_file
        When True (GFS convention), every message in a file is normalized
        to ``max(file_valid_times)`` so that accumulated fields (APCP,
        DSWRF, DLWRF, ...) which report a midpoint validDate align with
        the instantaneous fields' end-of-interval validDate.  Without
        this, _assemble_dataset doubles the time axis and each field is
        half NaN.

        When False (ERA5 / any multi-time file convention), each message
        keeps its own validDate.  Required when a single GRIB carries
        many forecast hours — collapsing would discard all but one hour.

    Raises FileNotFoundError if a non-optional field has no matching
    message across all files.
    """
    pg = pygrib_module or _import_pygrib()

    extracted: list[ExtractedMessage] = []
    seen_per_role: dict[str, int] = {f.role: 0 for f in fields}

    grib_paths = list(grib_paths)
    n_files = len(grib_paths)
    for idx, path in enumerate(grib_paths, start=1):
        _LOG.info("  [%d/%d] opening %s", idx, n_files, Path(path).name)
        grbs = pg.open(str(path))
        try:
            all_msgs = list(grbs)
        finally:
            close = getattr(grbs, "close", None)
            if callable(close):
                close()
        _LOG.info(
            "  [%d/%d] read %d messages from %s",
            idx, n_files, len(all_msgs), Path(path).name,
        )

        file_valid_times = [
            _as_datetime(m.validDate) for m in all_msgs
            if hasattr(m, "validDate")
        ]
        if not file_valid_times:
            _LOG.warning("No valid_times found in %s; skipping file", path)
            continue

        file_valid_np: np.datetime64 | None
        if one_time_per_file:
            file_time = max(file_valid_times)
            file_valid_np = np.datetime64(file_time, "s")
            distinct = len(set(file_valid_times))
            if distinct > 1:
                _LOG.debug(
                    "File %s carries %d distinct validDates; normalizing "
                    "all messages to %s",
                    path, distinct, file_time.isoformat(),
                )
        else:
            file_valid_np = None  # use each message's own validDate below

        for field in fields:
            for msg in _select_messages(all_msgs, field, levels):
                lats, lons = msg.latlons()
                raw = np.asarray(msg.values, dtype=np.float64)
                converted = field.multiplier * raw + field.offset
                level = int(getattr(msg, "level", 0))
                if file_valid_np is None:
                    msg_time = _as_datetime(msg.validDate)
                    valid_np = np.datetime64(msg_time, "s")
                else:
                    valid_np = file_valid_np
                extracted.append(
                    ExtractedMessage(
                        role=field.role,
                        valid_time=valid_np,
                        level=level,
                        latitudes=np.asarray(lats, dtype=np.float64),
                        longitudes=np.asarray(lons, dtype=np.float64),
                        values=converted,
                    )
                )
                seen_per_role[field.role] += 1

    for field in fields:
        if seen_per_role[field.role] == 0:
            msg = f"no GRIB2 messages matched field role={field.role!r}"
            if field.optional:
                _LOG.warning("%s (optional, skipping)", msg)
            else:
                raise FileNotFoundError(msg)

    return extracted


def _as_datetime(dt: Any) -> datetime:
    """pygrib.validDate is usually a datetime; tolerate strings too."""
    if isinstance(dt, datetime):
        return dt
    return datetime.fromisoformat(str(dt))


# ---------------------------------------------------------------------------
# Assemble into xarray.Dataset
# ---------------------------------------------------------------------------


def _check_grid_consistency(messages: Sequence[ExtractedMessage]) -> None:
    """All messages must share the same (lat, lon) grid; we use the first."""
    if not messages:
        return
    ref_lat = messages[0].latitudes
    ref_lon = messages[0].longitudes
    for m in messages[1:]:
        if m.latitudes.shape != ref_lat.shape:
            raise ValueError(
                f"grid shape mismatch: {m.role} has {m.latitudes.shape}, "
                f"expected {ref_lat.shape}"
            )


def _assemble_dataset(
    messages: Sequence[ExtractedMessage],
    fields: Sequence[GfsField],
    *,
    xarray_module: Any | None = None,
) -> Any:
    """Build an xarray.Dataset from extracted messages.

    Layout:
        Pressure-level fields  → dims (time, level, latitude, longitude)
        Surface/2D fields       → dims (time, latitude, longitude)
    """
    xr = xarray_module or _import_xarray()
    if not messages:
        raise ValueError("no messages to assemble")
    _check_grid_consistency(messages)

    # 1D coordinate axes derived from the first message's 2D grid.
    lats_2d = messages[0].latitudes
    lons_2d = messages[0].longitudes
    if lats_2d.ndim == 2:
        latitudes = lats_2d[:, 0]
        longitudes = lons_2d[0, :]
    else:
        latitudes = lats_2d
        longitudes = lons_2d

    by_role = {f.role: f for f in fields}

    # Group messages: per role, collect set of times and levels.
    grouped: dict[str, list[ExtractedMessage]] = {}
    for m in messages:
        grouped.setdefault(m.role, []).append(m)

    # Union of all times, levels seen across roles (for consistent coords).
    all_times = sorted({m.valid_time for m in messages})
    pl_levels = sorted({
        m.level for m in messages
        if by_role[m.role].type_of_level == "isobaricInhPa"
    }, reverse=True)  # surface → top by descending pressure

    time_idx = {t: i for i, t in enumerate(all_times)}
    level_idx = {lv: i for i, lv in enumerate(pl_levels)}

    data_vars: dict[str, Any] = {}
    ny, nx = lats_2d.shape if lats_2d.ndim == 2 else (lats_2d.size, lons_2d.size)

    for role, msgs in grouped.items():
        f = by_role[role]
        is_pl = f.type_of_level == "isobaricInhPa"
        if is_pl:
            arr = np.full(
                (len(all_times), len(pl_levels), ny, nx),
                np.nan, dtype=np.float64,
            )
            for m in msgs:
                arr[time_idx[m.valid_time], level_idx[m.level], :, :] = m.values
            data_vars[role] = (
                ("time", "level", "latitude", "longitude"),
                arr,
                {
                    "units": f.target_units,
                    "native_units": f.native_units,
                    "grib_short_name": ",".join(f.short_names),
                },
            )
        else:
            arr = np.full((len(all_times), ny, nx), np.nan, dtype=np.float64)
            for m in msgs:
                arr[time_idx[m.valid_time], :, :] = m.values
            data_vars[role] = (
                ("time", "latitude", "longitude"),
                arr,
                {
                    "units": f.target_units,
                    "native_units": f.native_units,
                    "grib_short_name": ",".join(f.short_names),
                    "type_of_level": f.type_of_level,
                    "level": f.level if f.level is not None else 0,
                },
            )

    coords = {
        "time": ("time", np.array(all_times, dtype="datetime64[s]")),
        "latitude": ("latitude", np.asarray(latitudes, dtype=np.float64)),
        "longitude": ("longitude", np.asarray(longitudes, dtype=np.float64)),
    }
    if pl_levels:
        coords["level"] = ("level", np.asarray(pl_levels, dtype=np.int32))

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs={"source": "GFS"})


# ---------------------------------------------------------------------------
# Public decode entry point
# ---------------------------------------------------------------------------


def read_gfs_to_dataset(
    grib_paths: Iterable[str | os.PathLike[str]],
    fields: Sequence[GfsField] = DEFAULT_GFS_FIELDS,
    levels: Sequence[int] | None = None,
    *,
    pygrib_module: Any | None = None,
    xarray_module: Any | None = None,
) -> Any:
    """Decode GFS GRIB2 files into one xarray.Dataset in target units.

    Parameters
    ----------
    grib_paths
        One or more GRIB2 files (typically one per forecast hour).
    fields
        Which roles to extract. Defaults to the full DEFAULT_GFS_FIELDS
        catalog.
    levels
        Optional list of pressure levels (hPa) to keep for pressure-
        level fields. ``None`` keeps every level present in the files.
    pygrib_module, xarray_module
        Test injection hooks. Production callers should leave these
        unset.
    """
    messages = _extract_messages(
        grib_paths, fields, levels, pygrib_module=pygrib_module
    )
    return _assemble_dataset(messages, fields, xarray_module=xarray_module)


# ---------------------------------------------------------------------------
# Download (Herbie)
# ---------------------------------------------------------------------------


def _normalize_cycle(cycle: datetime | str) -> datetime:
    if isinstance(cycle, datetime):
        return cycle
    # Accept "YYYY-MM-DD HH" or ISO8601-ish strings.
    return datetime.fromisoformat(str(cycle).replace("/", "-"))


def download_gfs_cycle(
    cycle: datetime | str,
    fxx_hours: Iterable[int],
    roles: Iterable[str],
    output_dir: str | os.PathLike[str] | None = None,
    *,
    model: str = "gfs",
    product: str = "pgrb2.0p25",
    herbie_module: Any | None = None,
) -> list[Path]:
    """Download one GFS cycle, subset by GRIB index to the requested roles.

    Returns the list of local GRIB2 paths in the same order as
    ``fxx_hours``. Errors from Herbie propagate to the caller.

    The default product ``pgrb2.0p25`` is the standard 0.25-degree GFS
    forecast file. NCEP releases an hourly variant ``pgrb2.0p25.f000``
    family — pass ``product='pgrb2.0p25'`` and let Herbie resolve the
    hourly URLs from ``fxx_hours``.
    """
    Herbie = herbie_module or _import_herbie()
    cycle_dt = _normalize_cycle(cycle)
    search = herbie_search_for(roles)
    out_dir = Path(output_dir) if output_dir is not None else None

    paths: list[Path] = []
    for fxx in fxx_hours:
        h = Herbie(
            cycle_dt,
            model=model,
            product=product,
            fxx=int(fxx),
            save_dir=str(out_dir) if out_dir else None,
        )
        local = h.download(search=search)
        if local is None:
            raise FileNotFoundError(
                f"Herbie returned no file for cycle={cycle_dt} fxx={fxx}"
            )
        paths.append(Path(local))
    return paths


# ---------------------------------------------------------------------------
# Convenience: build the regex from a Field sequence directly
# ---------------------------------------------------------------------------


def herbie_search_for_fields(fields: Iterable[GfsField]) -> str:
    """Like ``herbie_search_for`` but accepts GfsField objects."""
    pattern = herbie_search_for([f.role for f in fields])
    # Sanity check: a well-formed pattern compiles.
    re.compile(pattern)
    return pattern
