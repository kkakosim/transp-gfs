"""Turn a regridded GFS Dataset into 3D.DAT Header + Frame objects.

This is the glue layer between the regridder and the writer. It does
the meteorology-aware conversions the writer doesn't know about:

    * wind components (u, v) → speed + meteorological direction
    * specific humidity (g/kg) → vapor mixing ratio (g/kg)
    * RH + T + P → vapor mixing ratio (Tetens) as a fallback
    * pressure level → CALWRF-style sigma = P / 1013.0

Anything GFS does not natively provide (graupel, vertical velocity in
m/s, sea-surface temperature from a non-SST product) is set to zero;
the corresponding OutputFlags should be zero so those fields are not
written to 3D.DAT at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np


_LOG = logging.getLogger(__name__)

from gfs2calmet.dataset import (
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
)
from gfs2calmet.regrid import TargetGrid


# Surface reference pressure used by CALWRF to normalize sigma layers
# (subroutine line "sigma levels (normalized by 1013.0 hPa)").
_CALWRF_SIGMA_REF_HPA = 1013.0

# Below this 10-m wind speed CALWRF (line 1252) defaults wind direction
# to 360°. Keeps stagnant cells from emitting noisy wd values.
_CALWRF_CALM_WIND_THRESHOLD_MPS = 0.05


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


def wind_uv_to_speed_dir(
    u: np.ndarray, v: np.ndarray, *, calm_default_deg: float = 360.0,
    calm_threshold: float = _CALWRF_CALM_WIND_THRESHOLD_MPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert U, V (m/s, east-north) to wind speed (m/s) and
    meteorological direction (degrees, where the wind is coming FROM).

    Calms (speed below ``calm_threshold``) are reported with
    ``calm_default_deg`` to match CALWRF's convention.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ws = np.hypot(u, v)
    wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    wd = np.where(ws < calm_threshold, calm_default_deg, wd)
    return ws, wd


def mixing_ratio_from_specific_humidity_gkg(q_gkg: np.ndarray) -> np.ndarray:
    """w = q / (1 - q) where both are in kg/kg; expressed in g/kg in & out."""
    q_gkg = np.asarray(q_gkg, dtype=np.float64)
    q_frac = q_gkg / 1000.0
    w_frac = q_frac / np.clip(1.0 - q_frac, 1e-9, None)
    return 1000.0 * w_frac


def mixing_ratio_from_rh_t_p(
    rh_pct: np.ndarray, t_k: np.ndarray, p_hpa: np.ndarray,
) -> np.ndarray:
    """Vapor mixing ratio (g/kg) from RH (%), temperature (K), pressure (hPa).

    Uses the Magnus/Tetens form ``es = 6.112 * exp(17.67*Tc/(Tc+243.5))``
    for saturation vapor pressure over water and ``w = 0.622 e / (p - e)``.
    Accurate enough for met preprocessing at GFS resolution.
    """
    rh = np.clip(np.asarray(rh_pct, dtype=np.float64), 0.0, 100.0)
    t_c = np.asarray(t_k, dtype=np.float64) - 273.15
    p = np.asarray(p_hpa, dtype=np.float64)
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e = (rh / 100.0) * es
    w = 0.622 * e / np.clip(p - e, 1e-9, None)
    return 1000.0 * w


def rh_from_dewpoint(
    td_k: np.ndarray, t_k: np.ndarray,
) -> np.ndarray:
    """Relative humidity (%) from 2-m dewpoint and temperature (both in K).

    Uses the August-Roche-Magnus approximation.
    """
    td_c = np.asarray(td_k, dtype=np.float64) - 273.15
    t_c = np.asarray(t_k, dtype=np.float64) - 273.15
    es_t  = np.exp(17.625 * t_c  / (243.04 + t_c))
    es_td = np.exp(17.625 * td_c / (243.04 + td_c))
    return np.clip(100.0 * es_td / es_t, 0.0, 100.0)


def sigma_levels_from_pressures(
    pressure_levels_hpa: Sequence[int],
    ref_hpa: float = _CALWRF_SIGMA_REF_HPA,
) -> list[float]:
    """Map each pressure level to a sigma value via P/Pref.

    Follows the CALWRF convention (calwrf.f, "sigma levels (normalized
    by 1013.0 hPa)"). NOT a true MM5 half-sigma — CALMET only uses
    these as a layer descriptor for the diagnostic-wind step.
    """
    return [round(float(p) / ref_hpa, 3) for p in pressure_levels_hpa]


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderOptions:
    """Per-run header knobs that aren't derivable from the Dataset itself."""

    output_flags: OutputFlags
    pressure_levels: Sequence[int]
    nland: int
    default_elevation_m: int = 0
    default_landuse: int = 16        # USGS "water" category — caller overrides
    dataset_message: str = "Produced by gfs2calmet"
    comments: Sequence[str] = field(default_factory=tuple)
    # MAPTXT code stored verbatim in the projection record. CALMET only
    # has well-defined readers for 'LCC', 'MER', 'PS', 'UTM' — keep this
    # in sync with the TargetGrid's CRS.
    maptxt: str = "UTM"
    truelat1: float = 0.0
    truelat2: float = 0.0
    rlatc: float = 0.0
    rlonc: float = 0.0


def build_header(
    ds: Any,
    target: TargetGrid,
    options: HeaderOptions,
) -> Header:
    """Assemble the Header that goes in front of the data records.

    ``ds`` must be a regridded Dataset (output of
    :func:`gfs2calmet.regrid.regrid_dataset`) with ``time``, ``level``,
    ``y``, ``x`` coordinates plus 2D ``latitude`` / ``longitude``.
    """
    times = [_as_utc_datetime(t) for t in np.asarray(ds["time"].values)]
    if not times:
        raise ValueError("Dataset has no time samples")
    first = times[0]
    nzp = len(options.pressure_levels)
    nxp = target.nx
    nyp = target.ny

    sigma = sigma_levels_from_pressures(options.pressure_levels)

    # Build the NXP*NYP GridPoint records in (jx outer, ix inner) order
    # to match the writer's iteration. We populate dot-point lat/lon
    # from the 2D coords; cross-point coords get the CALWRF -999 marker
    # since GFS is mass-point only.
    lats = np.asarray(ds["latitude"].values, dtype=np.float64)
    lons = np.asarray(ds["longitude"].values, dtype=np.float64)
    if lats.shape != (nyp, nxp):
        raise ValueError(
            f"latitude shape {lats.shape} does not match target "
            f"({nyp}, {nxp})"
        )

    grid_points: list[GridPoint] = []
    for j in range(nyp):
        for i in range(nxp):
            grid_points.append(
                GridPoint(
                    iindex=i + 1,
                    jindex=j + 1,
                    xlat_dot=float(lats[j, i]),
                    xlong_dot=float(lons[j, i]),
                    ielev_dot=options.default_elevation_m,
                    iland=options.default_landuse,
                    xlat_crs=-999.0,
                    xlong_crs=-999.0,
                    ielev_crs=-999,
                )
            )

    rxmin = float(np.min(lons))
    rxmax = float(np.max(lons))
    rymin = float(np.min(lats))
    rymax = float(np.max(lats))

    return Header(
        dataset_name="3D.DAT",
        dataset_version="2.1",
        dataset_message=options.dataset_message,
        comments=[Comment(c) for c in options.comments],
        flags=options.output_flags,
        projection=Projection(
            maptxt=options.maptxt,
            rlatc=options.rlatc,
            rlonc=options.rlonc,
            truelat1=options.truelat1,
            truelat2=options.truelat2,
            x1dmn=target.x0_km,
            y1dmn=target.y0_km,
            dxy=target.dx_km,
        ),
        domain=GridDomain(nx=nxp, ny=nyp, nz=nzp),
        model_options=ModelOptions(nland=options.nland),
        time_window=TimeWindow(
            ibyrm=first.year, ibmom=first.month, ibdym=first.day,
            ibhrm=first.hour, nhrsmm5=max(len(times) - 1, 1),
            nxp=nxp, nyp=nyp, nzp=nzp,
        ),
        extraction=Extraction(
            nx1=1, ny1=1, nx2=nxp, ny2=nyp, nz1=1, nz2=nzp,
            rxmin=rxmin, rxmax=rxmax, rymin=rymin, rymax=rymax,
        ),
        sigma_levels=sigma,
        grid_points=grid_points,
    )


# ---------------------------------------------------------------------------
# Per-timestep frame construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameOptions:
    """Per-run frame knobs that aren't derivable from the Dataset itself."""

    pressure_levels: Sequence[int]
    default_sst_k: float = 0.0
    default_snow_cover: int = 0
    # If True, when q2 is missing we compute it from rh2 + t2 + mslp via
    # the Tetens formula. If False, q2 falls back to 0.0.
    derive_q2_from_rh: bool = True


def _select_field(ds: Any, name: str) -> np.ndarray | None:
    """Return ds[name].values or None if the variable isn't present."""
    if name in ds.data_vars:
        return np.asarray(ds[name].values, dtype=np.float64)
    return None


def _require_field(ds: Any, name: str) -> np.ndarray:
    v = _select_field(ds, name)
    if v is None:
        raise KeyError(f"required field {name!r} missing from Dataset")
    return v


def _nan_to_zero(arr: np.ndarray | None, name: str) -> np.ndarray | None:
    """Replace NaN with 0 and log a warning with the count if any found.

    NaN must never reach the 3D.DAT writer: ``fmt_f`` would emit "nan"
    and CALMET's FORTRAN reader would crash on it; ``.astype(int)`` on
    a NaN raises a RuntimeWarning and produces undefined integer
    values.

    The most likely sources of NaN are:
      * Regridder mapping target cells outside source coverage (Qatar
        is well inside global GFS, so this should not happen here).
      * A pressure level present in some forecast hours but not others
        (the per-file streaming decode then has an unfilled slice).
      * Source GRIB messages with masked/bitmapped missing data.

    When we find any NaN we log the field name and count once per
    build_frames() call so the operator can investigate.
    """
    if arr is None:
        return arr
    nan_mask = np.isnan(arr)
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return arr
    total = arr.size
    _LOG.warning(
        "Field %s: %d / %d NaN values replaced with 0 before write "
        "(%.2f%% — investigate if non-trivial)",
        name, n_nan, total, 100.0 * n_nan / total,
    )
    return np.where(nan_mask, 0.0, arr)


def build_frames(
    ds: Any,
    options: FrameOptions,
) -> list[Frame]:
    """Build one Frame per timestep.

    Iteration order inside each frame matches CALWRF: J outer, I inner.
    Pressure-level records are ordered surface→top (descending pressure)
    to match how CALMET expects half-sigma layers.
    """
    times = [_as_utc_datetime(t) for t in np.asarray(ds["time"].values)]
    levels_in_ds = list(np.asarray(ds["level"].values, dtype=int))
    if list(options.pressure_levels) != levels_in_ds:
        raise ValueError(
            "options.pressure_levels does not match ds['level']: "
            f"{list(options.pressure_levels)} vs {levels_in_ds}"
        )

    nt = len(times)
    nz = len(levels_in_ds)
    ny = ds.sizes["y"]
    nx = ds.sizes["x"]

    # Pressure-level fields (required). NaN is replaced with 0 with a
    # one-line warning per field — guards against the writer producing
    # "nan" tokens that CALMET cannot parse, and against undefined
    # integer values from .astype(int) on a NaN float.
    t_pl = _nan_to_zero(_require_field(ds, "t_pl"), "t_pl")
    u_pl = _nan_to_zero(_require_field(ds, "u_pl"), "u_pl")
    v_pl = _nan_to_zero(_require_field(ds, "v_pl"), "v_pl")
    h_pl = _nan_to_zero(_require_field(ds, "h_pl"), "h_pl")
    rh_pl = _nan_to_zero(_require_field(ds, "rh_pl"), "rh_pl")
    q_pl = _nan_to_zero(_select_field(ds, "q_pl"), "q_pl")  # optional

    # Surface fields.
    mslp = _nan_to_zero(_require_field(ds, "mslp"), "mslp")
    u10 = _nan_to_zero(_require_field(ds, "u10"), "u10")
    v10 = _nan_to_zero(_require_field(ds, "v10"), "v10")
    t2 = _nan_to_zero(_require_field(ds, "t2"), "t2")
    q2 = _nan_to_zero(_select_field(ds, "q2"), "q2")        # optional
    rh2 = _nan_to_zero(_select_field(ds, "rh2"), "rh2")     # optional
    d2 = _nan_to_zero(_select_field(ds, "d2"), "d2")        # optional (ERA5)
    tp = _nan_to_zero(_select_field(ds, "tp"), "tp")        # optional
    dswrf = _nan_to_zero(_select_field(ds, "dswrf"), "dswrf")
    dlwrf = _nan_to_zero(_select_field(ds, "dlwrf"), "dlwrf")
    # ERA5 masks SST over land. pygrib may return masked values either as
    # NaN or as a large GRIB fill value (typically ~9.969e+36 or 9999),
    # not as a real temperature.  CALMET (ITWPROG=2) rejects 3D.DAT if any
    # cell has an invalid surface temperature, so we fall back to t2
    # (2 m air temperature) wherever SST is missing or out of physical
    # range.  Valid surface temperatures fall in [200, 350] K
    # (~ -73 °C to +77 °C), well beyond any real-world ocean range while
    # rejecting fill values and zero-fills.
    sst_raw = _select_field(ds, "sst")
    if sst_raw is not None:
        invalid = ~np.isfinite(sst_raw) | (sst_raw < 200.0) | (sst_raw > 350.0)
        sst = np.where(invalid, t2, sst_raw)
        n_invalid = int(np.sum(invalid))
        if n_invalid > 0:
            _LOG.info(
                "SST: %d/%d cells filled from t2 (land mask / fill values)",
                n_invalid, sst_raw.size,
            )
    else:
        # No SST field at all in the source — fall back to t2 everywhere
        # so CALMET ITWPROG=2 has something valid.  Operator can override
        # by setting frame.default_sst_k explicitly (handled below).
        _LOG.warning(
            "No SST field in source dataset; substituting t2 for all cells"
        )
        sst = t2.copy()

    # Pre-compute wind speed/direction for every (time, level, y, x) and
    # (time, y, x); vectorised here, sliced per cell below.
    ws_pl, wd_pl = wind_uv_to_speed_dir(u_pl, v_pl)
    ws10, wd10 = wind_uv_to_speed_dir(u10, v10)

    # Vapor mixing ratio at each pressure level.
    if q_pl is not None:
        vapmr_pl = mixing_ratio_from_specific_humidity_gkg(q_pl)
    else:
        # rh_pl is %, t_pl in K. Pressure at each level expands to grid shape.
        p_grid = np.asarray(
            options.pressure_levels, dtype=np.float64
        )[np.newaxis, :, np.newaxis, np.newaxis]
        p_grid = np.broadcast_to(p_grid, t_pl.shape)
        vapmr_pl = mixing_ratio_from_rh_t_p(rh_pl, t_pl, p_grid)

    # Vapor mixing ratio at 2 m.
    if q2 is not None:
        q2_gkg = mixing_ratio_from_specific_humidity_gkg(q2)
    elif options.derive_q2_from_rh and rh2 is not None:
        q2_gkg = mixing_ratio_from_rh_t_p(rh2, t2, mslp)
    elif options.derive_q2_from_rh and d2 is not None:
        # ERA5 path: no rh2, but dewpoint (d2) is available.
        rh2_from_d2 = rh_from_dewpoint(d2, t2)
        q2_gkg = mixing_ratio_from_rh_t_p(rh2_from_d2, t2, mslp)
    else:
        q2_gkg = np.zeros_like(t2)

    # All NaN was scrubbed above, so .astype(int) is safe here. We use
    # np.int64 explicitly for portability between Linux (int=long) and
    # Windows (int=int32) — the values fit in int32 anyway.
    rh_pl_int = np.clip(np.round(rh_pl), 0, 100).astype(np.int64)
    h_pl_int = np.round(h_pl).astype(np.int64)
    wd_pl_int = np.round(wd_pl).astype(np.int64) % 360

    frames: list[Frame] = []
    for ti, t in enumerate(times):
        cells: list[CellData] = []
        for j in range(ny):
            for i in range(nx):
                surface = SurfaceRecord(
                    year=t.year, month=t.month, day=t.day, hour=t.hour,
                    ix=i + 1, jx=j + 1,
                    pres=float(mslp[ti, j, i]),
                    rain=float(tp[ti, j, i]) if tp is not None else 0.0,
                    sc=options.default_snow_cover,
                    radsw=float(dswrf[ti, j, i]) if dswrf is not None else 0.0,
                    radlw=float(dlwrf[ti, j, i]) if dlwrf is not None else 0.0,
                    t2=float(t2[ti, j, i]),
                    q2=float(q2_gkg[ti, j, i]),
                    wd10=float(wd10[ti, j, i]),
                    ws10=float(ws10[ti, j, i]),
                    sst=float(sst[ti, j, i]),
                )
                levels: list[VerticalRecord] = []
                for k, p_hpa in enumerate(options.pressure_levels):
                    levels.append(
                        VerticalRecord(
                            pres=int(p_hpa),
                            z=int(h_pl_int[ti, k, j, i]),
                            tempk=float(t_pl[ti, k, j, i]),
                            wd=int(wd_pl_int[ti, k, j, i]),
                            ws=float(ws_pl[ti, k, j, i]),
                            w=0.0,
                            rh=int(rh_pl_int[ti, k, j, i]),
                            vapmr=float(vapmr_pl[ti, k, j, i]),
                        )
                    )
                cells.append(CellData(surface=surface, levels=tuple(levels)))
        frames.append(Frame(cells=tuple(cells)))
    return frames


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_utc_datetime(value: Any) -> datetime:
    """Convert a numpy.datetime64 / pandas Timestamp / datetime → naive UTC."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, np.datetime64):
        # Convert to nanosecond precision then to integer, then datetime.
        ns = value.astype("datetime64[ns]").astype("int64")
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).replace(
            tzinfo=None
        )
    # Pandas Timestamp or similar
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    raise TypeError(f"unsupported time value type: {type(value).__name__}")
