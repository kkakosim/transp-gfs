"""Regrid a source lat/lon Dataset onto a CALMET driver grid.

The CALMET driver grid is described by ``TargetGrid``:

    * a projected CRS (typically UTM Zone N for regional runs, but any
      pyproj-compatible CRS works);
    * a south-west corner origin (km in projected coords);
    * uniform horizontal spacing (km);
    * shape (nx, ny).

We do simple bilinear interpolation from the source (regular lat/lon)
grid to the target cell centers. Target points outside the source
grid become NaN — callers should extend the GFS download domain to
cover the target plus a safety halo.

CRS handling:
    * Source longitudes may be in ``[0, 360]`` (GFS) or ``[-180, 180]``.
    * Source latitudes may be ascending or descending.
    Both are normalized internally before interpolation.

This module deliberately avoids scipy: bilinear over a uniform source
grid is a few lines of numpy and keeps the dependency footprint small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_pyproj() -> Any:
    import pyproj                          # noqa: PLC0415
    return pyproj


def _import_xarray() -> Any:
    import xarray                          # noqa: PLC0415
    return xarray


# ---------------------------------------------------------------------------
# Target grid spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetGrid:
    """CALMET driver grid in a projected CRS.

    All horizontal extents are in kilometers, matching CALMET's
    convention (X1DMN / Y1DMN / DXY in the 3D.DAT projection record).
    Internally we convert to meters for pyproj.

    Attributes
    ----------
    crs
        PROJ string or any pyproj.CRS input. Example for Qatar:
        ``"+proj=utm +zone=39 +ellps=WGS84 +units=m"``.
    x0_km, y0_km
        Coordinates of the SW corner of cell (0, 0) — i.e. the
        lower-left edge of the lower-left cell. Cell centers are
        offset by half a cell.
    dx_km, dy_km
        Cell spacing in km. CALMET assumes ``dx == dy``; we keep them
        separate so callers can detect violations.
    nx, ny
        Number of cells in X (east-west) and Y (south-north).
    """

    crs: str
    x0_km: float
    y0_km: float
    dx_km: float
    dy_km: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.dx_km <= 0 or self.dy_km <= 0:
            raise ValueError("dx_km and dy_km must be positive")

    # 1D cell-center arrays in projected km.
    def cell_centers_km(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x0_km + (np.arange(self.nx) + 0.5) * self.dx_km
        y = self.y0_km + (np.arange(self.ny) + 0.5) * self.dy_km
        return x, y

    # 2D meshes of (y, x) projected coordinates and (lat, lon).
    def cell_meshes(
        self, *, pyproj_module: Any | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X_km, Y_km, lat, lon)`` 2D arrays of shape (ny, nx).

        ``X_km`` and ``Y_km`` are the projected cell centers in km;
        ``lat`` and ``lon`` are the same points expressed as
        WGS84 (EPSG:4326) coordinates.
        """
        pyproj = pyproj_module or _import_pyproj()
        x_km, y_km = self.cell_centers_km()
        X_km, Y_km = np.meshgrid(x_km, y_km, indexing="xy")  # (ny, nx)

        transformer = pyproj.Transformer.from_crs(
            self.crs, "EPSG:4326", always_xy=True
        )
        # Flatten before passing to pyproj — older NumPy + pyproj combos
        # warn (and eventually error) when given 2D arrays of small size.
        x_flat = (X_km * 1000.0).ravel()
        y_flat = (Y_km * 1000.0).ravel()
        lon_flat, lat_flat = transformer.transform(x_flat, y_flat)
        lon = np.asarray(lon_flat, dtype=np.float64).reshape(X_km.shape)
        lat = np.asarray(lat_flat, dtype=np.float64).reshape(X_km.shape)
        return X_km, Y_km, lat, lon


# ---------------------------------------------------------------------------
# Bilinear interpolation primitives
# ---------------------------------------------------------------------------


def _canonicalize_source(
    src_lat: np.ndarray, src_lon: np.ndarray, src_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source arrays sorted into ascending lat and ascending lon.

    ``src_values`` is reordered to match. We *do not* attempt to
    handle a source grid that wraps the dateline (e.g. lon jumping
    from 359 to 1) — for that the caller should pre-shift.
    """
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        src_values = src_values[..., ::-1, :]
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]
        src_values = src_values[..., :, ::-1]
    return src_lat, src_lon, src_values


def _normalize_lon_to_source(
    tgt_lon: np.ndarray, src_lon: np.ndarray
) -> np.ndarray:
    """Shift target longitudes into the same range as the source.

    Handles the common GFS case where source lons are in [0, 360] but
    target lons (from pyproj inverse projection of a UTM grid) come
    out in [-180, 180].
    """
    src_min = float(src_lon[0])
    src_max = float(src_lon[-1])
    if src_min >= 0.0 and src_max > 180.0:
        # Source is on a [0, 360]-ish range — shift any negative target
        # longitudes up by 360 to align.
        return np.where(tgt_lon < 0.0, tgt_lon + 360.0, tgt_lon)
    return tgt_lon


def bilinear_regrid_2d(
    src_lat_1d: np.ndarray,
    src_lon_1d: np.ndarray,
    src_values_2d: np.ndarray,
    tgt_lat_2d: np.ndarray,
    tgt_lon_2d: np.ndarray,
) -> np.ndarray:
    """Bilinear-interpolate one 2D field from a regular lat/lon source
    grid onto an arbitrary 2D set of target points.

    The source grid must be regular (uniform spacing in each axis).
    Target points outside the source coverage become NaN.
    """
    src_lat_1d = np.asarray(src_lat_1d, dtype=np.float64)
    src_lon_1d = np.asarray(src_lon_1d, dtype=np.float64)
    src_values_2d = np.asarray(src_values_2d, dtype=np.float64)
    if src_values_2d.shape != (src_lat_1d.size, src_lon_1d.size):
        raise ValueError(
            f"src_values_2d {src_values_2d.shape} does not match "
            f"({src_lat_1d.size}, {src_lon_1d.size})"
        )

    src_lat_1d, src_lon_1d, src_values_2d = _canonicalize_source(
        src_lat_1d, src_lon_1d, src_values_2d
    )
    tgt_lon = _normalize_lon_to_source(tgt_lon_2d, src_lon_1d)

    dlat = src_lat_1d[1] - src_lat_1d[0]
    dlon = src_lon_1d[1] - src_lon_1d[0]
    nlat, nlon = src_values_2d.shape

    fi = (tgt_lon - src_lon_1d[0]) / dlon
    fj = (tgt_lat_2d - src_lat_1d[0]) / dlat

    i0 = np.floor(fi).astype(np.intp)
    j0 = np.floor(fj).astype(np.intp)

    # Validity is checked on the float fractional indices so that target
    # points lying exactly on the upper source boundary
    # (fi == nlon - 1, fj == nlat - 1) are valid — the clip + weight
    # math below reduces to the boundary cell value in that case.
    valid = (
        (fi >= 0.0) & (fi <= nlon - 1) &
        (fj >= 0.0) & (fj <= nlat - 1)
    )

    # Clip for safe indexing of invalid points (we mask them out below).
    i0c = np.clip(i0, 0, nlon - 2)
    j0c = np.clip(j0, 0, nlat - 2)
    wx = fi - i0c
    wy = fj - j0c

    v00 = src_values_2d[j0c,     i0c    ]
    v10 = src_values_2d[j0c,     i0c + 1]
    v01 = src_values_2d[j0c + 1, i0c    ]
    v11 = src_values_2d[j0c + 1, i0c + 1]

    out = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx       * (1.0 - wy) * v10
        + (1.0 - wx) * wy       * v01
        + wx       * wy         * v11
    )
    return np.where(valid, out, np.nan)


# ---------------------------------------------------------------------------
# Dataset-level regrid
# ---------------------------------------------------------------------------


def _regrid_var(
    var_values: np.ndarray,
    src_lat_1d: np.ndarray,
    src_lon_1d: np.ndarray,
    tgt_lat_2d: np.ndarray,
    tgt_lon_2d: np.ndarray,
) -> np.ndarray:
    """Regrid a variable whose last two axes are (lat, lon).

    Works for shapes (lat, lon), (time, lat, lon), and
    (time, level, lat, lon). Loops over leading axes in Python — fine
    for the modest forecast lengths involved (one cycle × tens of
    levels × hundreds of cells).
    """
    if var_values.ndim < 2:
        raise ValueError("variable must have at least (lat, lon) axes")
    leading_shape = var_values.shape[:-2]
    ny_t, nx_t = tgt_lat_2d.shape
    out = np.empty(leading_shape + (ny_t, nx_t), dtype=np.float64)
    if not leading_shape:
        return bilinear_regrid_2d(
            src_lat_1d, src_lon_1d, var_values, tgt_lat_2d, tgt_lon_2d
        )
    # Flatten leading dims for a single loop.
    flat_in = var_values.reshape(-1, var_values.shape[-2], var_values.shape[-1])
    flat_out = out.reshape(-1, ny_t, nx_t)
    for k in range(flat_in.shape[0]):
        flat_out[k] = bilinear_regrid_2d(
            src_lat_1d, src_lon_1d, flat_in[k], tgt_lat_2d, tgt_lon_2d
        )
    return out


def regrid_dataset(
    src_ds: Any,
    target: TargetGrid,
    *,
    pyproj_module: Any | None = None,
    xarray_module: Any | None = None,
    extra_attrs: Mapping[str, str] | None = None,
) -> Any:
    """Bilinear-regrid every data variable in ``src_ds`` to ``target``.

    The source Dataset is expected to come from
    :func:`gfs2calmet.gfs_reader.read_gfs_to_dataset` — i.e. it has
    1D ``latitude`` and ``longitude`` coordinates and data variables
    whose last two dims are ``(latitude, longitude)``.

    The returned Dataset uses dims ``(time, level, y, x)`` for
    pressure-level vars and ``(time, y, x)`` for surface vars, with
    2D ``latitude(y, x)`` and ``longitude(y, x)`` coordinates derived
    from the target grid plus 1D ``x_km(x)`` and ``y_km(y)``
    coordinates carrying the projected cell-center positions.

    Variable attributes from ``src_ds`` are preserved; ``regridded_to``
    is added with the target CRS string.
    """
    xr = xarray_module or _import_xarray()

    src_lat = np.asarray(src_ds["latitude"].values, dtype=np.float64)
    src_lon = np.asarray(src_ds["longitude"].values, dtype=np.float64)

    X_km, Y_km, tgt_lat, tgt_lon = target.cell_meshes(
        pyproj_module=pyproj_module
    )

    new_data_vars: dict[str, Any] = {}
    for name, var in src_ds.data_vars.items():
        # Re-order axes so (lat, lon) are last.
        if "latitude" not in var.dims or "longitude" not in var.dims:
            # Not a gridded field; pass through verbatim.
            new_data_vars[name] = var
            continue

        leading_dims = tuple(
            d for d in var.dims if d not in ("latitude", "longitude")
        )
        transposed = var.transpose(*leading_dims, "latitude", "longitude")
        regridded = _regrid_var(
            transposed.values,
            src_lat,
            src_lon,
            tgt_lat,
            tgt_lon,
        )
        new_dims = leading_dims + ("y", "x")
        attrs = dict(var.attrs)
        attrs["regridded_to"] = target.crs
        new_data_vars[name] = (new_dims, regridded, attrs)

    coords: dict[str, Any] = {
        "x_km": ("x", X_km[0, :]),
        "y_km": ("y", Y_km[:, 0]),
        "latitude": (("y", "x"), tgt_lat),
        "longitude": (("y", "x"), tgt_lon),
    }
    # Carry through any leading coords the source had (time, level, ...).
    for coord_name in src_ds.coords:
        if coord_name in ("latitude", "longitude"):
            continue
        if coord_name in coords:
            continue
        coords[coord_name] = src_ds.coords[coord_name]

    out_attrs = dict(src_ds.attrs)
    out_attrs["projection"] = target.crs
    out_attrs["target_origin_km"] = f"{target.x0_km},{target.y0_km}"
    out_attrs["target_spacing_km"] = f"{target.dx_km}x{target.dy_km}"
    out_attrs["target_shape"] = f"{target.ny}x{target.nx}"
    if extra_attrs:
        out_attrs.update(extra_attrs)

    return xr.Dataset(data_vars=new_data_vars, coords=coords, attrs=out_attrs)
