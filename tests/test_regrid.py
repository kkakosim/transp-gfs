"""Tests for the bilinear regridder.

We test in three layers:

1. ``bilinear_regrid_2d`` on plain numpy arrays (most assertions live
   here so failures point at the math, not the xarray plumbing).
2. ``TargetGrid.cell_meshes`` against a real pyproj UTM transform.
3. ``regrid_dataset`` on small xarray Datasets built by hand to
   confirm dim/coord/attr propagation.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from gfs2calmet.regrid import (
    TargetGrid,
    _canonicalize_source,
    _normalize_lon_to_source,
    bilinear_regrid_2d,
    regrid_dataset,
)


# ---------------------------------------------------------------------------
# Helpers for source arrays
# ---------------------------------------------------------------------------


def _gfs_like_source(
    lat_step: float = 0.25, lon_step: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A 5x5 lat/lon grid centred on Qatar, in GFS [0, 360] longitude."""
    lats = np.arange(26.0, 23.99, -lat_step)         # decreasing (GFS-style)
    lons = np.arange(50.0, 52.01, lon_step)          # ascending, [0, 360]
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    # Linear field: f(lat, lon) = 10*lat + lon — easy to verify bilinear.
    values = 10.0 * LAT + LON
    return lats, lons, values


# ---------------------------------------------------------------------------
# bilinear_regrid_2d
# ---------------------------------------------------------------------------


class TestBilinear:
    def test_recovers_linear_field_exactly(self) -> None:
        lats, lons, values = _gfs_like_source()
        # Pick target points strictly inside source bounds.
        tgt_lat = np.array([[25.1, 24.9], [24.3, 24.55]])
        tgt_lon = np.array([[50.6, 50.8], [51.2, 51.95]])
        out = bilinear_regrid_2d(lats, lons, values, tgt_lat, tgt_lon)
        expected = 10.0 * tgt_lat + tgt_lon
        assert np.allclose(out, expected, atol=1e-9)

    def test_identity_on_source_centers(self) -> None:
        lats, lons, values = _gfs_like_source()
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        out = bilinear_regrid_2d(lats, lons, values, LAT, LON)
        # Last row/col coincide with the upper boundary — bilinear still
        # exact since the field is linear. Lower-right corner cell is
        # the only one possibly clipped; compare interior.
        assert np.allclose(out[:-1, :-1], values[:-1, :-1], atol=1e-9)

    def test_handles_descending_source_lat(self) -> None:
        """GFS often returns latitudes from north to south."""
        lats_desc = np.linspace(26.0, 24.0, 9)            # 26 down to 24
        lons = np.linspace(50.0, 52.0, 9)
        LAT, LON = np.meshgrid(lats_desc, lons, indexing="ij")
        values = 10.0 * LAT + LON
        tgt_lat = np.array([[25.0]])
        tgt_lon = np.array([[51.0]])
        out = bilinear_regrid_2d(lats_desc, lons, values, tgt_lat, tgt_lon)
        assert np.allclose(out, [[10 * 25.0 + 51.0]], atol=1e-9)

    def test_target_outside_source_becomes_nan(self) -> None:
        lats, lons, values = _gfs_like_source()
        tgt_lat = np.array([[30.0]])  # well north of source
        tgt_lon = np.array([[51.0]])
        out = bilinear_regrid_2d(lats, lons, values, tgt_lat, tgt_lon)
        assert np.isnan(out).all()

    def test_lon_wrap_target_with_negative_lon_into_gfs_source(self) -> None:
        """A target lon at -178 must match a GFS source lon at 182."""
        lats = np.array([0.0, 1.0, 2.0])
        lons = np.array([180.0, 181.0, 182.0, 183.0, 184.0])  # GFS [0,360]
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        values = 10.0 * LAT + LON
        tgt_lat = np.array([[1.0]])
        tgt_lon = np.array([[-178.0]])  # equivalent to 182
        out = bilinear_regrid_2d(lats, lons, values, tgt_lat, tgt_lon)
        assert np.allclose(out, [[10.0 * 1.0 + 182.0]], atol=1e-9)

    def test_shape_mismatch_raises(self) -> None:
        lats = np.linspace(0, 1, 3)
        lons = np.linspace(0, 1, 4)
        bad_values = np.zeros((2, 2))
        with pytest.raises(ValueError, match="does not match"):
            bilinear_regrid_2d(lats, lons, bad_values, lats, lons)


class TestCanonicalizeSource:
    def test_flips_descending_lat(self) -> None:
        lat = np.array([2.0, 1.0, 0.0])
        lon = np.array([0.0, 1.0])
        vals = np.array([[20.0, 21.0], [10.0, 11.0], [0.0, 1.0]])
        out_lat, out_lon, out_vals = _canonicalize_source(lat, lon, vals)
        assert np.array_equal(out_lat, np.array([0.0, 1.0, 2.0]))
        # After flip, row 0 of vals should be what was row 2 (lat=0).
        assert np.array_equal(out_vals[0], np.array([0.0, 1.0]))
        assert np.array_equal(out_vals[-1], np.array([20.0, 21.0]))

    def test_leaves_already_ascending_alone(self) -> None:
        lat = np.array([0.0, 1.0])
        lon = np.array([10.0, 11.0])
        vals = np.array([[1.0, 2.0], [3.0, 4.0]])
        out_lat, out_lon, out_vals = _canonicalize_source(lat, lon, vals)
        assert np.array_equal(out_vals, vals)


class TestNormalizeLon:
    def test_shifts_negative_target_when_source_is_zero_360(self) -> None:
        src_lon = np.array([180.0, 181.0, 182.0, 183.0])
        tgt = np.array([[-178.0, -179.0]])
        out = _normalize_lon_to_source(tgt, src_lon)
        assert np.allclose(out, [[182.0, 181.0]])

    def test_passthrough_when_source_already_in_minus180_180(self) -> None:
        src_lon = np.array([-180.0, -179.0, -178.0])
        tgt = np.array([[-179.5]])
        out = _normalize_lon_to_source(tgt, src_lon)
        assert np.array_equal(out, tgt)


# ---------------------------------------------------------------------------
# TargetGrid + pyproj
# ---------------------------------------------------------------------------


class TestTargetGrid:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TargetGrid(crs="EPSG:32639", x0_km=0, y0_km=0,
                       dx_km=0, dy_km=1, nx=1, ny=1)
        with pytest.raises(ValueError, match="positive"):
            TargetGrid(crs="EPSG:32639", x0_km=0, y0_km=0,
                       dx_km=1, dy_km=1, nx=0, ny=1)

    def test_cell_centers_offset_half_cell_from_origin(self) -> None:
        g = TargetGrid(crs="EPSG:32639", x0_km=200, y0_km=2700,
                       dx_km=4, dy_km=4, nx=3, ny=2)
        x, y = g.cell_centers_km()
        assert np.array_equal(x, np.array([202.0, 206.0, 210.0]))
        assert np.array_equal(y, np.array([2702.0, 2706.0]))

    def test_utm_zone_39n_round_trips_qatar_corner(self) -> None:
        """UTM zone 39N at (500 km E, 2750 km N) is around 51E, 24.86N.

        This is the pyproj sanity check we already ran at the shell.
        """
        g = TargetGrid(
            crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
            x0_km=499, y0_km=2749, dx_km=2, dy_km=2, nx=1, ny=1,
        )
        X, Y, lat, lon = g.cell_meshes()
        assert X.shape == (1, 1) and Y.shape == (1, 1)
        # Center is (500, 2750) km; should be near 51E, 24.86N.
        assert lon[0, 0] == pytest.approx(51.0, abs=1e-3)
        assert lat[0, 0] == pytest.approx(24.865, abs=1e-2)


# ---------------------------------------------------------------------------
# regrid_dataset
# ---------------------------------------------------------------------------


def _make_source_dataset(
    times: list[datetime], levels: list[int]
) -> xr.Dataset:
    """Build a synthetic source Dataset with one pressure-level var
    (``t_pl``) and one surface var (``mslp``), both on a regular
    GFS-like 5x5 lat/lon grid."""
    lats, lons, _ = _gfs_like_source()
    nt, nl, ny, nx = len(times), len(levels), lats.size, lons.size

    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    # Field that depends on lat, lon, and (for t_pl) the level: easy
    # to validate after regridding.
    t_pl = np.empty((nt, nl, ny, nx))
    for ti in range(nt):
        for li, lv in enumerate(levels):
            t_pl[ti, li] = 10.0 * LAT + LON + 0.001 * lv

    mslp = np.empty((nt, ny, nx))
    for ti in range(nt):
        mslp[ti] = 1000.0 + 0.1 * LAT + 0.01 * LON

    return xr.Dataset(
        data_vars={
            "t_pl": (("time", "level", "latitude", "longitude"), t_pl,
                     {"units": "K", "native_units": "K"}),
            "mslp": (("time", "latitude", "longitude"), mslp,
                     {"units": "hPa"}),
        },
        coords={
            "time": ("time", np.array(times, dtype="datetime64[s]")),
            "level": ("level", np.asarray(levels, dtype=np.int32)),
            "latitude": ("latitude", lats),
            "longitude": ("longitude", lons),
        },
        attrs={"source": "GFS"},
    )


class TestRegridDataset:
    def test_pressure_level_var_keeps_time_level_and_gets_y_x(self) -> None:
        src = _make_source_dataset(
            times=[datetime(2026, 1, 15, 0)],
            levels=[1000, 500],
        )
        # Target grid that lands well inside Qatar (50-52E, 24-26N).
        # UTM zone 39N: 500 km E ≈ 51E; 2750 km N ≈ 24.86N.
        target = TargetGrid(
            crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
            x0_km=470, y0_km=2700, dx_km=10, dy_km=10, nx=3, ny=2,
        )
        out = regrid_dataset(src, target)
        assert out["t_pl"].dims == ("time", "level", "y", "x")
        assert out["t_pl"].shape == (1, 2, 2, 3)
        assert out["mslp"].dims == ("time", "y", "x")
        assert out["mslp"].shape == (1, 2, 3)

    def test_attributes_preserved_and_target_marker_added(self) -> None:
        src = _make_source_dataset([datetime(2026, 1, 15, 0)], [850])
        target = TargetGrid(
            crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
            x0_km=470, y0_km=2700, dx_km=10, dy_km=10, nx=2, ny=2,
        )
        out = regrid_dataset(src, target)
        assert out["t_pl"].attrs["units"] == "K"
        assert "+proj=utm" in out["t_pl"].attrs["regridded_to"]
        assert "+proj=utm" in out.attrs["projection"]
        assert "2x2" == out.attrs["target_shape"]

    def test_coords_carry_xy_and_latlon(self) -> None:
        src = _make_source_dataset([datetime(2026, 1, 15, 0)], [850])
        target = TargetGrid(
            crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
            x0_km=470, y0_km=2700, dx_km=10, dy_km=10, nx=3, ny=2,
        )
        out = regrid_dataset(src, target)
        assert out["x_km"].dims == ("x",)
        assert out["y_km"].dims == ("y",)
        assert out["latitude"].dims == ("y", "x")
        assert out["longitude"].dims == ("y", "x")
        # Latitude monotonically increases with y; longitude with x.
        assert (np.diff(out["latitude"].values, axis=0) > 0).all()
        assert (np.diff(out["longitude"].values, axis=1) > 0).all()

    def test_oob_target_yields_nan(self) -> None:
        """Target that lands far north of the source domain → NaN."""
        src = _make_source_dataset([datetime(2026, 1, 15, 0)], [850])
        # UTM Zone 39N at (500 km, 10000 km) is well north of Qatar (~near pole).
        target = TargetGrid(
            crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
            x0_km=470, y0_km=9990, dx_km=10, dy_km=10, nx=2, ny=2,
        )
        out = regrid_dataset(src, target)
        assert np.isnan(out["t_pl"].values).all()
        assert np.isnan(out["mslp"].values).all()

    def test_round_trip_linear_field_within_tolerance(self) -> None:
        """The synthetic field is linear in (lat, lon, level), so bilinear
        regridding should recover values to high precision at interior
        target points."""
        src = _make_source_dataset(
            times=[datetime(2026, 1, 15, 0)],
            levels=[1000, 850],
        )
        # A small inside-domain target near 51E, 24.86N.
        target = TargetGrid(
            crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
            x0_km=499, y0_km=2749, dx_km=2, dy_km=2, nx=2, ny=2,
        )
        out = regrid_dataset(src, target)
        # Use the target's own lat/lon coords to compute expected values.
        expected_t = (
            10.0 * out["latitude"].values
            + out["longitude"].values
        )  # at level=1000, +0.001*1000 = +1.0 added
        actual_lv1000 = out["t_pl"].sel(level=1000).values[0]
        # Bilinear over a smoothly-curved (UTM) target adds tiny error.
        assert np.allclose(actual_lv1000, expected_t + 1.0, atol=1e-3)

    def test_passes_through_non_grid_data_vars(self) -> None:
        src = _make_source_dataset([datetime(2026, 1, 15, 0)], [850])
        # Add a constant scalar variable that lacks lat/lon dims.
        src = src.assign(elapsed_hours=(("time",), np.array([0.0])))
        target = TargetGrid(
            crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
            x0_km=470, y0_km=2700, dx_km=10, dy_km=10, nx=2, ny=2,
        )
        out = regrid_dataset(src, target)
        assert out["elapsed_hours"].dims == ("time",)
        assert out["elapsed_hours"].values[0] == 0.0
