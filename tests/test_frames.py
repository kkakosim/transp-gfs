"""Tests for the frame builder.

Covers the meteorology conversions, the Header assembly, and the
order/structure of the emitted Frame objects.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from gfs2calmet import (
    Frame,
    OutputFlags,
    SurfaceRecord,
    TargetGrid,
    VerticalRecord,
)
from gfs2calmet.frames import (
    FrameOptions,
    HeaderOptions,
    build_frames,
    build_header,
    mixing_ratio_from_rh_t_p,
    mixing_ratio_from_specific_humidity_gkg,
    sigma_levels_from_pressures,
    wind_uv_to_speed_dir,
)


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


class TestWindUvToSpeedDir:
    @pytest.mark.parametrize(
        "u, v, expected_dir",
        [
            # Wind blowing from the north has v < 0 → dir = 0° ("from north").
            (0.0, -5.0, 0.0),
            # From east: u < 0 → dir = 90°.
            (-5.0, 0.0, 90.0),
            # From south: v > 0 → dir = 180°.
            (0.0, 5.0, 180.0),
            # From west: u > 0 → dir = 270°.
            (5.0, 0.0, 270.0),
        ],
    )
    def test_cardinal_directions(self, u, v, expected_dir) -> None:
        ws, wd = wind_uv_to_speed_dir(
            np.array([u]), np.array([v]),
        )
        assert ws[0] == pytest.approx(5.0)
        assert wd[0] == pytest.approx(expected_dir, abs=1e-9)

    def test_calm_defaults_to_360(self) -> None:
        ws, wd = wind_uv_to_speed_dir(np.array([0.0]), np.array([0.0]))
        assert ws[0] == 0.0
        assert wd[0] == 360.0

    def test_below_threshold_defaults_to_360(self) -> None:
        ws, wd = wind_uv_to_speed_dir(np.array([0.01]), np.array([0.01]))
        assert wd[0] == 360.0


class TestHumidityHelpers:
    def test_specific_humidity_to_mixing_ratio(self) -> None:
        # 10 g/kg specific humidity (q = 0.01) → w = 0.01 / 0.99 ≈ 10.101 g/kg
        out = mixing_ratio_from_specific_humidity_gkg(np.array([10.0]))
        assert out[0] == pytest.approx(10.0 / (1.0 - 0.01) * 1.0, rel=1e-3)
        assert out[0] == pytest.approx(10.10101, abs=1e-3)

    def test_rh_to_mixing_ratio_saturation(self) -> None:
        # At T = 20°C (293.15 K), P = 1000 hPa, RH = 100%:
        # es ≈ 23.4 hPa (Tetens), w ≈ 0.622*23.4/(1000-23.4) ≈ 14.9 g/kg.
        w = mixing_ratio_from_rh_t_p(
            np.array([100.0]), np.array([293.15]), np.array([1000.0])
        )
        assert w[0] == pytest.approx(14.9, rel=2e-2)

    def test_rh_zero_yields_zero_mixing_ratio(self) -> None:
        w = mixing_ratio_from_rh_t_p(
            np.array([0.0]), np.array([293.15]), np.array([1000.0])
        )
        assert w[0] == pytest.approx(0.0, abs=1e-9)


class TestSigmaLevels:
    def test_normalization_by_calwrf_reference(self) -> None:
        # sigma = P / 1013.0, rounded to 3 decimals.
        assert sigma_levels_from_pressures([1000, 850, 500]) == [
            0.987, 0.839, 0.494
        ]


# ---------------------------------------------------------------------------
# Header construction
# ---------------------------------------------------------------------------


def _make_regridded_dataset(
    nx: int = 3, ny: int = 2, nt: int = 1, levels=(1000, 850),
) -> xr.Dataset:
    """Minimal regrid-output-shaped Dataset.

    Required vars: t_pl, u_pl, v_pl, h_pl, rh_pl, mslp, u10, v10, t2.
    Optional vars (q2 etc.) omitted to exercise the RH-fallback path.
    """
    times = np.array(
        [datetime(2026, 1, 15, h) for h in range(nt)],
        dtype="datetime64[s]",
    )
    levels_arr = np.asarray(levels, dtype=np.int32)
    lats_2d = np.linspace(24.4, 24.5, ny)[:, None] * np.ones((1, nx))
    lons_2d = np.linspace(50.5, 50.6, nx)[None, :] * np.ones((ny, 1))
    x_km = 400.0 + 4.0 * np.arange(nx)
    y_km = 2700.0 + 4.0 * np.arange(ny)

    rng = np.random.default_rng(seed=42)
    shape_pl = (nt, len(levels), ny, nx)
    shape_sf = (nt, ny, nx)

    t_pl = 290.0 - 5.0 * np.arange(len(levels))[None, :, None, None] + np.zeros(shape_pl)
    u_pl = rng.normal(5.0, 1.0, shape_pl)
    v_pl = rng.normal(0.0, 1.0, shape_pl)
    h_pl = (
        100.0 + 1500.0 * np.arange(len(levels))[None, :, None, None]
        + np.zeros(shape_pl)
    )
    rh_pl = np.full(shape_pl, 60.0)

    mslp = np.full(shape_sf, 1012.5)
    u10 = np.full(shape_sf, 3.0)
    v10 = np.full(shape_sf, 1.0)
    t2 = np.full(shape_sf, 295.0)
    rh2 = np.full(shape_sf, 70.0)

    return xr.Dataset(
        data_vars={
            "t_pl": (("time", "level", "y", "x"), t_pl, {"units": "K"}),
            "u_pl": (("time", "level", "y", "x"), u_pl, {"units": "m/s"}),
            "v_pl": (("time", "level", "y", "x"), v_pl, {"units": "m/s"}),
            "h_pl": (("time", "level", "y", "x"), h_pl, {"units": "m"}),
            "rh_pl": (("time", "level", "y", "x"), rh_pl, {"units": "%"}),
            "mslp": (("time", "y", "x"), mslp, {"units": "hPa"}),
            "u10": (("time", "y", "x"), u10, {"units": "m/s"}),
            "v10": (("time", "y", "x"), v10, {"units": "m/s"}),
            "t2": (("time", "y", "x"), t2, {"units": "K"}),
            "rh2": (("time", "y", "x"), rh2, {"units": "%"}),
        },
        coords={
            "time": ("time", times),
            "level": ("level", levels_arr),
            "x_km": ("x", x_km),
            "y_km": ("y", y_km),
            "latitude": (("y", "x"), lats_2d),
            "longitude": (("y", "x"), lons_2d),
        },
        attrs={"source": "GFS"},
    )


def _target(nx: int = 3, ny: int = 2) -> TargetGrid:
    return TargetGrid(
        crs="+proj=utm +zone=39 +ellps=WGS84 +units=m",
        x0_km=400.0, y0_km=2700.0, dx_km=4.0, dy_km=4.0, nx=nx, ny=ny,
    )


class TestBuildHeader:
    def _opts(self) -> HeaderOptions:
        return HeaderOptions(
            output_flags=OutputFlags(ioutw=0, ioutq=1, ioutc=0, iouti=0,
                                     ioutg=0, iosrf=0),
            pressure_levels=[1000, 850],
            nland=38,
            default_elevation_m=10,
            default_landuse=16,
            dataset_message="test",
            comments=["a", "b"],
            maptxt="UTM",
        )

    def test_basic_fields(self) -> None:
        ds = _make_regridded_dataset()
        target = _target()
        header = build_header(ds, target, self._opts())
        assert header.dataset_name == "3D.DAT"
        assert header.dataset_version == "2.1"
        assert header.time_window.ibyrm == 2026
        assert header.time_window.ibmom == 1
        assert header.time_window.ibdym == 15
        assert header.time_window.ibhrm == 0
        assert header.time_window.nxp == 3
        assert header.time_window.nyp == 2
        assert header.time_window.nzp == 2

    def test_grid_points_use_dot_point_latlon_and_minus999_for_cross(self) -> None:
        ds = _make_regridded_dataset()
        target = _target()
        header = build_header(ds, target, self._opts())
        # NXP*NYP = 6 grid points in (j outer, i inner) order.
        assert len(header.grid_points) == 6
        first = header.grid_points[0]
        assert (first.iindex, first.jindex) == (1, 1)
        assert first.ielev_dot == 10
        assert first.iland == 16
        # Cross-point coords carry the CALWRF -999 marker.
        assert first.xlat_crs == -999.0
        assert first.ielev_crs == -999

    def test_sigma_levels_normalized_by_1013(self) -> None:
        ds = _make_regridded_dataset()
        target = _target()
        header = build_header(ds, target, self._opts())
        assert header.sigma_levels == [round(1000.0 / 1013.0, 3),
                                       round(850.0 / 1013.0, 3)]

    def test_extraction_bounds_from_2d_latlon(self) -> None:
        ds = _make_regridded_dataset()
        target = _target()
        header = build_header(ds, target, self._opts())
        e = header.extraction
        # The synthetic ds has lat in [24.4, 24.5] and lon in [50.5, 50.6].
        assert e.rxmin == pytest.approx(50.5)
        assert e.rxmax == pytest.approx(50.6)
        assert e.rymin == pytest.approx(24.4)
        assert e.rymax == pytest.approx(24.5)


# ---------------------------------------------------------------------------
# build_frames
# ---------------------------------------------------------------------------


class TestBuildFrames:
    def _opts(self) -> FrameOptions:
        return FrameOptions(
            pressure_levels=[1000, 850],
            default_sst_k=0.0,
            default_snow_cover=0,
            derive_q2_from_rh=True,
        )

    def test_one_frame_per_timestep(self) -> None:
        ds = _make_regridded_dataset(nt=3)
        frames = build_frames(ds, self._opts())
        assert len(frames) == 3

    def test_each_frame_has_nxp_nyp_cells_in_j_outer_i_inner_order(self) -> None:
        ds = _make_regridded_dataset(nx=3, ny=2)
        frames = build_frames(ds, self._opts())
        cells = frames[0].cells
        assert len(cells) == 6
        # First six should iterate IX inner: (1,1), (2,1), (3,1), (1,2), (2,2), (3,2).
        observed = [(c.surface.ix, c.surface.jx) for c in cells]
        assert observed == [
            (1, 1), (2, 1), (3, 1),
            (1, 2), (2, 2), (3, 2),
        ]

    def test_each_cell_has_nzp_vertical_records_in_descending_pressure(self) -> None:
        ds = _make_regridded_dataset()
        frames = build_frames(ds, self._opts())
        levels = [v.pres for v in frames[0].cells[0].levels]
        assert levels == [1000, 850]

    def test_wind_speed_and_direction_derived_from_u_v(self) -> None:
        ds = _make_regridded_dataset()
        frames = build_frames(ds, self._opts())
        # u10=3, v10=1 → ws = sqrt(10) ≈ 3.162, wd = (270 - atan2(1,3)*180/pi) % 360
        for cell in frames[0].cells:
            assert cell.surface.ws10 == pytest.approx(np.hypot(3.0, 1.0))
            expected_wd = (270.0 - np.degrees(np.arctan2(1.0, 3.0))) % 360.0
            assert cell.surface.wd10 == pytest.approx(expected_wd)

    def test_q2_derived_from_rh2_when_q2_absent(self) -> None:
        ds = _make_regridded_dataset()
        frames = build_frames(ds, self._opts())
        # rh2=70, t2=295, mslp=1012.5 → mixing ratio ≈ Tetens
        expected = mixing_ratio_from_rh_t_p(
            np.array([70.0]), np.array([295.0]), np.array([1012.5])
        )[0]
        for cell in frames[0].cells:
            assert cell.surface.q2 == pytest.approx(expected)

    def test_optional_surface_fields_default_to_zero(self) -> None:
        ds = _make_regridded_dataset()  # has no tp / dswrf / dlwrf / sst
        frames = build_frames(ds, self._opts())
        cell = frames[0].cells[0]
        assert cell.surface.rain == 0.0
        assert cell.surface.radsw == 0.0
        assert cell.surface.radlw == 0.0
        assert cell.surface.sc == 0
        # When SST is absent from the source dataset, we fall back to t2
        # so CALMET ITWPROG=2 always has a valid surface temperature.
        assert cell.surface.sst == cell.surface.t2

    def test_sst_land_mask_falls_back_to_t2(self) -> None:
        """ERA5 masks SST over land with fill values; those cells must
        fall back to t2 so CALMET ITWPROG=2 sees a valid temperature."""
        ds = _make_regridded_dataset()
        # Mix valid SST (over water) with fill values (over land).
        sst_arr = np.full(ds["t2"].shape, 9.969e36)  # GRIB fill
        sst_arr[0, 0, 0] = 301.5  # one valid water cell
        ds = ds.assign(sst=(("time", "y", "x"), sst_arr, {"units": "K"}))
        frames = build_frames(ds, self._opts())
        cell00 = frames[0].cells[0]                  # (j=0, i=0)
        assert cell00.surface.sst == pytest.approx(301.5)
        # Every other cell must have fallen back to its t2 value.
        for cell in frames[0].cells[1:]:
            assert cell.surface.sst == cell.surface.t2

    def test_vapmr_clamped_to_positive_floor(self) -> None:
        """CALMET's _waterp routine takes log(e) with e = p*w/(eps+w);
        if w == 0 the model dies with a domain error.  Upper-level air
        over a hot desert can have specific humidity small enough to
        round to 0.00 in the writer's F5.2 vapmr field.  build_frames
        must clamp vapmr (and q2) above zero so the written value is
        always >= 0.01 g/kg."""
        ds = _make_regridded_dataset()
        # rh_pl = 0% everywhere → mixing_ratio_from_rh_t_p returns 0.
        rh_pl_zero = np.zeros_like(ds["rh_pl"].values)
        ds = ds.assign(
            rh_pl=(("time", "level", "y", "x"), rh_pl_zero, ds["rh_pl"].attrs)
        )
        # rh2 = 0% → q2_gkg path also returns 0.
        rh2_zero = np.zeros_like(ds["rh2"].values)
        ds = ds.assign(
            rh2=(("time", "y", "x"), rh2_zero, ds["rh2"].attrs)
        )
        frames = build_frames(ds, self._opts())
        for cell in frames[0].cells:
            assert cell.surface.q2 >= 0.01
            for level in cell.levels:
                assert level.vapmr >= 0.01

    def test_sst_from_dataset_overrides_default(self) -> None:
        ds = _make_regridded_dataset()
        sst_k = np.full(ds["mslp"].shape, 302.5)
        ds = ds.assign(sst=(("time", "y", "x"), sst_k, {"units": "K"}))
        frames = build_frames(ds, self._opts())
        for cell in frames[0].cells:
            assert cell.surface.sst == pytest.approx(302.5)

    def test_pressure_levels_mismatch_raises(self) -> None:
        ds = _make_regridded_dataset()
        bad_opts = FrameOptions(pressure_levels=[1000, 700])  # ds has [1000,850]
        with pytest.raises(ValueError, match="does not match"):
            build_frames(ds, bad_opts)

    def test_missing_required_field_raises(self) -> None:
        ds = _make_regridded_dataset()
        ds_bad = ds.drop_vars("t_pl")
        with pytest.raises(KeyError, match="t_pl"):
            build_frames(ds_bad, self._opts())

    def test_nan_in_field_is_replaced_with_zero_and_warned(
        self, caplog
    ) -> None:
        """NaN must never reach the writer: fmt_f would emit 'nan' and
        CALMET would crash. build_frames replaces NaN with 0 and logs
        the field name + count."""
        ds = _make_regridded_dataset()
        # Inject NaN into rh_pl at (time=0, level=0, y=0, x=0).
        rh_corrupted = ds["rh_pl"].values.copy()
        rh_corrupted[0, 0, 0, 0] = np.nan
        ds = ds.assign(
            rh_pl=(("time", "level", "y", "x"), rh_corrupted, ds["rh_pl"].attrs)
        )

        with caplog.at_level("WARNING", logger="gfs2calmet.frames"):
            frames = build_frames(ds, self._opts())

        # No NaN reaches the produced VerticalRecord.
        vr = frames[0].cells[0].levels[0]
        assert not np.isnan(vr.tempk)
        assert not np.isnan(vr.ws)
        assert vr.rh == 0  # was NaN → 0 → int
        # Warning fired with the field name + count.
        msgs = [rec.message for rec in caplog.records]
        assert any("rh_pl" in m and "1" in m for m in msgs)

    def test_nan_in_surface_field_is_replaced_with_zero(self) -> None:
        ds = _make_regridded_dataset()
        mslp_corrupted = ds["mslp"].values.copy()
        mslp_corrupted[0, 0, 0] = np.nan
        ds = ds.assign(
            mslp=(("time", "y", "x"), mslp_corrupted, ds["mslp"].attrs)
        )
        frames = build_frames(ds, self._opts())
        # The corrupted cell now carries the substituted 0, not NaN.
        bad_cell = next(c for c in frames[0].cells
                        if c.surface.ix == 1 and c.surface.jx == 1)
        assert bad_cell.surface.pres == 0.0
        assert not np.isnan(bad_cell.surface.pres)
