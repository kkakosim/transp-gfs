"""Tests for the GFS reader — uses a fake pygrib module so neither
pygrib nor real GRIB2 data is required at test time.

Strategy: build small ``FakeMessage`` objects with the same duck-typed
surface pygrib exposes (shortName, typeOfLevel, level, validDate,
values, latlons), pass them through a fake ``pygrib.open`` callable,
and verify the reader's selection / conversion / assembly behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gfs2calmet.gfs_fields import (
    DEFAULT_GFS_FIELDS,
    PRESSURE_LEVEL_FIELDS,
    SURFACE_FIELDS,
    GfsField,
)
from gfs2calmet.gfs_reader import (
    ExtractedMessage,
    _assemble_dataset,
    _extract_messages,
    download_gfs_cycle,
    read_gfs_to_dataset,
)


# ---------------------------------------------------------------------------
# Fake pygrib
# ---------------------------------------------------------------------------


# A simple synthetic 3x4 GFS-like lat/lon grid (just enough to test shape).
_NY, _NX = 3, 4
_LATS_1D = np.array([24.5, 24.4, 24.3])
_LONS_1D = np.array([50.5, 50.6, 50.7, 50.8])
_LATS_2D, _LONS_2D = np.meshgrid(_LATS_1D, _LONS_1D, indexing="ij")


@dataclass
class FakeMessage:
    shortName: str
    typeOfLevel: str
    level: int
    validDate: datetime
    values: np.ndarray
    _lats: np.ndarray = None  # type: ignore[assignment]
    _lons: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._lats is None:
            object.__setattr__(self, "_lats", _LATS_2D)
        if self._lons is None:
            object.__setattr__(self, "_lons", _LONS_2D)

    def latlons(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lats, self._lons


def _make_fake_pygrib(file_messages: dict[str, list[FakeMessage]]) -> Any:
    """Build a fake ``pygrib`` module whose ``open(path)`` returns the
    messages registered for that path."""
    def fake_open(path: str) -> Any:
        msgs = list(file_messages.get(str(path), []))
        result = MagicMock(name=f"GribFile<{path}>")
        result.__iter__.return_value = iter(msgs)
        result.close = MagicMock()
        return result

    return SimpleNamespace(open=fake_open)


_t = datetime(2026, 1, 15, 0, 0, 0)


def _msg(
    short: str, tol: str, lvl: int, t: datetime, fill: float
) -> FakeMessage:
    return FakeMessage(
        shortName=short,
        typeOfLevel=tol,
        level=lvl,
        validDate=t,
        values=np.full((_NY, _NX), fill, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# _extract_messages: selection + conversion
# ---------------------------------------------------------------------------


class TestExtractMessages:
    def test_selects_pressure_level_temperature(self) -> None:
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
        )
        msgs = [
            _msg("t", "isobaricInhPa", 1000, _t, 290.0),
            _msg("t", "isobaricInhPa", 500, _t, 250.0),
            _msg("u", "isobaricInhPa", 1000, _t, 5.0),  # not requested
            _msg("t", "heightAboveGround", 2, _t, 295.0),  # wrong level type
        ]
        fake = _make_fake_pygrib({"test.grib2": msgs})
        out = _extract_messages(["test.grib2"], fields, pygrib_module=fake)
        assert len(out) == 2
        assert {m.level for m in out} == {1000, 500}
        # No conversion: values are pass-through.
        assert np.allclose(out[0].values, 290.0) or np.allclose(out[0].values, 250.0)

    def test_applies_unit_multiplier(self) -> None:
        fields = (
            GfsField(role="mslp", short_name="prmsl", type_of_level="meanSea",
                     level=0, native_units="Pa", target_units="hPa",
                     multiplier=0.01),
        )
        msgs = [_msg("prmsl", "meanSea", 0, _t, 101325.0)]
        fake = _make_fake_pygrib({"f.grib2": msgs})
        out = _extract_messages(["f.grib2"], fields, pygrib_module=fake)
        assert len(out) == 1
        assert np.allclose(out[0].values, 1013.25)

    def test_filters_to_requested_levels(self) -> None:
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
        )
        msgs = [
            _msg("t", "isobaricInhPa", 1000, _t, 290.0),
            _msg("t", "isobaricInhPa", 850, _t, 280.0),
            _msg("t", "isobaricInhPa", 500, _t, 250.0),
        ]
        fake = _make_fake_pygrib({"f.grib2": msgs})
        out = _extract_messages(
            ["f.grib2"], fields, levels=[1000, 500], pygrib_module=fake
        )
        assert {m.level for m in out} == {1000, 500}

    def test_missing_required_field_raises(self) -> None:
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
        )
        fake = _make_fake_pygrib({"f.grib2": []})
        with pytest.raises(FileNotFoundError, match="t_pl"):
            _extract_messages(["f.grib2"], fields, pygrib_module=fake)

    def test_missing_optional_field_warns_only(self, caplog) -> None:
        fields = (
            GfsField(role="q_pl", short_name="q", type_of_level="isobaricInhPa",
                     level=None, native_units="kg/kg", target_units="g/kg",
                     multiplier=1000.0, optional=True),
        )
        fake = _make_fake_pygrib({"f.grib2": []})
        with caplog.at_level("WARNING"):
            out = _extract_messages(["f.grib2"], fields, pygrib_module=fake)
        assert out == []
        assert any("q_pl" in rec.message for rec in caplog.records)

    def test_normalizes_mixed_validdates_within_one_file(self) -> None:
        """GFS files mix instantaneous (validDate = end of step) with
        accumulated/averaged messages (validDate = interval midpoint).
        All messages from one file must land on the same valid_time
        in the output (the maximum) so the assembled Dataset has
        exactly one time slice per file."""
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
            GfsField(role="dswrf", short_name="dswrf", type_of_level="surface",
                     level=0, native_units="W/m^2", target_units="W/m^2",
                     optional=True),
        )
        instant = datetime(2026, 1, 15, 1, 0)        # forecast hour
        midpoint = datetime(2026, 1, 15, 0, 30)      # 0-1h average
        msgs = [
            _msg("t", "isobaricInhPa", 850, instant, 280.0),       # instantaneous
            _msg("dswrf", "surface", 0, midpoint, 100.0),          # averaged
        ]
        fake = _make_fake_pygrib({"f001.grib2": msgs})
        out = _extract_messages(["f001.grib2"], fields, pygrib_module=fake)
        # Both messages share the max valid_time, not their per-message ones.
        assert {m.valid_time for m in out} == {np.datetime64(instant, "s")}
        assert len(out) == 2

    def test_one_time_per_file_false_preserves_per_message_times(self) -> None:
        """ERA5 ships every requested hour inside a single GRIB; the GFS
        ``collapse to max time`` heuristic would discard all but one hour.
        With one_time_per_file=False each message keeps its own validDate."""
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
        )
        t0 = datetime(2022, 7, 1, 0, 0)
        t1 = datetime(2022, 7, 1, 1, 0)
        t2 = datetime(2022, 7, 1, 2, 0)
        msgs = [
            _msg("t", "isobaricInhPa", 850, t0, 280.0),
            _msg("t", "isobaricInhPa", 850, t1, 281.0),
            _msg("t", "isobaricInhPa", 850, t2, 282.0),
        ]
        fake = _make_fake_pygrib({"era5.grib2": msgs})
        out = _extract_messages(
            ["era5.grib2"], fields,
            pygrib_module=fake, one_time_per_file=False,
        )
        assert len(out) == 3
        assert {m.valid_time for m in out} == {
            np.datetime64(t0, "s"),
            np.datetime64(t1, "s"),
            np.datetime64(t2, "s"),
        }

    def test_matches_any_short_name_in_tuple(self) -> None:
        """Surface fields accept multiple acceptable shortNames so
        the reader works against any ecCodes version. Here the file
        has the legacy ``u`` shortName instead of the modern ``10u``."""
        fields = (
            GfsField(
                role="u10", short_name=("10u", "u"),
                type_of_level="heightAboveGround", level=10,
                native_units="m/s", target_units="m/s",
            ),
        )
        msgs = [_msg("u", "heightAboveGround", 10, _t, 3.5)]  # legacy name
        fake = _make_fake_pygrib({"f.grib2": msgs})
        out = _extract_messages(["f.grib2"], fields, pygrib_module=fake)
        assert len(out) == 1
        assert np.allclose(out[0].values, 3.5)

    def test_does_not_match_short_name_outside_tuple(self) -> None:
        fields = (
            GfsField(
                role="u10", short_name=("10u", "u"),
                type_of_level="heightAboveGround", level=10,
                native_units="m/s", target_units="m/s",
            ),
        )
        # 'gust' is the wind gust shortName — not in our tuple.
        msgs = [_msg("gust", "heightAboveGround", 10, _t, 8.0)]
        fake = _make_fake_pygrib({"f.grib2": msgs})
        with pytest.raises(FileNotFoundError, match="u10"):
            _extract_messages(["f.grib2"], fields, pygrib_module=fake)

    def test_iterates_multiple_files(self) -> None:
        fields = (
            GfsField(role="t2", short_name="t", type_of_level="heightAboveGround",
                     level=2, native_units="K", target_units="K"),
        )
        t1 = datetime(2026, 1, 15, 0)
        t2 = datetime(2026, 1, 15, 3)
        fake = _make_fake_pygrib({
            "f000.grib2": [_msg("t", "heightAboveGround", 2, t1, 295.0)],
            "f003.grib2": [_msg("t", "heightAboveGround", 2, t2, 297.0)],
        })
        out = _extract_messages(
            ["f000.grib2", "f003.grib2"], fields, pygrib_module=fake
        )
        assert {m.valid_time for m in out} == {
            np.datetime64(t1, "s"), np.datetime64(t2, "s")
        }


# ---------------------------------------------------------------------------
# _assemble_dataset
# ---------------------------------------------------------------------------


class TestAssembleDataset:
    def _make_t_pl_messages(self) -> list[ExtractedMessage]:
        out = []
        for lv, val in [(1000, 290.0), (850, 280.0), (500, 250.0)]:
            out.append(
                ExtractedMessage(
                    role="t_pl",
                    valid_time=np.datetime64(_t, "s"),
                    level=lv,
                    latitudes=_LATS_2D,
                    longitudes=_LONS_2D,
                    values=np.full((_NY, _NX), val),
                )
            )
        return out

    def test_pressure_level_dataset_has_time_level_lat_lon_dims(self) -> None:
        msgs = self._make_t_pl_messages()
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
        )
        ds = _assemble_dataset(msgs, fields)
        assert ds["t_pl"].dims == ("time", "level", "latitude", "longitude")
        assert ds["t_pl"].shape == (1, 3, _NY, _NX)
        # Levels are sorted descending (surface first).
        assert list(ds["level"].values) == [1000, 850, 500]

    def test_surface_dataset_has_time_lat_lon_dims_no_level(self) -> None:
        msgs = [
            ExtractedMessage(
                role="mslp",
                valid_time=np.datetime64(_t, "s"),
                level=0,
                latitudes=_LATS_2D,
                longitudes=_LONS_2D,
                values=np.full((_NY, _NX), 1013.0),
            )
        ]
        fields = (
            GfsField(role="mslp", short_name="prmsl", type_of_level="meanSea",
                     level=0, native_units="Pa", target_units="hPa",
                     multiplier=0.01),
        )
        ds = _assemble_dataset(msgs, fields)
        assert ds["mslp"].dims == ("time", "latitude", "longitude")
        assert ds["mslp"].shape == (1, _NY, _NX)
        assert "level" not in ds.coords  # no pressure-level fields present

    def test_attrs_carry_unit_metadata(self) -> None:
        msgs = self._make_t_pl_messages()
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
        )
        ds = _assemble_dataset(msgs, fields)
        assert ds["t_pl"].attrs["units"] == "K"
        assert ds["t_pl"].attrs["native_units"] == "K"
        assert ds["t_pl"].attrs["grib_short_name"] == "t"

    def test_grid_mismatch_raises(self) -> None:
        msgs = [
            ExtractedMessage(
                role="t2", valid_time=np.datetime64(_t, "s"), level=2,
                latitudes=_LATS_2D, longitudes=_LONS_2D,
                values=np.full((_NY, _NX), 295.0),
            ),
            ExtractedMessage(
                role="t2", valid_time=np.datetime64(_t, "s"), level=2,
                latitudes=np.zeros((4, 4)), longitudes=np.zeros((4, 4)),
                values=np.zeros((4, 4)),
            ),
        ]
        fields = (
            GfsField(role="t2", short_name="t", type_of_level="heightAboveGround",
                     level=2, native_units="K", target_units="K"),
        )
        with pytest.raises(ValueError, match="grid shape mismatch"):
            _assemble_dataset(msgs, fields)

    def test_empty_messages_raises(self) -> None:
        with pytest.raises(ValueError, match="no messages"):
            _assemble_dataset([], DEFAULT_GFS_FIELDS)


# ---------------------------------------------------------------------------
# End-to-end (read_gfs_to_dataset)
# ---------------------------------------------------------------------------


class TestReadGfsToDataset:
    def test_full_decode_with_pressure_and_surface_fields(self) -> None:
        fields = (
            GfsField(role="t_pl", short_name="t", type_of_level="isobaricInhPa",
                     level=None, native_units="K", target_units="K"),
            GfsField(role="mslp", short_name="prmsl", type_of_level="meanSea",
                     level=0, native_units="Pa", target_units="hPa",
                     multiplier=0.01),
        )
        msgs = [
            _msg("t", "isobaricInhPa", 1000, _t, 290.0),
            _msg("t", "isobaricInhPa", 500, _t, 250.0),
            _msg("prmsl", "meanSea", 0, _t, 101000.0),
        ]
        fake = _make_fake_pygrib({"cycle.grib2": msgs})
        ds = read_gfs_to_dataset(["cycle.grib2"], fields, pygrib_module=fake)

        assert ds["t_pl"].shape == (1, 2, _NY, _NX)
        assert ds["mslp"].shape == (1, _NY, _NX)
        assert ds["mslp"].values[0, 0, 0] == pytest.approx(1010.0)


# ---------------------------------------------------------------------------
# Download wrapper (Herbie)
# ---------------------------------------------------------------------------


class TestDownloadGfsCycle:
    def test_calls_herbie_with_subset_search_and_correct_fxx(
        self, tmp_path: Path
    ) -> None:
        fake_herbie = MagicMock(name="HerbieClass")
        fake_instance = MagicMock(name="HerbieInstance")
        fake_herbie.return_value = fake_instance
        fake_instance.download.return_value = tmp_path / "fake.grib2"

        roles = ["t_pl", "mslp"]
        paths = download_gfs_cycle(
            cycle="2026-01-15T00:00",
            fxx_hours=[0, 3, 6],
            roles=roles,
            output_dir=tmp_path,
            herbie_module=fake_herbie,
        )

        assert len(paths) == 3
        # One Herbie() per fxx
        assert fake_herbie.call_count == 3
        # Each call used the right cycle and fxx
        seen_fxx = sorted(call.kwargs["fxx"] for call in fake_herbie.call_args_list)
        assert seen_fxx == [0, 3, 6]
        # Each call used the gfs/pgrb2.0p25 defaults
        for call in fake_herbie.call_args_list:
            assert call.kwargs["model"] == "gfs"
            assert call.kwargs["product"] == "pgrb2.0p25"
        # Each download() got the subset search regex
        for call in fake_instance.download.call_args_list:
            search = call.kwargs.get("search") or (call.args[0] if call.args else "")
            assert "TMP" in search and "PRMSL" in search

    def test_raises_when_herbie_returns_none(self, tmp_path: Path) -> None:
        fake_herbie = MagicMock()
        fake_instance = MagicMock()
        fake_herbie.return_value = fake_instance
        fake_instance.download.return_value = None
        with pytest.raises(FileNotFoundError):
            download_gfs_cycle(
                cycle="2026-01-15T00:00",
                fxx_hours=[0],
                roles=["mslp"],
                output_dir=tmp_path,
                herbie_module=fake_herbie,
            )

    def test_accepts_datetime_cycle(self, tmp_path: Path) -> None:
        fake_herbie = MagicMock()
        fake_instance = MagicMock()
        fake_herbie.return_value = fake_instance
        fake_instance.download.return_value = tmp_path / "f.grib2"
        cycle = datetime(2026, 1, 15, 12, 0)
        download_gfs_cycle(
            cycle=cycle,
            fxx_hours=[0],
            roles=["mslp"],
            output_dir=tmp_path,
            herbie_module=fake_herbie,
        )
        # Cycle is passed through to Herbie as the datetime we supplied.
        assert fake_herbie.call_args.args[0] == cycle
