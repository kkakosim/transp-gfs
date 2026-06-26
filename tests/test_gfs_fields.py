"""Tests for the GFS field-spec catalog."""

from __future__ import annotations

import re

import pytest

from gfs2calmet.gfs_fields import (
    DEFAULT_GFS_FIELDS,
    HERBIE_IDX_PATTERNS,
    PRESSURE_LEVEL_FIELDS,
    ROLES,
    SURFACE_FIELDS,
    GfsField,
    herbie_search_for,
)


class TestCatalogIntegrity:
    def test_every_default_field_role_is_documented(self) -> None:
        for f in DEFAULT_GFS_FIELDS:
            assert f.role in ROLES, f"role {f.role} not in ROLES tuple"

    def test_every_role_has_an_idx_pattern(self) -> None:
        for r in ROLES:
            assert r in HERBIE_IDX_PATTERNS, f"role {r} missing idx pattern"

    def test_no_duplicate_roles_in_default_catalog(self) -> None:
        roles = [f.role for f in DEFAULT_GFS_FIELDS]
        assert len(roles) == len(set(roles))

    def test_pressure_level_fields_use_isobaricinhpa(self) -> None:
        for f in PRESSURE_LEVEL_FIELDS:
            assert f.type_of_level == "isobaricInhPa"
            assert f.level is None  # caller selects level set at read time

    def test_surface_fields_pin_a_specific_level(self) -> None:
        for f in SURFACE_FIELDS:
            assert f.level is not None


class TestConversions:
    def test_specific_humidity_converts_kgkg_to_gkg(self) -> None:
        q_pl = next(f for f in PRESSURE_LEVEL_FIELDS if f.role == "q_pl")
        assert q_pl.convert(0.008) == pytest.approx(8.0)

    def test_mslp_converts_pa_to_hpa(self) -> None:
        mslp = next(f for f in SURFACE_FIELDS if f.role == "mslp")
        assert mslp.convert(101325.0) == pytest.approx(1013.25)

    def test_precipitation_converts_mm_to_cm(self) -> None:
        tp = next(f for f in SURFACE_FIELDS if f.role == "tp")
        assert tp.convert(25.0) == pytest.approx(2.5)

    def test_no_conversion_for_temperature(self) -> None:
        t_pl = next(f for f in PRESSURE_LEVEL_FIELDS if f.role == "t_pl")
        assert t_pl.convert(293.15) == pytest.approx(293.15)


class TestHerbieSearch:
    def test_single_role_compiles(self) -> None:
        pattern = herbie_search_for(["t_pl"])
        re.compile(pattern)

    def test_combines_multiple_roles_with_alternation(self) -> None:
        pattern = herbie_search_for(["u_pl", "v_pl", "mslp"])
        assert "UGRD" in pattern
        assert "VGRD" in pattern
        assert "PRMSL" in pattern

    def test_matches_pressure_level_idx_line(self) -> None:
        pattern = re.compile(herbie_search_for(["t_pl"]))
        line = "12:3456:d=2026011500:TMP:500 mb:6 hour fcst:"
        assert pattern.search(line) is not None

    def test_does_not_match_unrelated_levels(self) -> None:
        # The "t_pl" pattern targets pressure-level temperature only;
        # a 2 m temperature line must not match it.
        pattern = re.compile(herbie_search_for(["t_pl"]))
        line = "12:3456:d=2026011500:TMP:2 m above ground:anl:"
        assert pattern.search(line) is None

    def test_matches_surface_field_idx_line(self) -> None:
        pattern = re.compile(herbie_search_for(["mslp"]))
        line = "5:9876:d=2026011500:PRMSL:mean sea level:anl:"
        assert pattern.search(line) is not None

    def test_empty_roles_raises(self) -> None:
        with pytest.raises(ValueError):
            herbie_search_for([])

    def test_unknown_role_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            herbie_search_for(["not_a_real_role"])


class TestGfsFieldDataclass:
    def test_frozen(self) -> None:
        f = GfsField(
            role="t_pl", short_name="t", type_of_level="isobaricInhPa",
            level=None, native_units="K", target_units="K",
        )
        with pytest.raises(Exception):  # FrozenInstanceError, but exact name varies
            f.role = "u_pl"  # type: ignore[misc]

    def test_short_names_normalizes_string_to_one_tuple(self) -> None:
        f = GfsField(
            role="t_pl", short_name="t", type_of_level="isobaricInhPa",
            level=None, native_units="K", target_units="K",
        )
        assert f.short_names == ("t",)

    def test_short_names_passes_tuple_through(self) -> None:
        f = GfsField(
            role="u10", short_name=("10u", "u"),
            type_of_level="heightAboveGround", level=10,
            native_units="m/s", target_units="m/s",
        )
        assert f.short_names == ("10u", "u")


class TestSurfaceFieldShortNames:
    """Surface fields use a tuple of acceptable ecCodes shortNames so
    we match whether the user's ecCodes install exposes the modern
    (``10u``, ``2t``, ``2d``) or legacy (``u``, ``t``, ``dpt``) form."""

    def test_u10_accepts_both_10u_and_u(self) -> None:
        f = next(x for x in SURFACE_FIELDS if x.role == "u10")
        assert "10u" in f.short_names
        assert "u" in f.short_names

    def test_v10_accepts_both_10v_and_v(self) -> None:
        f = next(x for x in SURFACE_FIELDS if x.role == "v10")
        assert "10v" in f.short_names
        assert "v" in f.short_names

    def test_t2_accepts_both_2t_and_t(self) -> None:
        f = next(x for x in SURFACE_FIELDS if x.role == "t2")
        assert "2t" in f.short_names
        assert "t" in f.short_names

    def test_d2_accepts_both_2d_and_dpt(self) -> None:
        f = next(x for x in SURFACE_FIELDS if x.role == "d2")
        assert "2d" in f.short_names
        assert "dpt" in f.short_names

    def test_mslp_accepts_both_prmsl_and_msl(self) -> None:
        f = next(x for x in SURFACE_FIELDS if x.role == "mslp")
        assert "prmsl" in f.short_names
        assert "msl" in f.short_names
