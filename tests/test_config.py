"""Tests for the YAML run-config loader.

We focus on: (1) the loader parses a valid config into the expected
dataclass tree, (2) missing required keys raise, (3) extra unknown
keys raise (no silent drift).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest

from gfs2calmet.config import RunConfig, load_config


_FULL_VALID_YAML = dedent("""
    target_grid:
      crs: "+proj=utm +zone=39 +ellps=WGS84 +units=m"
      x0_km: 400.0
      y0_km: 2700.0
      dx_km: 4.0
      dy_km: 4.0
      nx: 50
      ny: 50

    gfs:
      cycle: "2026-01-15T00:00"
      forecast_hours: [0, 1, 2, 3]
      pressure_levels: [1000, 925, 850, 500]
      product: pgrb2.0p25
      model: gfs
      output_dir: ./data/grib

    output_flags:
      ioutw: 0
      ioutq: 1
      ioutc: 0
      iouti: 0
      ioutg: 0
      iosrf: 0

    header:
      maptxt: UTM
      nland: 38
      default_elevation_m: 0
      default_landuse: 16
      dataset_message: "test"
      comments:
        - "comment 1"
        - "comment 2"
      truelat1: 0.0
      truelat2: 0.0
      rlatc: 0.0
      rlonc: 0.0

    frame:
      default_sst_k: 0.0
      default_snow_cover: 0
      derive_q2_from_rh: true

    output_path: ./out/test.3D.DAT
""").lstrip()


def _write_config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(body, encoding="utf-8")
    return p


class TestFullValidConfig:
    def test_loads_into_runconfig(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _FULL_VALID_YAML)
        cfg = load_config(p)
        assert isinstance(cfg, RunConfig)

    def test_target_grid_parsed(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _FULL_VALID_YAML)
        cfg = load_config(p)
        assert cfg.target_grid.nx == 50
        assert cfg.target_grid.dx_km == 4.0
        assert "+proj=utm" in cfg.target_grid.crs

    def test_gfs_cycle_parsed_to_datetime(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _FULL_VALID_YAML)
        cfg = load_config(p)
        assert cfg.gfs.cycle == datetime(2026, 1, 15, 0, 0)
        assert cfg.gfs.forecast_hours == [0, 1, 2, 3]
        assert cfg.gfs.pressure_levels == [1000, 925, 850, 500]

    def test_output_flags_parsed(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _FULL_VALID_YAML)
        cfg = load_config(p)
        assert cfg.output_flags.ioutq == 1
        assert cfg.output_flags.ioutc == 0

    def test_header_options_carry_flags_and_levels(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _FULL_VALID_YAML)
        cfg = load_config(p)
        assert cfg.header.nland == 38
        assert cfg.header.maptxt == "UTM"
        assert list(cfg.header.pressure_levels) == [1000, 925, 850, 500]
        # The flags object is shared between RunConfig.output_flags and
        # HeaderOptions.output_flags — same value, not duplicated state.
        assert cfg.header.output_flags is cfg.output_flags

    def test_frame_options_carry_levels(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _FULL_VALID_YAML)
        cfg = load_config(p)
        assert list(cfg.frame.pressure_levels) == [1000, 925, 850, 500]
        assert cfg.frame.derive_q2_from_rh is True

    def test_comments_preserved_in_order(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _FULL_VALID_YAML)
        cfg = load_config(p)
        assert list(cfg.header.comments) == ["comment 1", "comment 2"]


class TestSchemaEnforcement:
    def test_missing_root_section_raises(self, tmp_path: Path) -> None:
        # Drop the output_path key.
        bad = _FULL_VALID_YAML.replace(
            "output_path: ./out/test.3D.DAT\n", ""
        )
        p = _write_config(tmp_path, bad)
        with pytest.raises(KeyError, match="output_path"):
            load_config(p)

    def test_unknown_root_key_raises(self, tmp_path: Path) -> None:
        bad = _FULL_VALID_YAML + "rogue_section: 42\n"
        p = _write_config(tmp_path, bad)
        with pytest.raises(KeyError, match="rogue_section"):
            load_config(p)

    def test_missing_target_grid_key_raises(self, tmp_path: Path) -> None:
        bad = _FULL_VALID_YAML.replace("  nx: 50\n", "")
        p = _write_config(tmp_path, bad)
        with pytest.raises(KeyError, match="nx"):
            load_config(p)

    def test_unknown_target_grid_key_raises(self, tmp_path: Path) -> None:
        bad = _FULL_VALID_YAML.replace(
            "  ny: 50\n", "  ny: 50\n  rogue: 1\n",
        )
        p = _write_config(tmp_path, bad)
        with pytest.raises(KeyError, match="rogue"):
            load_config(p)

    def test_missing_output_flag_raises(self, tmp_path: Path) -> None:
        bad = _FULL_VALID_YAML.replace("  ioutq: 1\n", "")
        p = _write_config(tmp_path, bad)
        with pytest.raises(KeyError, match="ioutq"):
            load_config(p)

    def test_invalid_output_flag_value_raises(self, tmp_path: Path) -> None:
        # OutputFlags itself rejects values other than 0/1.
        bad = _FULL_VALID_YAML.replace("  ioutq: 1\n", "  ioutq: 2\n")
        p = _write_config(tmp_path, bad)
        with pytest.raises(ValueError, match="ioutq"):
            load_config(p)


class TestOptionalDefaults:
    def test_omitted_optional_header_keys_use_dataclass_defaults(
        self, tmp_path: Path
    ) -> None:
        # Strip out the optional header keys; only nland + maptxt remain
        # plus the loader's own dataclass defaults take over.
        minimal_header = dedent("""\
            header:
              maptxt: UTM
              nland: 38
        """)
        body = _FULL_VALID_YAML
        start = body.index("header:")
        end = body.index("\nframe:")
        body = body[:start] + minimal_header + body[end + 1:]
        p = _write_config(tmp_path, body)
        cfg = load_config(p)
        assert cfg.header.default_elevation_m == 0
        assert cfg.header.default_landuse == 16
        assert cfg.header.comments == ()

    def test_omitted_optional_frame_keys_use_defaults(self, tmp_path: Path) -> None:
        body = _FULL_VALID_YAML.replace(
            dedent("""\
                frame:
                  default_sst_k: 0.0
                  default_snow_cover: 0
                  derive_q2_from_rh: true
            """),
            "frame: {}\n",
        )
        p = _write_config(tmp_path, body)
        cfg = load_config(p)
        assert cfg.frame.default_sst_k == 0.0
        assert cfg.frame.default_snow_cover == 0
        assert cfg.frame.derive_q2_from_rh is True
