"""Load + validate the YAML run config used by the CLI.

No values are baked in — every field a CALMET run needs must appear in
the config file (per the project's "fully config-driven, no defaults"
decision). The dataclasses below describe the schema; the loader
raises on missing or extra keys so configuration drift surfaces early.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from gfs2calmet.dataset import OutputFlags
from gfs2calmet.frames import FrameOptions, HeaderOptions
from gfs2calmet.regrid import TargetGrid


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GfsCycleConfig:
    """Which forecast cycle to download and which hours to keep."""

    cycle: datetime
    start_date: datetime
    end_date: datetime
    forecast_hours: list[int]          # derived from start_date/end_date
    product: str = "pgrb2.0p25"
    model: str = "gfs"
    output_dir: str | None = None
    pressure_levels: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Full configuration for one ``python -m gfs2calmet`` invocation."""

    target_grid: TargetGrid
    gfs: GfsCycleConfig
    output_flags: OutputFlags
    header: HeaderOptions
    frame: FrameOptions
    output_path: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _require_keys(d: dict[str, Any], required: set[str], where: str) -> None:
    missing = required - d.keys()
    if missing:
        raise KeyError(f"{where}: missing keys {sorted(missing)}")
    extra = d.keys() - required
    if extra:
        raise KeyError(f"{where}: unknown keys {sorted(extra)}")


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _target_grid_from_dict(d: dict[str, Any]) -> TargetGrid:
    _require_keys(
        d, {"crs", "x0_km", "y0_km", "dx_km", "dy_km", "nx", "ny"},
        "target_grid",
    )
    return TargetGrid(
        crs=str(d["crs"]),
        x0_km=float(d["x0_km"]),
        y0_km=float(d["y0_km"]),
        dx_km=float(d["dx_km"]),
        dy_km=float(d["dy_km"]),
        nx=int(d["nx"]),
        ny=int(d["ny"]),
    )


def _gfs_cycle_from_dict(d: dict[str, Any]) -> GfsCycleConfig:
    required = {"cycle", "start_date", "end_date", "pressure_levels"}
    optional = {"product", "model", "output_dir"}
    extra = d.keys() - (required | optional)
    if extra:
        raise KeyError(f"gfs: unknown keys {sorted(extra)}")
    missing = required - d.keys()
    if missing:
        raise KeyError(f"gfs: missing keys {sorted(missing)}")

    cycle = _parse_datetime(d["cycle"])
    start = _parse_datetime(d["start_date"])
    end = _parse_datetime(d["end_date"])

    if start < cycle:
        raise ValueError(
            f"gfs.start_date ({start}) is before gfs.cycle ({cycle})"
        )
    if end < start:
        raise ValueError(
            f"gfs.end_date ({end}) is before gfs.start_date ({start})"
        )

    fxx_start = int((start - cycle).total_seconds() // 3600)
    fxx_end = int((end - cycle).total_seconds() // 3600)
    forecast_hours = list(range(fxx_start, fxx_end + 1))

    return GfsCycleConfig(
        cycle=cycle,
        start_date=start,
        end_date=end,
        forecast_hours=forecast_hours,
        pressure_levels=[int(p) for p in d["pressure_levels"]],
        product=str(d.get("product", "pgrb2.0p25")),
        model=str(d.get("model", "gfs")),
        output_dir=str(d["output_dir"]) if d.get("output_dir") else None,
    )


def _output_flags_from_dict(d: dict[str, Any]) -> OutputFlags:
    _require_keys(
        d, {"ioutw", "ioutq", "ioutc", "iouti", "ioutg", "iosrf"},
        "output_flags",
    )
    return OutputFlags(
        ioutw=int(d["ioutw"]),
        ioutq=int(d["ioutq"]),
        ioutc=int(d["ioutc"]),
        iouti=int(d["iouti"]),
        ioutg=int(d["ioutg"]),
        iosrf=int(d["iosrf"]),
    )


def _header_options_from_dict(
    d: dict[str, Any], flags: OutputFlags, pressure_levels: list[int],
) -> HeaderOptions:
    required = {"nland", "maptxt"}
    optional = {
        "default_elevation_m", "default_landuse", "dataset_message",
        "comments", "truelat1", "truelat2", "rlatc", "rlonc",
    }
    extra = d.keys() - (required | optional)
    if extra:
        raise KeyError(f"header: unknown keys {sorted(extra)}")
    missing = required - d.keys()
    if missing:
        raise KeyError(f"header: missing keys {sorted(missing)}")
    comments = tuple(str(c) for c in d.get("comments", []))
    return HeaderOptions(
        output_flags=flags,
        pressure_levels=pressure_levels,
        nland=int(d["nland"]),
        default_elevation_m=int(d.get("default_elevation_m", 0)),
        default_landuse=int(d.get("default_landuse", 16)),
        dataset_message=str(d.get("dataset_message", "Produced by gfs2calmet")),
        comments=comments,
        maptxt=str(d["maptxt"]),
        truelat1=float(d.get("truelat1", 0.0)),
        truelat2=float(d.get("truelat2", 0.0)),
        rlatc=float(d.get("rlatc", 0.0)),
        rlonc=float(d.get("rlonc", 0.0)),
    )


def _frame_options_from_dict(
    d: dict[str, Any], pressure_levels: list[int],
) -> FrameOptions:
    optional = {"default_sst_k", "default_snow_cover", "derive_q2_from_rh"}
    extra = d.keys() - optional
    if extra:
        raise KeyError(f"frame: unknown keys {sorted(extra)}")
    return FrameOptions(
        pressure_levels=pressure_levels,
        default_sst_k=float(d.get("default_sst_k", 0.0)),
        default_snow_cover=int(d.get("default_snow_cover", 0)),
        derive_q2_from_rh=bool(d.get("derive_q2_from_rh", True)),
    )


def load_config(path: str | os.PathLike[str]) -> RunConfig:
    """Parse a YAML run config into a typed RunConfig dataclass."""
    import yaml  # noqa: PLC0415 — local so the import path stays clean

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    _require_keys(
        raw,
        {
            "target_grid", "gfs", "output_flags",
            "header", "frame", "output_path",
        },
        "config root",
    )

    target = _target_grid_from_dict(raw["target_grid"])
    gfs = _gfs_cycle_from_dict(raw["gfs"])
    flags = _output_flags_from_dict(raw["output_flags"])
    header_opts = _header_options_from_dict(
        raw["header"], flags, gfs.pressure_levels
    )
    frame_opts = _frame_options_from_dict(raw["frame"], gfs.pressure_levels)
    output_path = str(raw["output_path"])

    return RunConfig(
        target_grid=target,
        gfs=gfs,
        output_flags=flags,
        header=header_opts,
        frame=frame_opts,
        output_path=output_path,
    )
