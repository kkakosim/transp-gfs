"""Catalog of GFS GRIB2 fields needed to build a CALMET 3D.DAT file.

Each ``GfsField`` records how to extract one field from a GRIB2 file via
pygrib and how to convert it to the units the 3D.DAT writer expects.
The roles defined here are referenced by the downstream "frame builder"
stage that assembles VerticalRecord and SurfaceRecord objects.

Shortname / typeOfLevel conventions follow ecCodes (the parameter
database used by both pygrib and cfgrib). NCEP-style names (UGRD,
PRMSL, APCP, ...) live separately in HERBIE_IDX_PATTERNS because
Herbie's GRIB index subset uses those NCEP names rather than the
ecCodes shortNames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Field spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GfsField:
    """One GFS GRIB2 field to extract.

    Attributes
    ----------
    role : str
        Canonical role used downstream. See ``ROLES`` for the full set.
    short_name : str
        pygrib / ecCodes ``shortName`` filter (lower-case 'u', 't',
        'prmsl', ...).
    type_of_level : str
        pygrib / ecCodes ``typeOfLevel`` filter
        ('isobaricInhPa', 'heightAboveGround', 'surface', 'meanSea').
    level : int | None
        Specific level value (``10`` for 10-m wind, ``500`` for 500 hPa).
        ``None`` means "all levels of this typeOfLevel" — used for
        pressure-level fields where the caller selects the level set.
    native_units, target_units : str
        Documented for traceability; not enforced by the reader.
    multiplier, offset : float
        Linear conversion ``y = multiplier * x + offset`` applied to
        the raw values to produce target units.
    optional : bool
        When True the reader logs a warning if the field is missing
        instead of raising.
    """

    role: str
    short_name: str
    type_of_level: str
    level: int | None
    native_units: str
    target_units: str
    multiplier: float = 1.0
    offset: float = 0.0
    optional: bool = False

    def convert(self, value: float) -> float:
        return self.multiplier * value + self.offset


# Roles documented for the frame builder. The reader does not enforce
# this set — extending the catalog with new roles is the supported way
# to add fields.
ROLES: tuple[str, ...] = (
    # Pressure-level (3D)
    "u_pl",
    "v_pl",
    "t_pl",
    "h_pl",
    "rh_pl",
    "q_pl",
    # Surface (2D)
    "mslp",
    "u10",
    "v10",
    "t2",
    "q2",
    "rh2",
    "d2",
    "tp",
    "dswrf",
    "dlwrf",
)


# ---------------------------------------------------------------------------
# Default catalog
# ---------------------------------------------------------------------------


# Pressure-level winds, temperature, geopotential height, humidity.
# Note: for GFS the ``gh`` shortName already yields geopotential *height*
# in meters, so no /g conversion is needed (unlike ECMWF's ``z``).
PRESSURE_LEVEL_FIELDS: tuple[GfsField, ...] = (
    GfsField(
        role="u_pl", short_name="u", type_of_level="isobaricInhPa", level=None,
        native_units="m/s", target_units="m/s",
    ),
    GfsField(
        role="v_pl", short_name="v", type_of_level="isobaricInhPa", level=None,
        native_units="m/s", target_units="m/s",
    ),
    GfsField(
        role="t_pl", short_name="t", type_of_level="isobaricInhPa", level=None,
        native_units="K", target_units="K",
    ),
    GfsField(
        role="h_pl", short_name="gh", type_of_level="isobaricInhPa", level=None,
        native_units="m", target_units="m",
    ),
    GfsField(
        role="rh_pl", short_name="r", type_of_level="isobaricInhPa", level=None,
        native_units="%", target_units="%",
    ),
    GfsField(
        role="q_pl", short_name="q", type_of_level="isobaricInhPa", level=None,
        native_units="kg/kg", target_units="g/kg", multiplier=1000.0,
        optional=True,
    ),
)


# Surface / near-surface fields. PRMSL is "meanSea"; 2-m and 10-m fields
# are "heightAboveGround"; precip and radiation are "surface".
SURFACE_FIELDS: tuple[GfsField, ...] = (
    GfsField(
        role="mslp", short_name="prmsl", type_of_level="meanSea", level=0,
        native_units="Pa", target_units="hPa", multiplier=0.01,
    ),
    GfsField(
        role="u10", short_name="u", type_of_level="heightAboveGround", level=10,
        native_units="m/s", target_units="m/s",
    ),
    GfsField(
        role="v10", short_name="v", type_of_level="heightAboveGround", level=10,
        native_units="m/s", target_units="m/s",
    ),
    GfsField(
        role="t2", short_name="t", type_of_level="heightAboveGround", level=2,
        native_units="K", target_units="K",
    ),
    GfsField(
        role="q2", short_name="q", type_of_level="heightAboveGround", level=2,
        native_units="kg/kg", target_units="g/kg", multiplier=1000.0,
        optional=True,
    ),
    GfsField(
        role="rh2", short_name="r", type_of_level="heightAboveGround", level=2,
        native_units="%", target_units="%", optional=True,
    ),
    GfsField(
        role="d2", short_name="dpt", type_of_level="heightAboveGround", level=2,
        native_units="K", target_units="K", optional=True,
    ),
    GfsField(
        role="tp", short_name="tp", type_of_level="surface", level=0,
        native_units="kg/m^2 (mm)", target_units="cm", multiplier=0.1,
        optional=True,
    ),
    GfsField(
        role="dswrf", short_name="dswrf", type_of_level="surface", level=0,
        native_units="W/m^2", target_units="W/m^2", optional=True,
    ),
    GfsField(
        role="dlwrf", short_name="dlwrf", type_of_level="surface", level=0,
        native_units="W/m^2", target_units="W/m^2", optional=True,
    ),
)


DEFAULT_GFS_FIELDS: tuple[GfsField, ...] = (
    *PRESSURE_LEVEL_FIELDS,
    *SURFACE_FIELDS,
)


# ---------------------------------------------------------------------------
# Herbie idx subset patterns
# ---------------------------------------------------------------------------


# NCEP names as they appear in a GFS GRIB index (.idx) line, e.g.:
#   "1:0:d=2026011500:UGRD:10 m above ground:anl:"
# Herbie's ``search`` argument is a regex applied to those lines. We
# expose a per-role fragment so callers can build a single combined
# search to pull only the fields they need.
HERBIE_IDX_PATTERNS: dict[str, str] = {
    "u_pl":   r":UGRD:\d+ mb:",
    "v_pl":   r":VGRD:\d+ mb:",
    "t_pl":   r":TMP:\d+ mb:",
    "h_pl":   r":HGT:\d+ mb:",
    "rh_pl":  r":RH:\d+ mb:",
    "q_pl":   r":SPFH:\d+ mb:",
    "mslp":   r":PRMSL:mean sea level:",
    "u10":    r":UGRD:10 m above ground:",
    "v10":    r":VGRD:10 m above ground:",
    "t2":     r":TMP:2 m above ground:",
    "q2":     r":SPFH:2 m above ground:",
    "rh2":    r":RH:2 m above ground:",
    "d2":     r":DPT:2 m above ground:",
    "tp":     r":APCP:surface:",
    "dswrf":  r":DSWRF:surface:",
    "dlwrf":  r":DLWRF:surface:",
}


def herbie_search_for(roles: Iterable[str]) -> str:
    """Build a regex that matches any GFS idx line for the given roles.

    Raises KeyError if a role has no documented idx pattern.
    """
    parts: list[str] = []
    for r in roles:
        parts.append(f"(?:{HERBIE_IDX_PATTERNS[r]})")
    if not parts:
        raise ValueError("roles must be non-empty")
    return "|".join(parts)
