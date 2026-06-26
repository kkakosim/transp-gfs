"""ERA5 reanalysis GRIB2 field catalog.

Mirrors the structure of gfs_fields.py but with ecCodes shortNames and
unit conversions correct for ERA5 GRIB2 output from the Copernicus CDS.

Key differences vs GFS:
  geopotential  — ERA5 ``z`` (m²/s²) vs GFS ``gh`` (m); multiply by 1/g.
  precipitation — ERA5 ``tp`` in metres vs GFS in kg/m²; multiply by 100.
  radiation     — ERA5 ``ssrd``/``strd`` are accumulated J/m²; the reader
                  deaccumulates them to hourly-mean W/m².
  SST           — ERA5 has a native ``sst`` shortName at typeOfLevel=surface.
"""

from __future__ import annotations

from gfs2calmet.gfs_fields import GfsField


_G = 9.80665  # m/s²


# ---------------------------------------------------------------------------
# Pressure-level fields
# ---------------------------------------------------------------------------

ERA5_PRESSURE_LEVEL_FIELDS: tuple[GfsField, ...] = (
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
        role="h_pl", short_name="z", type_of_level="isobaricInhPa", level=None,
        native_units="m2/s2", target_units="m", multiplier=1.0 / _G,
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


# ---------------------------------------------------------------------------
# Surface / near-surface fields
# ---------------------------------------------------------------------------

ERA5_SURFACE_FIELDS: tuple[GfsField, ...] = (
    GfsField(
        role="mslp", short_name="msl",
        type_of_level="meanSea", level=0,
        native_units="Pa", target_units="hPa", multiplier=0.01,
    ),
    GfsField(
        role="u10", short_name="10u",
        type_of_level="heightAboveGround", level=10,
        native_units="m/s", target_units="m/s",
    ),
    GfsField(
        role="v10", short_name="10v",
        type_of_level="heightAboveGround", level=10,
        native_units="m/s", target_units="m/s",
    ),
    GfsField(
        role="t2", short_name="2t",
        type_of_level="heightAboveGround", level=2,
        native_units="K", target_units="K",
    ),
    # 2m specific humidity is NOT a standard ERA5 single-level variable;
    # q2 is derived downstream from 2m dewpoint + t2 + mslp instead.
    GfsField(
        role="d2", short_name="2d",
        type_of_level="heightAboveGround", level=2,
        native_units="K", target_units="K", optional=True,
    ),
    GfsField(
        role="tp", short_name="tp",
        type_of_level="surface", level=0,
        native_units="m", target_units="cm", multiplier=100.0,
        optional=True,
    ),
    # ssrd / strd are accumulated J/m²; stored here in native units and
    # deaccumulated to W/m² by deaccumulate_radiation() after assembly.
    GfsField(
        role="dswrf", short_name="ssrd",
        type_of_level="surface", level=0,
        native_units="J/m2", target_units="J/m2",
        optional=True,
    ),
    GfsField(
        role="dlwrf", short_name="strd",
        type_of_level="surface", level=0,
        native_units="J/m2", target_units="J/m2",
        optional=True,
    ),
    GfsField(
        role="sst", short_name="sst",
        type_of_level="surface", level=0,
        native_units="K", target_units="K", optional=True,
    ),
)


DEFAULT_ERA5_FIELDS: tuple[GfsField, ...] = (
    *ERA5_PRESSURE_LEVEL_FIELDS,
    *ERA5_SURFACE_FIELDS,
)


# ---------------------------------------------------------------------------
# CDS API variable names
# ---------------------------------------------------------------------------
# Copernicus CDS requires verbose parameter names in the download request,
# not ecCodes shortNames.  These map role → CDS variable name.

CDS_PRESSURE_LEVEL_VARIABLES: list[str] = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "relative_humidity",
    "specific_humidity",
]

CDS_SINGLE_LEVEL_VARIABLES: list[str] = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "sea_surface_temperature",
]
