"""Dataclasses describing the contents of a CALMET 3D.DAT (v2.1) file.

Field names mirror the spec in CALPUFF v6 User Instructions Table 7-33.
Units and types are documented alongside each field.

No CALMET-specific defaults are baked in (per project scope decision):
callers must supply every field that is not flagged optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Header components
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comment:
    """One comment record (Header Records #2 .. NCOMM+2). Max 132 chars."""
    text: str


@dataclass(frozen=True)
class OutputFlags:
    """Header Record #NCOMM+3 — Format(6i3).

    Controls which optional fields are emitted in each NZP vertical
    record. Note the dependencies enforced by CALMET:

        IOUTI requires IOUTC
        IOUTG requires IOUTI

    IOSRF flags whether a separate 2D.DAT surface file is also produced
    (does not affect 3D.DAT contents).
    """
    ioutw: int  # vertical velocity W
    ioutq: int  # RH and vapor mixing ratio
    ioutc: int  # cloud + rain mixing ratios
    iouti: int  # ice + snow mixing ratios
    ioutg: int  # graupel mixing ratio
    iosrf: int  # 2D.DAT surface file flag

    def __post_init__(self) -> None:
        for name, v in (
            ("ioutw", self.ioutw), ("ioutq", self.ioutq),
            ("ioutc", self.ioutc), ("iouti", self.iouti),
            ("ioutg", self.ioutg), ("iosrf", self.iosrf),
        ):
            if v not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1, got {v}")
        if self.iouti and not self.ioutc:
            raise ValueError("IOUTI=1 requires IOUTC=1")
        if self.ioutg and not self.iouti:
            raise ValueError("IOUTG=1 requires IOUTI=1")

    @property
    def n_hydrometeors(self) -> int:
        """Number of hydrometeor mixing ratios emitted per vertical record.

        Used by the compression rule: when all are zero, the writer emits
        a single negative ``-N.000`` field where N = n_hydrometeors.
        """
        n = 0
        if self.ioutc:
            n += 2  # CLDMR, RAINMR
        if self.iouti:
            n += 2  # ICEMR, SNOWMR
        if self.ioutg:
            n += 1  # GRPMR
        return n


@dataclass(frozen=True)
class Projection:
    """Header Record #NCOMM+4 — Format(a4, f9.4, f10.4, 2f7.2, 2f10.3, f8.3, 2i4, i3).

    The MAPTXT code identifies the projection family used to define
    grid coordinates. Common values: ``LCC`` (Lambert Conformal),
    ``MER`` (Mercator), ``PS`` (Polar Stereographic), ``UTM`` (UTM).
    Stored as up to 3 characters (will be space-padded to 4).
    """
    maptxt: str          # 3-char projection code
    rlatc: float         # center latitude (deg N positive)
    rlonc: float         # center longitude (deg E positive)
    truelat1: float      # first true latitude (deg)
    truelat2: float      # second true latitude (deg)
    x1dmn: float         # SW dot point X (km) in original (un-extracted) domain
    y1dmn: float         # SW dot point Y (km) in original domain
    dxy: float           # grid spacing (km)


@dataclass(frozen=True)
class GridDomain:
    """Original-domain grid sizing, also part of Header Record #NCOMM+4."""
    nx: int              # cells in X (W-E) in original domain
    ny: int              # cells in Y (S-N) in original domain
    nz: int              # sigma layers in original domain


@dataclass(frozen=True)
class ModelOptions:
    """Header Record #NCOMM+5 — Format(30i3).

    These are MM5-specific physics-option codes. For non-MM5 sources
    (GFS, ECMWF, WRF, RUC, RAMS, TAPM) every code can be set to 0.
    NLAND (the last value) should be the number of landuse categories
    used by the downstream CALMET run — supply it explicitly.
    """
    inhyd: int = 0          # hydrostatic flag
    imphys: int = 0         # moisture / microphysics scheme
    icupa: int = 0          # cumulus parameterization
    ibltyp: int = 0         # PBL scheme
    ifrad: int = 0          # radiation scheme
    isoil: int = 0          # soil model
    ifddan: int = 0         # grid analysis nudging flag
    ifddaob: int = 0        # observation nudging flag
    flags_2d: tuple[int, ...] = field(
        default_factory=lambda: tuple([0] * 12)
    )                       # 12 flags only used by 2D.DAT
    nland: int = 0          # landuse category count

    def __post_init__(self) -> None:
        if len(self.flags_2d) != 12:
            raise ValueError(
                f"flags_2d must have exactly 12 entries, got {len(self.flags_2d)}"
            )


@dataclass(frozen=True)
class TimeWindow:
    """Header Record #NCOMM+6 — Format(i4, 3i2, i5, 3i4).

    ``ibyrm/ibmom/ibdym/ibhrm`` together encode the first valid UTC
    timestamp in the file (written as concatenated YYYYMMDDHH).
    ``nhrsmm5`` is the period length in hours. NXP/NYP/NZP describe the
    *extraction* subdomain (the actual data extent in this file).
    """
    ibyrm: int           # year
    ibmom: int           # month (1-12)
    ibdym: int           # day
    ibhrm: int           # hour (UTC, 0-23)
    nhrsmm5: int         # length of period in hours
    nxp: int             # cells in X in extraction subdomain
    nyp: int             # cells in Y in extraction subdomain
    nzp: int             # vertical levels (== number of vertical records)


@dataclass(frozen=True)
class Extraction:
    """Header Record #NCOMM+7 — Format(6i4, 2f10.4, 2f9.4)."""
    nx1: int             # SW corner I-index in original domain
    ny1: int             # SW corner J-index
    nx2: int             # NE corner I-index
    ny2: int             # NE corner J-index
    nz1: int             # lowest extracted layer K-index
    nz2: int             # highest extracted layer K-index
    rxmin: float         # westernmost longitude (deg E)
    rxmax: float         # easternmost longitude (deg E)
    rymin: float         # southernmost latitude (deg N)
    rymax: float         # northernmost latitude (deg N)


@dataclass(frozen=True)
class GridPoint:
    """One of the NXP*NYP grid-point records.

    Format(2i4, f9.4, f10.4, i5, i3, 1x, f9.4, f10.4, i5).
    """
    iindex: int          # I in extraction subdomain
    jindex: int          # J in extraction subdomain
    xlat_dot: float      # dot-point latitude (deg)
    xlong_dot: float     # dot-point longitude (deg)
    ielev_dot: int       # terrain elevation at dot point (m MSL)
    iland: int           # landuse category at cross point
    xlat_crs: float      # cross-point latitude
    xlong_crs: float     # cross-point longitude
    ielev_crs: int       # terrain elevation at cross point


@dataclass
class Header:
    """Aggregate of every Header Record (#1 .. NXP*NYP grid records)."""
    dataset_name: str          # "3D.DAT"
    dataset_version: str       # "2.1"
    dataset_message: str       # free text up to 64 chars
    comments: list[Comment]
    flags: OutputFlags
    projection: Projection
    domain: GridDomain
    model_options: ModelOptions
    time_window: TimeWindow
    extraction: Extraction
    sigma_levels: list[float]  # length must equal time_window.nzp
    grid_points: list[GridPoint]  # length must equal nxp*nyp

    def __post_init__(self) -> None:
        if len(self.sigma_levels) != self.time_window.nzp:
            raise ValueError(
                f"sigma_levels has {len(self.sigma_levels)} entries; "
                f"expected NZP={self.time_window.nzp}"
            )
        expected = self.time_window.nxp * self.time_window.nyp
        if len(self.grid_points) != expected:
            raise ValueError(
                f"grid_points has {len(self.grid_points)} entries; "
                f"expected NXP*NYP={expected}"
            )


# ---------------------------------------------------------------------------
# Per-timestep data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceRecord:
    """Per (time, IX, JX) surface record.

    Format(i4, 3i2, 2i3, f7.1, f5.2, i2, 3f8.1, f8.2, 3f8.1).
    Date components are written as zero-padded i2.2 so the leading
    integer column is a YYYYMMDDHH string. Fields not available from
    the source model should be set to 0.0.
    """
    year: int
    month: int
    day: int
    hour: int
    ix: int
    jx: int
    pres: float          # sea-level pressure (hPa)
    rain: float          # past-hour rainfall (cm)
    sc: int              # snow cover indicator (0 or 1)
    radsw: float         # SW radiation at surface (W/m^2)
    radlw: float         # LW radiation at top (W/m^2)
    t2: float            # 2 m temperature (K)
    q2: float            # 2 m specific humidity (g/kg)
    wd10: float          # 10 m wind direction (deg, met. convention)
    ws10: float          # 10 m wind speed (m/s)
    sst: float           # sea surface temperature (K)


@dataclass(frozen=True)
class VerticalRecord:
    """One of the NZP vertical records per grid cell per time.

    Format(i4, i6, f6.1, i4, f5.1, f6.2, i3, f5.2, 5f6.3).
    Several fields are optional and only written when the corresponding
    OutputFlags bit is set:
        w      — only if IOUTW = 1
        rh, vapmr — only if IOUTQ = 1
        cldmr, rainmr — only if IOUTC = 1 (which requires IOUTQ)
        icemr, snowmr — only if IOUTI = 1
        grpmr  — only if IOUTG = 1
    """
    pres: int            # pressure (mbar, integer)
    z: int               # elevation (m MSL, integer)
    tempk: float         # temperature (K)
    wd: int              # wind direction (deg, integer)
    ws: float            # wind speed (m/s)
    w: float = 0.0       # vertical velocity (m/s)
    rh: int = 0          # relative humidity (%)
    vapmr: float = 0.0   # vapor mixing ratio (g/kg)
    cldmr: float = 0.0   # cloud mixing ratio (g/kg)
    rainmr: float = 0.0  # rain mixing ratio (g/kg)
    icemr: float = 0.0   # ice mixing ratio (g/kg)
    snowmr: float = 0.0  # snow mixing ratio (g/kg)
    grpmr: float = 0.0   # graupel mixing ratio (g/kg)


@dataclass(frozen=True)
class CellData:
    """All vertical records for one grid cell at one time."""
    surface: SurfaceRecord
    levels: tuple[VerticalRecord, ...]


@dataclass(frozen=True)
class Frame:
    """One timestep worth of data.

    ``cells`` is ordered ``[(jx_offset, ix_offset)]`` row-major in the
    extraction subdomain (i.e. the writer iterates JX outer, IX inner,
    matching the order observed in CALMM5/CALWRF output).
    """
    cells: tuple[CellData, ...]


# Convenience type alias for the writer's frames argument.
Frames = Iterable[Frame]
