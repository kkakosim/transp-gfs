# gfs2calmet

Convert NOAA **GFS** GRIB2 forecasts into a **CALMET 3D.DAT v2.1** prognostic
input file (the format produced by CALWRF / CALMM5 / CALETA / CALRUC /
CALRAMS / CALTAPM). The output drops directly into a CALMET run with
`IPROG=14`, `ISTEPPGS=3600`.

- **Source**: GFS 0.25° pressure-level + surface fields, downloaded via
  Herbie's GRIB-index subset.
- **Sink**: A single text-format `3D.DAT` file written byte-compatible
  with the canonical CALWRF v2.0.3 reference implementation.
- **Host**: Built and tested on Windows; designed for production runs
  on Ubuntu 22.04+.

Status: writer, decoder, regridder, frame builder, CLI all implemented
and tested (128 tests). Several known gaps marked **TODO** below before
the chain is truly production-ready — none are blockers, all are
incremental.

---

## Table of contents

1. [Pipeline at a glance](#pipeline-at-a-glance)
2. [Install](#install)
3. [Quickstart](#quickstart)
4. [Architecture](#architecture)
5. [Configuration reference](#configuration-reference)
6. [Running the converter](#running-the-converter)
7. [3D.DAT output format](#3ddat-output-format)
8. [Wiring into CALMET](#wiring-into-calmet)
9. [Modifying the pipeline](#modifying-the-pipeline)
10. [Testing](#testing)
11. [Roadmap / known gaps](#roadmap--known-gaps)
12. [References](#references)

---

## Pipeline at a glance

```
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ Herbie       │    │ pygrib       │    │ pyproj +     │    │ frames.py    │    │ writer.py    │
   │ download     │ →  │ decode +     │ →  │ bilinear     │ →  │ U,V → ws,wd  │ →  │ FORTRAN-     │
   │ (GRIB2       │    │ unit convert │    │ regrid to    │    │ q → mixing   │    │ format       │
   │  subset)     │    │ → xarray.DS  │    │ UTM driver   │    │ ratio, dates │    │ 3D.DAT       │
   │              │    │              │    │ grid         │    │              │    │              │
   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        ↑                   ↑                   ↑                   ↑                   ↑
   gfs_reader.py       gfs_reader.py       regrid.py           frames.py           writer.py
   download_gfs_       read_gfs_to_        regrid_dataset()    build_header()      write_3ddat()
   cycle()             dataset()                               build_frames()
```

Every stage has tests with mocked dependencies, so changes to one
module don't require running the full chain to validate.

---

## Install

### Ubuntu / Linux — recommended (conda-forge)

The GRIB stack (`eccodes`, `pygrib`) is painful to build from source.
Conda-forge ships prebuilt binaries that work out of the box.

```bash
# 1. Get miniforge/mamba if you don't have it.
# 2. From the repo root:
conda env create -f environment.yml
conda activate gfs2calmet
pip install -e .

# Smoke test
gfs2calmet --help
pytest -q
```

### Ubuntu / Linux — alternative (apt + venv)

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

The script `apt install`s `libeccodes-dev`, creates `.venv`,
`pip install`s the project with pipeline + test extras, and verifies
imports. Needs `sudo`.

### Windows

```powershell
# Miniforge / mamba strongly recommended over plain pip.
conda env create -f environment.yml
conda activate gfs2calmet
pip install -e .

# Smoke test
gfs2calmet --help
pytest -q
```

Plain `pip install pygrib` on Windows tends to fail because eccodes
wheels aren't available. Conda-forge solves this.

### Minimal install (writer + format helpers only)

```bash
pip install numpy pytest xarray pyproj pyyaml
# (skips pygrib + herbie; tests for those modules use mocks)
pytest -q
```

This is the install path for iterating on the writer or format
helpers without touching the full download stack.

---

## Quickstart

```bash
# 1. Copy and edit the example config (or use the Qatar one as-is).
cp config_qatar.yaml my_run.yaml
$EDITOR my_run.yaml      # change `gfs.cycle`, output path, etc.

# 2. Run the full chain.
gfs2calmet my_run.yaml -v
#   → downloads ~73 GFS files into ./data/grib (configurable)
#   → decodes with pygrib
#   → bilinear-regrids onto your UTM grid
#   → writes ./out/QATAR_GFS_<cycle>.3D.DAT

# 3. Feed the output to CALMET by editing CALMET.INP:
#    M3DDAT = ./out/QATAR_GFS_<cycle>.3D.DAT
#    IBYR/IBMO/IBDY/IBHR + IEYR/... matching the cycle range
#    (everything else from your existing CALMET.INP stays unchanged)
```

Two convenience wrappers:

```bash
bash scripts/run_qatar.sh                    # uses config_qatar.yaml
bash scripts/run_qatar.sh other_config.yaml  # different config
bash scripts/run_qatar.sh --skip-download    # reuse cached GRIB2
```

---

## Architecture

| Module | Responsibility | Tested via |
|---|---|---|
| [`fortran_format.py`](src/gfs2calmet/fortran_format.py) | FORTRAN `Iw`, `Iw.m`, `Fw.d`, `Aw`, `nX` edit descriptors with overflow→asterisks | [`test_fortran_format.py`](tests/test_fortran_format.py) |
| [`dataset.py`](src/gfs2calmet/dataset.py) | Frozen dataclasses for every 3D.DAT record (Header, OutputFlags, Projection, GridPoint, SurfaceRecord, VerticalRecord, Frame, CellData) | Used everywhere |
| [`writer.py`](src/gfs2calmet/writer.py) | Serializes Header + Frames to a 3D.DAT v2.1 file; aligned with CALWRF v2.0.3 source (`NCOMM` as `i4`, hydrometeor compression rule, sub-threshold clamp) | [`test_writer_3ddat.py`](tests/test_writer_3ddat.py) |
| [`gfs_fields.py`](src/gfs2calmet/gfs_fields.py) | Catalog of 16 GFS roles (6 pressure-level + 10 surface) with GRIB filters, units, converters, and Herbie idx patterns | [`test_gfs_fields.py`](tests/test_gfs_fields.py) |
| [`gfs_reader.py`](src/gfs2calmet/gfs_reader.py) | `download_gfs_cycle()` (Herbie) + `read_gfs_to_dataset()` (pygrib → xarray); lazy imports both deps | [`test_gfs_reader.py`](tests/test_gfs_reader.py) (fake pygrib + fake Herbie) |
| [`regrid.py`](src/gfs2calmet/regrid.py) | `TargetGrid` dataclass + pure-numpy bilinear regridder + xarray `regrid_dataset()`; handles descending-lat and [0,360]↔[-180,180] lon | [`test_regrid.py`](tests/test_regrid.py) |
| [`frames.py`](src/gfs2calmet/frames.py) | Wind U,V → speed + met direction; q → mixing ratio (or Tetens fallback); `build_header()` + `build_frames()` | [`test_frames.py`](tests/test_frames.py) |
| [`config.py`](src/gfs2calmet/config.py) | Strict YAML loader. Missing keys raise; unknown keys raise (no silent drift) | [`test_config.py`](tests/test_config.py) |
| [`cli.py`](src/gfs2calmet/cli.py) | `python -m gfs2calmet` / `gfs2calmet` entry point; ties the four pipeline stages together | [`test_cli.py`](tests/test_cli.py) (all stages mocked) |

Project layout:

```
gfs2calmet/
├── README.md                       ← you are here
├── pyproject.toml                  ← editable install, console_scripts entry
├── requirements.txt                ← flat dep list (alternative to pyproject)
├── environment.yml                 ← conda env (recommended)
├── config_example.yaml             ← reference config with every key documented
├── config_qatar.yaml               ← populated from your CALMET.INP
├── CALMET.INP                      ← your existing WRF-driven CALMET control file
├── scripts/
│   ├── setup_ubuntu.sh             ← apt + venv setup
│   └── run_qatar.sh                ← wrapper around `gfs2calmet`
├── src/gfs2calmet/                 ← package source
│   ├── __init__.py                 ← public API re-exports
│   ├── __main__.py                 ← `python -m gfs2calmet`
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── fortran_format.py
│   ├── frames.py
│   ├── gfs_fields.py
│   ├── gfs_reader.py
│   ├── regrid.py
│   └── writer.py
├── tests/                          ← 128 tests, all offline (mocks for pygrib/Herbie)
└── CALWRF_v2.0.3_L190426/          ← reference FORTRAN source used for cross-validation
    └── code/calwrf.f               ← canonical 3D.DAT writer (FORMAT statements quoted in our code)
```

---

## Configuration reference

The YAML is **strictly validated**: missing keys raise, unknown keys
raise. No defaults are quietly substituted. Edit [config_example.yaml](config_example.yaml)
or [config_qatar.yaml](config_qatar.yaml).

### `target_grid` — CALMET driver grid

| Key | Type | Meaning |
|---|---|---|
| `crs` | string | PROJ string or any pyproj-compatible CRS. UTM example: `"+proj=utm +zone=39 +ellps=WGS84 +datum=WGS84 +units=m"` |
| `x0_km` | float | SW corner X of cell (1,1) in projected km. Matches `XORIGKM` in CALMET.INP. |
| `y0_km` | float | SW corner Y, matches `YORIGKM`. |
| `dx_km`, `dy_km` | float | Cell spacing in km. CALMET requires `dx == dy`. |
| `nx`, `ny` | int | Cell counts in X (W→E) and Y (S→N). |

The 3D.DAT grid does **not** have to match CALMET's grid; CALMET
interpolates internally. For GFS 0.25° (~27 km) input, 4 km is a
reasonable target (~7× oversample).

### `gfs` — what to download and which levels to keep

| Key | Type | Meaning |
|---|---|---|
| `cycle` | ISO datetime | UTC GFS cycle, e.g. `"2026-01-15T00:00"`. NCEP cycles: 00/06/12/18 UTC. |
| `forecast_hours` | list[int] | Forecast hours to include. NCEP GFS is hourly through f120. |
| `pressure_levels` | list[int] | Pressure levels (hPa) to keep, **descending order** (surface→top). |
| `product` | string | GFS product name. Default `pgrb2.0p25` (0.25° global). |
| `model` | string | Herbie model id. Default `gfs`. |
| `output_dir` | string | Where Herbie caches downloaded GRIB2 files. Re-used across runs. |

### `output_flags` — which 3D.DAT optional vertical fields to write

All values are 0 or 1. Dependencies enforced by `OutputFlags.__post_init__`:
`IOUTI` requires `IOUTC`; `IOUTG` requires `IOUTI`.

| Flag | Effect when 1 |
|---|---|
| `ioutw` | Emit vertical velocity W (m/s). **Currently leave 0** — GFS provides Omega (Pa/s) and conversion is not yet implemented. |
| `ioutq` | Emit RH (%) + vapor mixing ratio (g/kg). Set 1 when `IRHPROG=1` in CALMET.INP. |
| `ioutc` | Emit cloud + rain mixing ratios. GFS doesn't carry these at pressure levels; leave 0. |
| `iouti` | Emit ice + snow mixing ratios. Leave 0. |
| `ioutg` | Emit graupel mixing ratio. Leave 0. |
| `iosrf` | Also produce a 2D.DAT companion file. Leave 0 — CALMET reads 2D fields from the 3D.DAT surface block. |

### `header` — projection record + grid-point metadata

| Key | Type | Meaning |
|---|---|---|
| `maptxt` | string (≤4 chars) | Verbatim into 3D.DAT projection record. CALMET recognizes `LCC`, `MER`, `PS`, `UTM`. |
| `nland` | int | Number of landuse categories (USGS-38 → 38, USGS-25 → 25). Match GEO.DAT. |
| `default_elevation_m` | int | Per-cell terrain elevation placeholder. CALMET uses GEO.DAT for real terrain — this field is informational. |
| `default_landuse` | int | Per-cell landuse placeholder. Same story. `16` = USGS water, matching `ILUOC3D=16`. |
| `dataset_message` | string | Goes into 3D.DAT header record 1, 64-char field. |
| `comments` | list[string] | Each becomes one header comment record (132-char field). |
| `truelat1`, `truelat2`, `rlatc`, `rlonc` | float | LCC / PS projection parameters. Set to 0 for UTM. |

### `frame` — per-timestep conversion knobs

| Key | Type | Meaning |
|---|---|---|
| `default_sst_k` | float | Sea-surface temperature default when no SST product is downloaded. 0 leaves the field unused. |
| `default_snow_cover` | int | 0 (no snow) or 1. Doha rarely needs anything else. |
| `derive_q2_from_rh` | bool | When `q2` isn't fetched, derive it from `rh2 + t2 + mslp` via Tetens. `false` falls back to 0. |

### `output_path`

Final 3D.DAT path. Convention: `QATAR_GFS_<YYYYMMDDHH>.3D.DAT`.

---

## Running the converter

```
gfs2calmet CONFIG [--skip-download] [-v|-vv]

positional:
  CONFIG           path to a run YAML

options:
  --skip-download  reuse existing GRIB2 files in gfs.output_dir
                   (handy when iterating on grid/output flags without
                    re-downloading hundreds of MB)
  -v               INFO logging
  -vv              DEBUG logging
```

The CLI exits 0 on success, non-zero on the first error. All four
pipeline stages log at INFO level.

### Common workflows

```bash
# Fresh run for a new cycle.
gfs2calmet config_qatar.yaml -v

# Iterate on grid resolution. Reuse GRIB2 you've already downloaded.
$EDITOR config_qatar.yaml       # change dx_km/nx
gfs2calmet config_qatar.yaml --skip-download -v

# Quick smoke run — small fxx list + tiny grid.
cp config_qatar.yaml smoke.yaml
$EDITOR smoke.yaml              # set forecast_hours: [0, 1, 2], nx: 5, ny: 5
gfs2calmet smoke.yaml -vv

# Run the full test suite.
pytest -q

# Inspect a generated 3D.DAT — first 30 lines is the entire header block.
head -n 50 out/QATAR_GFS_*.3D.DAT
```

---

## 3D.DAT output format

The writer targets **3D.DAT version 2.1** as defined in
*CALPUFF v6 User Instructions*, Section 7.7, Tables 7-32 / 7-33.
We cross-checked every FORMAT statement against the canonical
[CALWRF v2.0.3 source](CALWRF_v2.0.3_L190426/code/calwrf.f) shipped
in this repo. Three discrepancies between the manual and the
real-world implementation are handled the CALWRF way:

1. `NCOMM` is written with `(i4)` — not free-format. ([writer.py](src/gfs2calmet/writer.py))
2. Hydrometeor compression uses CALWRF's `|x| < 0.00049` threshold —
   not exact zero — so floating-point residuals still trigger the
   `-N.000` collapse rule.
3. Compression is skipped when only one ratio would be emitted (a
   CALWRF edge case our flag dependencies prevent, but defended
   against for byte-identical output).

Each record's FORTRAN FORMAT is quoted next to its writer function
in [writer.py](src/gfs2calmet/writer.py), and the iteration order
inside each frame (`J` outer, `I` inner; surface record then NZP
vertical records) matches CALWRF lines 1236–1310.

---

## Wiring into CALMET

Your existing CALMET.INP already runs in WRF-driven mode with
`IPROG=14`. Switching the driver from WRF to GFS only needs:

| Section | Field | Old (WRF) | New (GFS) |
|---|---|---|---|
| Subgroup (d) | `M3DDAT` | `calwrf_em.m3d` | path to the file `gfs2calmet` writes |
| Group 1 | `IBYR/IBMO/IBDY/IBHR` | WRF run start | first valid time in your 3D.DAT (UTC) |
| Group 1 | `IEYR/IEMO/IEDY/IEHR` | WRF run end | last valid time + step |

Everything else stays put:

- `IPROG = 14`, `ISTEPPGS = 3600` — initial-guess mode, hourly.
- `NM3D = 1` — assuming you keep one 3D.DAT per CALMET run.
- `NOOBS = 2`, `NSSTA = 0`, `NUSTA = 0`, `NPSTA = -1` — pure prognostic.
- `IRHPROG = 1`, `ITPROG = 2`, `ITWPROG = 2` — pull RH / T / lapse rate
  from 3D.DAT.
- `MCLOUD = 3` — derive cloud cover from RH at 850 mb (CALMET's job,
  not ours).
- `ILUOC3D = 16` — USGS water category. Matches our
  `header.default_landuse`.
- Grid block (`PMAP`, `IUTMZN`, `XORIGKM`, `YORIGKM`, `NX`, `NY`,
  `DGRIDKM`, `ZFACE`) — your CALMET grid is independent from the 3D.DAT
  grid; CALMET interpolates.

The `GEO.DAT` you already have remains the authoritative source for
terrain and land-use in CALMET. Our `default_elevation_m` /
`default_landuse` only fill the corresponding 3D.DAT columns for
record-format completeness; CALMET overrides them.

---

## Modifying the pipeline

### Change the driver grid (size / projection / resolution)

Edit `target_grid` in your YAML. Nothing else needs touching:

```yaml
target_grid:
  crs: "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=25 +lon_0=51 ..."
  x0_km: -200.0
  y0_km: -200.0
  dx_km: 2.0
  dy_km: 2.0
  nx: 200
  ny: 200
```

If you switch projection family, also update `header.maptxt`
(`LCC`/`MER`/`PS`/`UTM`) and `truelat1/2`, `rlatc/rlonc` for
LCC/PS.

### Add a new GFS field

1. Append a `GfsField(...)` entry to `PRESSURE_LEVEL_FIELDS` or
   `SURFACE_FIELDS` in [`gfs_fields.py`](src/gfs2calmet/gfs_fields.py).
2. Add a matching Herbie idx pattern to `HERBIE_IDX_PATTERNS` and
   the role to `ROLES`.
3. If the new field has a corresponding 3D.DAT slot, wire it into
   [`frames.py`](src/gfs2calmet/frames.py)'s `build_frames()`.
4. Add a unit test in [`test_gfs_fields.py`](tests/test_gfs_fields.py)
   and a converter test in [`test_frames.py`](tests/test_frames.py)
   if you're adding a transformation.

### Change pressure levels

Edit `gfs.pressure_levels` (descending order). The reader filters
GFS messages to that list; the writer emits one vertical record per
level per cell per time.

### Swap to ECMWF Open Data

Create `ecmwf_fields.py` modelled on `gfs_fields.py` — ECMWF uses:
- `z` (m²/s²) for geopotential — divide by 9.80665 (multiplier=0.10197).
- 3-hourly cadence — temporal interpolation needed before the frame
  builder.
- Pressure levels limited to 1000, 925, 850, 700, 600, 500 hPa in
  the open stream.

Everything below the field catalog (reader / regrid / frames /
writer) stays unchanged because the xarray Dataset structure is
identical.

### Run on a different domain (not Qatar)

1. Copy `config_qatar.yaml` → `config_<domain>.yaml`.
2. Pick a UTM zone (or LCC) that minimizes distortion across your
   domain.
3. Compute SW corner in projected km. Easiest: `pyproj` round-trip
   from lat/lon corner.
4. Update `target_grid`, `header.maptxt`, comments.
5. Update CALMET.INP's `IUTMZN`, `XORIGKM`, `YORIGKM`, `NX`, `NY`,
   `DGRIDKM` to whatever CALMET grid you want (independent of 3D.DAT
   grid).

---

## Testing

```bash
pytest -q              # 128 tests, ~1 second
pytest -v              # verbose
pytest tests/test_writer_3ddat.py    # single module
pytest -k "compression"              # by keyword
```

No test reaches the network. `pygrib` and `Herbie` are mocked at the
module boundary, so the test environment doesn't need either
installed.

CI-equivalent run:

```bash
pytest -q && echo OK
```

---

## Roadmap / known gaps

In rough priority order. None block a first useful run; each is an
incremental improvement.

1. **Hourly APCP differencing** — GFS `tp` is accumulated over
   variable windows (1-h / 3-h / 6-h depending on fxx). Current
   writer takes the field as-is. Add a preprocessing pass in
   [`frames.py`](src/gfs2calmet/frames.py) that subtracts consecutive
   fxx accumulations and resets at bucket boundaries.
2. **GFS Omega → W conversion** — fetch `VVEL` on pressure levels,
   convert via `w ≈ -ω · R_d · T / (P · g)`, then flip `ioutw: 1`
   in the config.
3. **DEM terrain sampling** — populate `ielev_dot` from SRTM /
   GMTED2010 / GTOPO30 sampled onto the target grid. Cosmetic only —
   CALMET uses GEO.DAT.
4. **SST field** — pull GFS `tmpsfc` masked by land/sea and fill
   `sst` in surface records. `ITWPROG=2` in your CALMET.INP wants it.
5. **Snow cover** — pull `SNOWC` or `SNOD` for `sc` field.
6. **Integration test** — small canned GRIB2 fixture under
   `tests/data/` exercising decode → regrid → write end-to-end,
   `@pytest.mark.integration` so CI stays offline.
7. **CALMET.INP template generator** — emit a control file with
   matching grid + dates given the same YAML config. Closes the loop
   so the user never edits CALMET.INP by hand.
8. **Verification harness** — open the produced 3D.DAT in CALMET,
   parse `CALMET.LST` for `ERROR`/`WARNING`, compare CALMET-diagnosed
   winds at Doha/Hamad airport against GFS at the same point.

---

## References

- **Format spec**: *CALPUFF v6 User Instructions*, Section 7.7,
  Tables 7-32 and 7-33. Local copy parsed during development at
  `_calmet_v6_manual.txt` (not checked in).
- **Canonical 3D.DAT writer**: [CALWRF v2.0.3](CALWRF_v2.0.3_L190426/code/calwrf.f).
  All FORMAT statements quoted in our writer come from this source.
- **CALMET / CALPUFF official site**: <https://www.calpuff.org/>
- **GFS data**: NOMADS — <https://nomads.ncep.noaa.gov/>
- **Herbie**: <https://herbie.readthedocs.io>
- **pygrib**: <https://jswhit.github.io/pygrib/>
- **pyproj**: <https://pyproj4.github.io/pyproj/>
- **xarray**: <https://docs.xarray.dev/>

---

## Project conventions

- **Times** are UTC throughout the pipeline. CALMET's `ABTZ` handles
  the conversion to local time (Qatar: `UTC+0300`). Never mix.
- **Grids** are projected (km) inside the converter; pyproj does
  km↔m conversion at the CRS boundary.
- **Units** in xarray variables follow the documented `target_units`
  on each `GfsField`. `native_units` is kept as a `var.attrs` entry
  for traceability.
- **No silent defaults** in the config — missing keys raise.
- **Strict alignment with CALWRF** wherever the manual and source
  disagree. The source wins.
