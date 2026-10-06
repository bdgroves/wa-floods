"""Shared settings for the December 2025 Washington flood animation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
FONTS = ROOT / "assets" / "fonts"

# Washington plus a margin: Portland to the south, the Canadian border to the north.
WEST, EAST = -125.0, -116.6
SOUTH, NORTH = 45.45, 49.02
EPSG = 26910                  # NAD83 / UTM 10N, metres
DEM_PX = 250                  # metres
EXAGGERATION = 3.5
SMOOTH_PX = 1.3

# The floods: a strong atmospheric river Dec 8-11 and a second storm Dec 15-18.
# Dates are UTC days for the downloads; the clock on screen is Pacific time.
START = "2025-12-01"
END = "2025-12-20"
TIMEZONE = "America/Los_Angeles"
UTC_OFFSET_H = -8             # all of the window is Pacific standard time

# Rain colour scale, inches per hour (HRRR's hourly accumulation).
RAIN_TICKS = [(0.05, "0.05"), (0.1, "0.1"), (0.25, "0.25"), (0.5, "0.5+")]
RAIN_MAX_IN = 0.5

# Rivers: HydroRIVERS reaches draining at least this much land are drawn.
RIVER_MIN_UPLAND_KM2 = 150
# A gauge is matched to a river reach within this distance of it.
GAUGE_SNAP_M = 3000
# Flow is coloured by how it compares to the gauge's usual flow for that date
# (USGS daily median, many years of record): ratio = flow / median.
FLOW_TICKS = [(0.5, "½×"), (1, "normal"), (3, "3×"), (10, "10×"), (30, "30×")]
FLOW_RATIO_MAX = 30.0
# Gauges whose crest beats their own record (USGS annual peaks) get a label.
RECORD_MIN_YEARS = 20         # ...if the record is at least this long

DEM = DATA / "wa_dem.tif"
BOUNDARY = DATA / "washington.geojson"
RAIN = DATA / f"rain_hrrr_{START}_{END}.npz"
RIVERS = DATA / "rivers_hydrorivers_wa.gpkg"
GAUGES = DATA / f"gauges_usgs_{START}_{END}.npz"
GAUGE_INFO = DATA / "gauges_usgs_sites.json"

# Reuse the terrain and outline from the projects next door if they're there.
SIBLING_DATA = ROOT.parent / "wa-heat-dome" / "data"


@dataclass
class City:
    name: str
    lat: float
    lon: float
    priority: int = 50


CITIES = [
    City("Seattle", 47.6062, -122.3321, 100),
    City("Mount Vernon", 48.4212, -122.3341, 95),
    City("Everett", 47.9790, -122.2021, 80),
    City("Bellingham", 48.7519, -122.4787, 75),
    City("Spokane", 47.6588, -117.4260, 70),
    City("Yakima", 46.6021, -120.5059, 65),
    City("Wenatchee", 47.4235, -120.3103, 60),
]


# Output shapes, same cameras as the heat-dome project.
FORMATS = {
    "16x9": {"size": (1920, 1080), "theta": 45.0, "fov": 38.0, "south_m": 30_000},
    "1x1": {"size": (1080, 1080), "theta": 45.0, "fov": 55.0, "south_m": 60_000},
    "4x5": {"size": (1080, 1350), "theta": 40.0, "fov": 70.0, "south_m": 0},
}


def out_dir(fmt: str) -> Path:
    return OUT if fmt == "16x9" else OUT / fmt


def video_path(fmt: str) -> Path:
    return OUT / ("wa_floods_2025.mp4" if fmt == "16x9" else f"wa_floods_2025_{fmt}.mp4")
