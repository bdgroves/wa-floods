"""Build the image draped over the terrain for any hour of the floods.

Layers, bottom to top:

1. Land: a dark winter palette so the rivers can glow. Outside Washington it
   fades further toward charcoal.
2. Rivers: every HydroRIVERS reach draining 150+ km2, wider for bigger rivers,
   coloured by how its gauge's flow compares with the usual flow for that date:
   dim blue below normal, bright blue at normal, cyan to white as it climbs
   to 10x and 30x, and hot pink once a gauge passes its own record peak.
   Each reach takes the reading of the first gauge downstream of it.
   Rivers in flood also swell a pixel wider.
3. Rain is not in this image: it's rendered separately and laid over the
   render as a veil by finish.py (see rain_code).

Water is painted key green and swapped for blue after rendering (see finish.py).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform, transform_geom
from scipy import ndimage

import config

HRRR_LCC = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +R=6371229 +units=m +no_defs"
PST = dt.timezone(dt.timedelta(hours=config.UTC_OFFSET_H), "PST")

OCEAN_KEY = np.array([0, 210, 0], dtype=np.float32)
OCEAN_PREVIEW = np.array([14, 38, 76], dtype=np.float32)
OUTSIDE_DIM = 0.55
OUTSIDE_TO = np.array([30, 32, 38], dtype=np.float32)

# land: (elevation m, west-side colour, east-side colour) - a dark, wet December
LAND = [
    (0, (70, 92, 76), (110, 102, 84)),
    (700, (62, 86, 72), (98, 94, 80)),
    (1400, (92, 100, 100), (102, 102, 98)),
    (2000, (128, 134, 140), (128, 132, 138)),
    (3000, (186, 192, 200), (186, 192, 200)),
]

# rivers: (flow / usual flow, colour). Record crests get RECORD.
RIVER_RAMP = [(0.3, (40, 66, 112)), (1.0, (52, 120, 214)), (3.0, (84, 190, 252)),
              (10.0, (196, 238, 255)), (30.0, (255, 255, 255))]
RECORD = np.array([255, 84, 196], dtype=np.float32)
UNGAUGED = np.array([44, 78, 128], dtype=np.float32)
RIVER_OUTSIDE = 0.45          # how much of a river's colour shows outside Washington

# rain: (mm per hour, colour); opacity = 1 - exp(-mm / RAIN_SCALE), capped
RAIN_RAMP = [(0, (150, 168, 196)), (4, (176, 194, 222)), (15, (214, 226, 242))]
RAIN_SCALE = 3.0
RAIN_OPACITY_MAX = 0.42
RAIN_CODE_MAX = 40.0          # mm/h at the top of the rain pass's colour code (log scale)


def ramp(v: np.ndarray, stops, log: bool = False) -> np.ndarray:
    xs = np.array([s for s, _ in stops], dtype=np.float32)
    cs = np.array([c for _, c in stops], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if log:
        xs, v = np.log10(xs), np.log10(np.clip(v, 1e-6, None))
    v = np.clip(v, xs[0], xs[-1])
    return np.stack([np.interp(v, xs, cs[:, k]) for k in range(3)], axis=-1)


def river_color(ratio: np.ndarray) -> np.ndarray:
    return ramp(ratio, RIVER_RAMP, log=True)


def rain_alpha(mm: np.ndarray) -> np.ndarray:
    return np.clip(1 - np.exp(-np.asarray(mm, dtype=np.float32) / RAIN_SCALE), 0, RAIN_OPACITY_MAX)


def rain_color(mm: np.ndarray) -> np.ndarray:
    return ramp(mm, RAIN_RAMP)


def utc(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


class Drape:
    def __init__(self, water_key: bool = True):
        self.water = OCEAN_KEY if water_key else OCEAN_PREVIEW
        with rasterio.open(config.DEM) as src:
            self.dem = src.read(1).astype(np.float32)
            self.tf, self.crs = src.transform, src.crs
        h, w = self.dem.shape
        self.shape = (h, w)
        cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        xs = (self.tf.c + cols * self.tf.a).ravel()
        ys = (self.tf.f + rows * self.tf.e).ravel()
        lons, lats = transform(self.crs, "EPSG:4326", xs, ys)
        self.lons, self.lats = np.asarray(lons).reshape(h, w), np.asarray(lats).reshape(h, w)
        self.ocean = self.dem <= 0.5
        self.inside, self.border = self._washington()
        self.base = self._land()

        r = np.load(config.RAIN)
        self.glat, self.glon, self.rain = r["lat"], r["lon"], r["rain"]
        self.gi, self.gj = self._grid_index()
        g = np.load(config.GAUGES)
        self.g = {k: g[k] for k in g.files}
        self.times = [utc(s) for s in self.g["times_utc"]]
        self.t0 = self.times[0]
        rain_t0 = utc(str(r["times_utc"][0]))
        self.rain_offset = (rain_t0 - self.t0).total_seconds() / 3600
        self._rivers()

    # ------------------------------------------------------------ geometry

    def _grid_index(self):
        """Fractional (row, col) into the HRRR grid for every DEM pixel."""
        gx, gy = transform("EPSG:4326", HRRR_LCC, self.glon.ravel().tolist(), self.glat.ravel().tolist())
        gx, gy = np.asarray(gx).reshape(self.glat.shape), np.asarray(gy).reshape(self.glat.shape)
        x0, dx = gx[0, 0], np.mean(np.diff(gx, axis=1))
        y0, dy = gy[0, 0], np.mean(np.diff(gy, axis=0))
        px, py = transform("EPSG:4326", HRRR_LCC, self.lons.ravel().tolist(), self.lats.ravel().tolist())
        px, py = np.asarray(px).reshape(self.shape), np.asarray(py).reshape(self.shape)
        return (py - y0) / dy, (px - x0) / dx

    def _washington(self):
        gj = json.loads(config.BOUNDARY.read_text())
        geoms = [transform_geom("EPSG:4326", self.crs, f["geometry"]) for f in gj["features"]]
        inside = features.rasterize(geoms, out_shape=self.shape, transform=self.tf).astype(bool)
        edge = inside ^ ndimage.binary_erosion(inside, iterations=1)
        edge = ndimage.binary_dilation(edge, iterations=1) & ~self.ocean
        return inside, edge

    def _land(self) -> np.ndarray:
        z = np.where(self.ocean, 0, self.dem)
        zs = np.array([s for s, _, _ in LAND], dtype=np.float32)
        west = np.stack([np.interp(z, zs, [c[k] for _, c, _ in LAND]) for k in range(3)], axis=-1)
        east = np.stack([np.interp(z, zs, [c[k] for _, _, c in LAND]) for k in range(3)], axis=-1)
        e = 1 / (1 + np.exp(-(self.lons + 121.05) / 0.25))
        rgb = west * (1 - e[..., None]) + east * e[..., None]
        outside = ~self.inside & ~self.ocean
        rgb[outside] = rgb[outside] * (1 - OUTSIDE_DIM) + OUTSIDE_TO * OUTSIDE_DIM
        return rgb.astype(np.float32)

    # ------------------------------------------------------------ rivers

    def _rivers(self):
        import geopandas as gpd
        from shapely.geometry import Point

        rv = gpd.read_file(config.RIVERS)
        if rv.crs is None or rv.crs.to_epsg() != config.EPSG:
            rv = rv.to_crs(epsg=config.EPSG)
        rv = rv.sort_values("UPLAND_SKM").reset_index(drop=True)   # big rivers drawn last, on top
        n = len(rv)
        self.n_reach = n
        up = rv["UPLAND_SKM"].to_numpy(dtype=np.float64)
        # width in pixels (250 m): ~1.6 px for small rivers, ~6 for the Columbia
        width = np.clip(1.6 + 1.1 * np.log10(up / config.RIVER_MIN_UPLAND_KM2), 1.6, 6.5)
        px = abs(self.tf.a)

        def burn(extra_px: float) -> np.ndarray:
            shapes = ((geom.buffer((wd + extra_px) * px / 2, cap_style=2), k + 1)
                      for k, (geom, wd) in enumerate(zip(rv.geometry, width)))
            return features.rasterize(shapes, out_shape=self.shape, transform=self.tf,
                                      fill=0, dtype="int32") - 1      # -1 = no river

        self.reach_map = burn(0.0)
        self.reach_map_wide = burn(1.6)          # the extra ring lights up when a river floods
        self.reach_map_rec = burn(4.5)           # rivers past their record get a broad pink band
        for m in (self.reach_map, self.reach_map_wide, self.reach_map_rec):
            m[self.ocean] = -1

        # gauges -> reaches: the closest reach whose drainage area best matches the gauge's
        gx, gy = transform("EPSG:4326", self.crs, self.g["lon"].tolist(), self.g["lat"].tolist())
        sindex = rv.sindex
        gauge_reach = np.full(len(gx), -1)
        for i, (x, y) in enumerate(zip(gx, gy)):
            pt = Point(x, y)
            cand = list(sindex.query(pt.buffer(config.GAUGE_SNAP_M)))
            area = float(self.g["drain_km2"][i])
            best, score = -1, np.inf
            for c in cand:
                d = rv.geometry.iloc[c].distance(pt)
                if d > config.GAUGE_SNAP_M:
                    continue
                s = d / config.GAUGE_SNAP_M
                if np.isfinite(area) and area > 0:
                    s += 2.0 * abs(np.log10(area / up[c]))
                if s < score:
                    best, score = c, s
            # a gauge on a creek far smaller than any nearby mapped river shouldn't colour that river
            if best >= 0 and np.isfinite(area) and area > 0 and abs(np.log10(area / up[best])) > 0.7:
                best = -1
            gauge_reach[i] = best
        self.gauge_reach = gauge_reach

        # each reach takes the first gauge downstream of it (or on it)
        ids = rv["HYRIV_ID"].to_numpy()
        pos = {int(hid): k for k, hid in enumerate(ids)}
        nxt = np.array([pos.get(int(d), -1) for d in rv["NEXT_DOWN"].to_numpy()])
        on_reach = {}
        for i, r in enumerate(gauge_reach):
            if r >= 0 and (r not in on_reach or self.g["drain_km2"][i] > self.g["drain_km2"][on_reach[r]]):
                on_reach[r] = i
        reach_gauge = np.full(n, -1)
        for k in range(n):
            j, steps = k, 0
            while j >= 0 and steps < 400:
                if j in on_reach:
                    reach_gauge[k] = on_reach[j]
                    break
                j, steps = nxt[j], steps + 1
        self.reach_gauge = reach_gauge
        print(f"  {n:,} river reaches; {int((gauge_reach >= 0).sum())} of {len(gauge_reach)} gauges matched; "
              f"{int((reach_gauge >= 0).sum()):,} reaches take a gauge's reading")

        # flow relative to usual, per gauge per hour
        local_days = [t.astimezone(PST).day for t in self.times]
        p50 = self.g["p50"][:, local_days].T                       # (hours, gauges)
        flow = self.g["flow"]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.ratio = (flow / np.where(p50 > 0, p50, np.nan)).astype(np.float32)
        rec = self.g["record"]
        ok = np.isfinite(rec) & (self.g["record_n"] >= config.RECORD_MIN_YEARS) & (gauge_reach >= 0)
        self.is_record = flow > np.where(ok, rec, np.inf)[None, :]  # (hours, gauges)

    # ------------------------------------------------------------ per frame

    def hours(self) -> float:
        return (self.times[-1] - self.t0).total_seconds() / 3600

    def gauge_ratio(self, hour: float) -> np.ndarray:
        arr = self.ratio
        k0 = int(np.clip(np.floor(hour), 0, len(arr) - 1))
        k1 = min(k0 + 1, len(arr) - 1)
        f = float(np.clip(hour - k0, 0, 1))
        a0, a1 = arr[k0], arr[k1]
        out = a0 * (1 - f) + a1 * f
        return np.where(np.isnan(out), np.where(np.isnan(a0), a1, a0), out)

    def gauge_record(self, hour: float) -> np.ndarray:
        """True for gauges that have passed their record at or before this hour (they stay lit)."""
        k = int(np.clip(np.floor(hour), 0, len(self.is_record) - 1))
        return self.is_record[: k + 1].any(axis=0)

    def field(self, hour: float) -> np.ndarray:
        """Rain (mm in the hour) at every DEM pixel."""
        k = hour - self.rain_offset
        k0 = int(np.clip(np.floor(k), 0, len(self.rain) - 1))
        k1 = min(k0 + 1, len(self.rain) - 1)
        f = float(np.clip(k - k0, 0, 1))
        g = self.rain[k0].astype(np.float32) * (1 - f) + self.rain[k1].astype(np.float32) * f
        return np.clip(ndimage.map_coordinates(g, [self.gi, self.gj], order=1, mode="nearest"), 0, None)

    def image(self, hour: float, rain: bool = True) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.base.copy()
        mm = self.field(hour)
        if rain:                          # preview only; the final pass lays rain over as a veil
            a = rain_alpha(mm)[..., None]
            rgb = rgb * (1 - a) + rain_color(mm) * a

        # rivers
        gr = self.gauge_ratio(hour)
        recd = self.gauge_record(hour)
        rg = self.reach_gauge
        reach_ratio = np.where(rg >= 0, gr[np.maximum(rg, 0)], np.nan)
        reach_rec = np.where(rg >= 0, recd[np.maximum(rg, 0)], False)
        col = river_color(np.nan_to_num(reach_ratio, nan=1.0))
        col[np.isnan(reach_ratio)] = UNGAUGED
        col[reach_rec] = RECORD
        # the wider ring only for rivers running 3x usual or more, fading in from 3x to 10x
        with np.errstate(divide="ignore", invalid="ignore"):
            swell = np.clip(np.log10(np.nan_to_num(reach_ratio, nan=0) / 3.0) / np.log10(10 / 3.0), 0, 1)
        swell = np.nan_to_num(swell, nan=0.0)
        swell[reach_rec] = 1.0
        ring = (self.reach_map_wide >= 0) & (self.reach_map < 0)
        rr = self.reach_map_wide[ring]
        k = swell[rr][:, None]
        rgb[ring] = rgb[ring] * (1 - k) + col[rr] * k
        band = self.reach_map_rec >= 0
        band[band] = reach_rec[self.reach_map_rec[band]]
        rgb[band] = RECORD
        on = self.reach_map >= 0
        rgb[on] = col[self.reach_map[on]]
        # Washington is the story: rivers beyond the state line are drawn dimmer (and don't glow)
        beyond = (self.reach_map_wide >= 0) & ~self.inside
        rgb[beyond] = rgb[beyond] * RIVER_OUTSIDE + self.base[beyond] * (1 - RIVER_OUTSIDE)

        rgb[self.ocean] = self.water
        line = self.border & ~self.ocean
        rgb[line] = rgb[line] * 0.4 + 200 * 0.6
        return np.clip(rgb, 0, 255).astype(np.uint8), mm

    @staticmethod
    def rain_code(mm: np.ndarray) -> np.ndarray:
        """Rain encoded as colour for a second, rain-only render: the ratio red / (red + blue)
        survives the viewer's lighting, and finish.py reads the rain back from it (log scale)."""
        v = np.clip(np.log1p(mm) / np.log1p(RAIN_CODE_MAX), 0, 1)
        return np.stack([255 * v, np.zeros_like(v), 255 * (1 - v)], axis=-1).astype(np.uint8)

    def time_label(self, hour: float) -> str:
        """Local (Pacific standard) time for the clock, ISO format."""
        return (self.t0 + dt.timedelta(hours=hour)).astimezone(PST).strftime("%Y-%m-%dT%H:%M")

    def gauge_labels(self) -> list[dict]:
        """Gauges that beat their record: where, when (first hour over), and the numbers."""
        out, seen = [], {}
        for i in np.flatnonzero(self.is_record.any(axis=0)):
            if not self.g["drain_km2"][i] >= config.RIVER_MIN_UPLAND_KM2:
                continue                  # a creek too small to be drawn: still pink, no label
            first = int(np.argmax(self.is_record[:, i]))
            peak = float(np.nanmax(self.g["flow"][:, i]))
            out.append({"river": str(self.g["river"][i]), "name": str(self.g["name"][i]),
                        "lat": float(self.g["lat"][i]), "lon": float(self.g["lon"][i]),
                        "hour": float(first), "peak_cfs": peak,
                        "record_cfs": float(self.g["record"][i]), "record_year": str(self.g["record_year"][i]),
                        "drain_km2": float(self.g["drain_km2"][i])})
        # one label per river: the gauge draining the most land (furthest downstream)
        for lab in out:
            r = lab["river"]
            if r not in seen or lab["drain_km2"] > seen[r]["drain_km2"]:
                seen[r] = lab
        return sorted(seen.values(), key=lambda l: -l["peak_cfs"])
