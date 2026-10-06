"""Download river flow at every USGS stream gauge in and around Washington.

For December 1-20, 2025:

  * Flow (discharge, cubic feet per second) every 15 minutes, averaged to hours.
  * Each gauge's usual flow for each December day: the USGS daily median over
    its whole record. Flow / median is what colours the rivers.
  * Each gauge's annual peaks, to tell when a December crest beat the record.
  * Gauge name, location and drainage area, to match it to a river reach.

All from the USGS Water Services (no key needed). Responses are cached in
data/cache/usgs/, so a rerun doesn't download again.

    pixi run gauges
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config

WS = "https://waterservices.usgs.gov/nwis"
PEAKS = "https://nwis.waterdata.usgs.gov/nwis/peak"
CACHE = config.DATA / "cache" / "usgs"
NO_DATA = -999999


def get(url: str, cache_name: str | None = None, tries: int = 4) -> str:
    if cache_name:
        f = CACHE / cache_name
        if f.exists():
            return f.read_text(encoding="utf-8")
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wa-floods (brooksgroves.com)",
                                                        "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=300) as r:
                text = r.read().decode("utf-8", "replace")
            if cache_name:
                (CACHE / cache_name).write_text(text, encoding="utf-8")
            return text
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return ""
            err = e
        except Exception as e:                       # noqa: BLE001 - network hiccups, retry
            err = e
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url.split('?')[0]}: {err}")


def rdb_rows(text: str) -> list[dict]:
    """Rows of a USGS tab-delimited RDB response."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[2:]]


def utc_hours() -> list[dt.datetime]:
    t = dt.datetime.fromisoformat(config.START).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(config.END).replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=23)
    out = []
    while t <= end:
        out.append(t)
        t += dt.timedelta(hours=1)
    return out


def instantaneous() -> dict:
    """15-minute discharge for every gauge in the box, in strips (USGS caps the box size)."""
    start = (dt.date.fromisoformat(config.START) - dt.timedelta(days=1)).isoformat()
    end = (dt.date.fromisoformat(config.END) + dt.timedelta(days=1)).isoformat()
    sites = {}
    edges = np.linspace(config.WEST, config.EAST, 4)
    for k in range(3):
        bbox = f"{edges[k]:.4f},{config.SOUTH:.4f},{edges[k + 1]:.4f},{config.NORTH:.4f}"
        q = {"format": "json", "bBox": bbox, "parameterCd": "00060", "startDT": start, "endDT": end,
             "siteStatus": "all", "siteType": "ST"}
        print(f"  flow, strip {k + 1}/3...")
        text = get(f"{WS}/iv/?{urllib.parse.urlencode(q)}", f"iv_{k}_{start}_{end}.json")
        if not text:
            continue
        for ts in json.loads(text)["value"]["timeSeries"]:
            info = ts["sourceInfo"]
            site = info["siteCode"][0]["value"]
            vals = []
            for block in ts["values"]:
                for v in block["value"]:
                    try:
                        x = float(v["value"])
                    except (TypeError, ValueError):
                        continue
                    if x <= NO_DATA + 1 or x < 0:
                        continue
                    vals.append((dt.datetime.fromisoformat(v["dateTime"]).astimezone(dt.timezone.utc), x))
            if len(vals) < 24:
                continue
            geo = info["geoLocation"]["geogLocation"]
            if site not in sites or len(vals) > len(sites[site]["vals"]):
                sites[site] = {"name": info["siteName"], "lat": float(geo["latitude"]),
                               "lon": float(geo["longitude"]), "vals": vals}
    return sites


def hourly(vals, hours) -> np.ndarray:
    """Average 15-minute readings into the UTC hours; fill gaps of up to 6 hours."""
    t0 = hours[0]
    sums, counts = np.zeros(len(hours)), np.zeros(len(hours))
    for t, x in vals:
        k = int((t - t0).total_seconds() // 3600)
        if 0 <= k < len(hours):
            sums[k] += x
            counts[k] += 1
    out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    have = np.flatnonzero(~np.isnan(out))
    if have.size >= 2:
        idx = np.arange(len(out))
        filled = np.interp(idx, have, out[have])
        gap = np.array([min(abs(i - have).min(), 99) for i in idx])
        out = np.where(gap <= 6, filled, np.nan)
    return out


def site_details(site_nos: list[str]) -> dict:
    info = {}
    for k in range(0, len(site_nos), 100):
        batch = site_nos[k:k + 100]
        q = {"format": "rdb", "sites": ",".join(batch), "siteOutput": "expanded", "siteStatus": "all"}
        for r in rdb_rows(get(f"{WS}/site/?{urllib.parse.urlencode(q)}", f"site_{k}.rdb")):
            try:
                area = float(r.get("drain_area_va") or "nan") * 2.58999
            except ValueError:
                area = float("nan")
            info[r["site_no"]] = {"drain_km2": area}
    return info


def december_medians(site_nos: list[str]) -> dict:
    """USGS daily-median flow for each December day, per gauge."""
    med = {}
    for k in range(0, len(site_nos), 10):
        batch = site_nos[k:k + 10]
        q = {"format": "rdb", "sites": ",".join(batch), "statReportType": "daily",
             "statTypeCd": "p50", "parameterCd": "00060"}
        for r in rdb_rows(get(f"{WS}/stat/?{urllib.parse.urlencode(q)}", f"stat_{'_'.join(batch)}.rdb")):
            if r.get("month_nu") in ("11", "12"):
                try:
                    v = float(r["p50_va"])
                except (KeyError, ValueError):
                    continue
                med.setdefault(r["site_no"], {})[(int(r["month_nu"]), int(r["day_nu"]))] = v
        print(f"  medians {min(k + 10, len(site_nos))}/{len(site_nos)}", end="\r")
    print()
    return med


def record_peak(site: str):
    """(highest annual peak before this water year, its year, years of record)."""
    q = {"site_no": site, "agency_cd": "USGS", "format": "rdb"}
    best, year, n = None, None, 0
    for r in rdb_rows(get(f"{PEAKS}?{urllib.parse.urlencode(q)}", f"peak_{site}.rdb")):
        d, v = r.get("peak_dt", ""), r.get("peak_va", "")
        if not d or not v or d >= "2025-10-01":
            continue
        try:
            v = float(v)
        except ValueError:
            continue
        n += 1
        if best is None or v > best:
            best, year = v, d[:4]
    return best, year, n


def river_name(site_name: str) -> str:
    """'SKAGIT RIVER NEAR MOUNT VERNON, WA' -> 'Skagit River'."""
    s = re.split(r"\s+(?:NEAR|NR|AT|ABOVE|ABV|BELOW|BLW|BL|AB|@|DS|US)\s+", site_name.upper())[0]
    s = s.split(",")[0].strip()
    s = re.sub(r"^(N F|NF|N\.F\.|NO FK|NORTH FORK)\s+", "NORTH FORK ", s)
    s = re.sub(r"^(S F|SF|S\.F\.|SO FK|SOUTH FORK)\s+", "SOUTH FORK ", s)
    s = re.sub(r"^(M F|MF|MIDDLE FORK)\s+", "MIDDLE FORK ", s)
    s = s.replace(" R ", " RIVER ").replace(" CR ", " CREEK ")
    s = re.sub(r"\bR$", "RIVER", s)
    s = re.sub(r"\bCR$", "CREEK", s)
    return " ".join(w.capitalize() if not w.startswith("(") else w for w in s.split())


def main() -> int:
    if config.GAUGES.exists():
        print(f"Gauges already present: {config.GAUGES.name}")
        return 0
    CACHE.mkdir(parents=True, exist_ok=True)
    print("USGS stream gauges, December 2025...")
    sites = instantaneous()
    hours = utc_hours()
    flows, keep = [], []
    for s, d in sites.items():
        h = hourly(d["vals"], hours)
        if np.isfinite(h).sum() >= len(hours) * 0.6:
            flows.append(h)
            keep.append(s)
    print(f"  {len(keep)} gauges with flow for most of the window")
    details = site_details(keep)
    meds = december_medians(keep)
    print("  annual peaks...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        peaks = dict(zip(keep, pool.map(record_peak, keep)))

    n = len(keep)
    p50 = np.full((n, 32), np.nan, dtype=np.float32)       # by December day of month
    for i, s in enumerate(keep):
        for (m, day), v in meds.get(s, {}).items():
            if m == 12:
                p50[i, day] = v
        # a gauge with no median for some days: borrow the nearest day's
        row = p50[i, 1:]
        if np.isfinite(row).any():
            idx = np.arange(row.size)
            ok = np.isfinite(row)
            p50[i, 1:] = np.interp(idx, idx[ok], row[ok])
    rec = np.array([peaks[s][0] if peaks[s][0] else np.nan for s in keep], dtype=np.float32)
    rec_year = np.array([peaks[s][1] or "" for s in keep])
    rec_n = np.array([peaks[s][2] for s in keep], dtype=np.int32)
    np.savez_compressed(
        config.GAUGES, site=np.array(keep), name=np.array([sites[s]["name"] for s in keep]),
        river=np.array([river_name(sites[s]["name"]) for s in keep]),
        lat=np.array([sites[s]["lat"] for s in keep]), lon=np.array([sites[s]["lon"] for s in keep]),
        drain_km2=np.array([details.get(s, {}).get("drain_km2", np.nan) for s in keep], dtype=np.float32),
        times_utc=np.array([f"{h:%Y-%m-%dT%H:00}" for h in hours]),
        flow=np.array(flows, dtype=np.float32).T, p50=p50, record=rec, record_year=rec_year, record_n=rec_n)

    # what broke records (the story's headline numbers)
    peak = np.nanmax(np.array(flows), axis=1)
    beat = [(keep[i], sites[keep[i]]["name"], peak[i], rec[i], rec_year[i], rec_n[i]) for i in range(n)
            if np.isfinite(rec[i]) and rec_n[i] >= config.RECORD_MIN_YEARS and peak[i] > rec[i]]
    print(f"Saved {config.GAUGES.name}: {n} gauges x {len(hours)} hours; "
          f"{len(beat)} topped their record (hourly means, so true crests ran a little higher):")
    for s, name, p, r, y, k in sorted(beat, key=lambda b: -b[2])[:15]:
        print(f"  {name[:46]:<46} {p:>9,.0f} cfs  (record {r:,.0f} in {y}, {k} yrs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
