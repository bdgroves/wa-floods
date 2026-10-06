"""Download the river network: HydroRIVERS v1.0, North America.

HydroRIVERS (WWF / McGill, free for any use) is a global river network
derived from 15-arc-second terrain. Every reach knows the reach downstream of
it (NEXT_DOWN), the land it drains (UPLAND_SKM) and its long-term average flow
(DIS_AV_CMS), which is what's needed to draw rivers wider as they grow and to
hand each reach the reading of the nearest gauge.

The North America file is a 66 MB zip, downloaded once into data/cache/.

    pixi run rivers
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import zipfile

import config

URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_na_shp.zip"
KEEP = ["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", "LENGTH_KM", "UPLAND_SKM", "DIS_AV_CMS", "ORD_STRA", "ORD_CLAS"]
MARGIN = 0.5


HEADERS = {
    # data.hydrosheds.org turns away Python's default downloader (HTTP 403).
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
    "Accept": "application/zip,application/octet-stream,*/*",
    "Referer": "https://www.hydrosheds.org/products/hydrorivers",
}


def download(dest, tries: int = 4) -> bool:
    tmp = dest.with_suffix(".part")
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"  {got / 1e6:5.1f} / {total / 1e6:.1f} MB", end="\r")
            print()
            if not zipfile.is_zipfile(tmp):
                raise ValueError("not a zip (the server sent a web page)")
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            print(f"  attempt {k + 1}: HTTP {e.code}")
            if e.code in (401, 403, 404):
                break
        except Exception as e:                       # noqa: BLE001 - network hiccups, retry
            print(f"  attempt {k + 1}: {e}")
        time.sleep(3 * (k + 1))
    tmp.unlink(missing_ok=True)
    return False


def main() -> int:
    if config.RIVERS.exists():
        print(f"Rivers already present: {config.RIVERS.name}")
        return 0
    import geopandas as gpd

    cache = config.DATA / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "HydroRIVERS_v10_na_shp.zip"
    if not zpath.exists():
        print("Downloading HydroRIVERS North America (66 MB)...")
        if not download(zpath):
            print("\nThe HydroSHEDS server refused the download. Grab it in a browser instead:\n"
                  f"  1. Open {URL}\n"
                  f"  2. Save the zip as {zpath}\n"
                  "  3. Run this step again (it uses the saved zip).")
            return 1
    with zipfile.ZipFile(zpath) as z:
        shp = next(n for n in z.namelist() if n.lower().endswith(".shp"))
    bbox = (config.WEST - MARGIN, config.SOUTH - MARGIN, config.EAST + MARGIN, config.NORTH + MARGIN)
    print("Reading the Washington part...")
    gdf = gpd.read_file(f"zip://{zpath.as_posix()}!{shp}", bbox=bbox, engine="pyogrio")
    gdf = gdf[[c for c in KEEP if c in gdf.columns] + ["geometry"]]
    gdf = gdf[gdf["UPLAND_SKM"] >= config.RIVER_MIN_UPLAND_KM2]
    gdf = gdf.to_crs(epsg=config.EPSG)
    gdf.to_file(config.RIVERS, driver="GPKG")
    big = gdf.sort_values("DIS_AV_CMS", ascending=False).head(3)
    print(f"Saved {len(gdf):,} river reaches draining {config.RIVER_MIN_UPLAND_KM2}+ km2 "
          f"(largest average flow {big['DIS_AV_CMS'].iloc[0]:,.0f} m3/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
