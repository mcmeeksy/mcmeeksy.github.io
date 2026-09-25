""" 
sentinel2_fetcher.py
────────────────────
Pulls Sentinel-2 L2A imagery for a given area-of-interest and date range,
saves patches as .npy files ready for train_litter_temporal.py.

Supports three free API backends — pick whichever you can access:

  1. Google Earth Engine (GEE)          ← easiest, most reliable
  2. Microsoft Planetary Computer STAC  ← no account needed, great for Chicago
  3. Copernicus Data Space (CDSE) STAC  ← official ESA, free account required

────────────────────
QUICK START
────────────────────
  pip install earthengine-api pystac-client planetary-computer \
              odc-stac rasterio numpy tqdm requests

  # For GEE backend only:
  earthengine authenticate

  python sentinel2_fetcher.py

Output layout (matches what train_litter_temporal.py expects):
  data/temporal/patches/
    chicago_lake_01/
      2024-03-15.npy    ← shape (6, 128, 128) float32, values 0-1
      2024-04-02.npy
      ...
  data/temporal/labels/
    chicago_lake_01/
      2024-03-15.npy    ← shape (128, 128) int8  (all zeros — you label these)
"""

import os
import json
import time
import hashlib
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────────
# 0. CONFIG  ← edit this section
# ───────────────────────────────────────────────────────────────────────────
FETCH_CONFIG = {
    # ── Backend ────────────────────────────────────────────────────────────
    # Choose: "gee" | "planetary_computer" | "copernicus"
    "backend": "planetary_computer",

    # ── Area of Interest (Chicago-focused defaults) ────────────────────────
    # Each entry = one "location" folder in the output
    "locations": [
        {
            "id":     "chicago_lakefront",
            "bbox":   [-87.62, 41.85, -87.58, 41.89],   # [W, S, E, N]
            "label":  "Lake Michigan shoreline — Chicago",
        },
        {
            "id":     "chicago_river_north",
            "bbox":   [-87.65, 41.88, -87.62, 41.91],
            "label":  "Chicago River north branch",
        },
        {
            "id":     "calumet_river",
            "bbox":   [-87.58, 41.70, -87.54, 41.74],
            "label":  "Calumet River industrial zone",
        },
        {
            "id":     "lincoln_park_coast",
            "bbox":   [-87.64, 41.92, -87.61, 41.96],
            "label":  "Lincoln Park coastline",
        },
    ],

    # ── Date range ─────────────────────────────────────────────────────────
    "start_date":   "2022-01-01",
    "end_date":     "2024-12-31",

    # ── Sentinel-2 filters ─────────────────────────────────────────────────
    "max_cloud_pct":   20,        # skip scenes cloudier than this
    "bands":           ["B02", "B03", "B04", "B05", "B08", "B11"],
    # Indices:           0      1      2      3      4      5
    # Blue  Green  Red   RedEdge NIR   SWIR

    # ── Patch settings ─────────────────────────────────────────────────────
    "patch_size":      128,       # pixels (at 10m resolution = 1.28km)
    "target_res_m":    10,        # resample all bands to 10m

    # ── Output ─────────────────────────────────────────────────────────────
    "output_data_dir":  "./data/temporal/patches",
    "output_label_dir": "./data/temporal/labels",
    "cache_dir":        "./data/cache",            # raw downloads cached here
    "overwrite":        False,                     # skip if .npy already exists

    # ── GEE-specific (only needed if backend="gee") ────────────────────────
    "gee_project":      "your-gee-project-id",    # from earthengine authenticate

    # ── Copernicus-specific (only if backend="copernicus") ─────────────────
    "cdse_user":        "",   # register free at dataspace.copernicus.eu
    "cdse_password":    "",
}

BAND_SCALE = 10000.0   # S2 L2A reflectance is stored as DN/10000


# ───────────────────────────────────────────────────────────────────────────
# 1. SHARED UTILITIES
# ───────────────────────────────────────────────────────────────────────────
def bbox_to_geojson(bbox):
    """[W,S,E,N] → GeoJSON polygon dict."""
    W, S, E, N = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[W, S], [E, S], [E, N], [W, N], [W, S]]]
    }


def date_range_monthly(start, end):
    """Yield (start, end) pairs month by month."""
    cur = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while cur < end_dt:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur.strftime("%Y-%m-%d"), min(nxt, end_dt).strftime("%Y-%m-%d")
        cur = nxt


def normalise(arr):
    """Clip to valid S2 range and scale to [0, 1]."""
    arr = np.clip(arr, 0, BAND_SCALE) / BAND_SCALE
    return arr.astype(np.float32)


def save_patch(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def save_empty_label(shape_hw, path):
    """Save a zeroed label file. You fill these in with your annotation tool."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        np.save(path, np.zeros(shape_hw, dtype=np.int8))


def patch_id(loc_id, date_str):
    return f"{loc_id}/{date_str}"


# ───────────────────────────────────────────────────────────────────────────
# 2. BACKEND: MICROSOFT PLANETARY COMPUTER  (recommended — no account needed)
# ───────────────────────────────────────────────────────────────────────────
def fetch_planetary_computer(cfg):
    """
    Uses the Planetary Computer STAC API + odc-stac to load S2 L2A data.
    Free, no login required. Best option for getting started quickly.

    Install:
        pip install pystac-client planetary-computer odc-stac odc-geo rasterio
    """
    try:
        import planetary_computer
        import pystac_client
        import odc.stac
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        raise ImportError(
            "pip install pystac-client planetary-computer odc-stac odc-geo rasterio"
        )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    bands     = cfg["bands"]
    patch_sz  = cfg["patch_size"]
    out_data  = cfg["output_data_dir"]
    out_label = cfg["output_label_dir"]
    fetched   = 0

    for loc in cfg["locations"]:
        loc_id  = loc["id"]
        bbox    = loc["bbox"]
        print(f"\n📍 Location: {loc_id}  ({loc['label']})")

        search = catalog.search(
            collections  = ["sentinel-2-l2a"],
            bbox         = bbox,
            datetime     = f"{cfg['start_date']}/{cfg['end_date']}",
            query        = {"eo:cloud_cover": {"lt": cfg["max_cloud_pct"]}},
            sortby       = "datetime",
        )

        items = list(search.items())
        print(f"   Found {len(items)} scenes under {cfg['max_cloud_pct']}% cloud")

        for item in tqdm(items, desc=f"  {loc_id}", unit="scene"):
            date_str  = item.datetime.strftime("%Y-%m-%d")
            out_path  = os.path.join(out_data,  loc_id, f"{date_str}.npy")
            lbl_path  = os.path.join(out_label, loc_id, f"{date_str}.npy")

            if os.path.exists(out_path) and not cfg["overwrite"]:
                continue

            try:
                # Load bands via odc-stac — handles reprojection + resampling
                ds = odc.stac.load(
                    [item],
                    bands      = bands,
                    bbox       = bbox,
                    resolution = cfg["target_res_m"],
                    resampling = Resampling.bilinear,
                    groupby    = "solar_day",
                )
                # ds shape: (time, band, y, x)
                arr = ds.isel(time=0).to_array().values  # (C, H, W)

                if arr.shape[1] < 32 or arr.shape[2] < 32:
                    continue   # scene too small for this bbox

                # Centre-crop to patch_size
                _, H, W = arr.shape
                ch = (H - patch_sz) // 2
                cw = (W - patch_sz) // 2
                if ch < 0 or cw < 0:
                    arr = np.pad(arr, ((0,0),(max(0,-ch),max(0,-ch)),(max(0,-cw),max(0,-cw))))
                    ch, cw = 0, 0
                patch = arr[:, ch:ch+patch_sz, cw:cw+patch_sz]

                save_patch(normalise(patch), out_path)
                save_empty_label((patch_sz, patch_sz), lbl_path)
                fetched += 1

            except Exception as e:
                tqdm.write(f"    ⚠️  {date_str} failed: {e}")

    print(f"\n✅ Planetary Computer: saved {fetched} patches")
    return fetched


# ───────────────────────────────────────────────────────────────────────────
# 3. BACKEND: GOOGLE EARTH ENGINE
# ───────────────────────────────────────────────────────────────────────────
def fetch_gee(cfg):
    """
    Uses the GEE Python API to export median-composite monthly patches.

    Requires:
        pip install earthengine-api
        earthengine authenticate     ← one-time browser auth

    GEE gives you cloud-masked, atmospherically corrected S2 SR composites
    and is the most reliable for Chicago inland/coastal water bodies.
    """
    try:
        import ee
    except ImportError:
        raise ImportError("pip install earthengine-api  then  earthengine authenticate")

    ee.Initialize(project=cfg["gee_project"])
    print("✅ GEE initialised")

    bands     = cfg["bands"]
    patch_sz  = cfg["patch_size"]
    out_data  = cfg["output_data_dir"]
    out_label = cfg["output_label_dir"]
    fetched   = 0

    # Cloud masking function for S2 SR
    def mask_s2_clouds(image):
        qa = image.select("QA60")
        cloud_bit_mask    = 1 << 10
        cirrus_bit_mask   = 1 << 11
        mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
                  .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
        return image.updateMask(mask).divide(BAND_SCALE).copyProperties(image, ["system:time_start"])

    for loc in cfg["locations"]:
        loc_id = loc["id"]
        W, S, E, N = loc["bbox"]
        region = ee.Geometry.Rectangle([W, S, E, N])
        print(f"\n📍 Location: {loc_id}  ({loc['label']})")

        for month_start, month_end in date_range_monthly(cfg["start_date"], cfg["end_date"]):
            date_str  = month_start
            out_path  = os.path.join(out_data,  loc_id, f"{date_str}.npy")
            lbl_path  = os.path.join(out_label, loc_id, f"{date_str}.npy")

            if os.path.exists(out_path) and not cfg["overwrite"]:
                continue

            try:
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(region)
                    .filterDate(month_start, month_end)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["max_cloud_pct"]))
                    .map(mask_s2_clouds)
                    .select(bands)
                )

                if collection.size().getInfo() == 0:
                    continue

                # Median composite for the month
                composite = collection.median().clip(region)

                # Export as numpy via getDownloadURL (small areas only)
                url = composite.getDownloadURL({
                    "region":      region,
                    "dimensions":  f"{patch_sz}x{patch_sz}",
                    "format":      "NPY",
                    "bands":       bands,
                })

                resp = requests.get(url, timeout=120)
                resp.raise_for_status()

                # GEE NPY download returns a structured array
                arr_struct = np.load(__import__("io").BytesIO(resp.content), allow_pickle=True)
                arr = np.stack([arr_struct[b] for b in bands], axis=0).astype(np.float32)

                if arr.shape[1] < 32:
                    continue

                save_patch(np.clip(arr, 0, 1), out_path)
                save_empty_label((patch_sz, patch_sz), lbl_path)
                fetched += 1
                time.sleep(0.5)   # be polite to the API

            except Exception as e:
                print(f"    ⚠️  {date_str} failed: {e}")

    print(f"\n✅ GEE: saved {fetched} patches")
    return fetched


# ───────────────────────────────────────────────────────────────────────────
# 4. BACKEND: COPERNICUS DATA SPACE (ESA official)
# ───────────────────────────────────────────────────────────────────────────
def fetch_copernicus(cfg):
    """
    Uses the Copernicus Data Space Ecosystem (CDSE) STAC + OData APIs.
    Free account required: https://dataspace.copernicus.eu

    Install:
        pip install pystac-client requests rasterio numpy
    """
    try:
        import pystac_client
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling
    except ImportError:
        raise ImportError("pip install pystac-client rasterio")

    # ── Auth ──────────────────────────────────────────────────────────────
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    token_resp = requests.post(token_url, data={
        "grant_type": "password",
        "client_id":  "cdse-public",
        "username":   cfg["cdse_user"],
        "password":   cfg["cdse_password"],
    })
    if token_resp.status_code != 200:
        raise RuntimeError(
            "CDSE auth failed — check cdse_user/cdse_password in FETCH_CONFIG.\n"
            "Register free at https://dataspace.copernicus.eu"
        )
    access_token = token_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    print("✅ Copernicus auth OK")

    catalog = pystac_client.Client.open(
        "https://catalogue.dataspace.copernicus.eu/stac",
        headers=headers,
    )

    bands     = cfg["bands"]
    patch_sz  = cfg["patch_size"]
    out_data  = cfg["output_data_dir"]
    out_label = cfg["output_label_dir"]
    fetched   = 0

    for loc in cfg["locations"]:
        loc_id = loc["id"]
        bbox   = loc["bbox"]
        W, S, E, N = bbox
        print(f"\n📍 Location: {loc_id}  ({loc['label']})")

        search = catalog.search(
            collections = ["SENTINEL-2"],
            bbox        = bbox,
            datetime    = f"{cfg['start_date']}/{cfg['end_date']}",
            query       = {
                "eo:cloud_cover":    {"lt": cfg["max_cloud_pct"]},
                "s2:product_type":   {"eq": "S2MSI2A"},   # L2A only
            },
            max_items = 200,
        )

        items = list(search.items())
        print(f"   Found {len(items)} scenes")

        for item in tqdm(items, desc=f"  {loc_id}", unit="scene"):
            date_str = item.datetime.strftime("%Y-%m-%d")
            out_path = os.path.join(out_data,  loc_id, f"{date_str}.npy")
            lbl_path = os.path.join(out_label, loc_id, f"{date_str}.npy")

            if os.path.exists(out_path) and not cfg["overwrite"]:
                continue

            try:
                band_arrays = []
                for band in bands:
                    if band not in item.assets:
                        raise KeyError(f"Band {band} not in item assets")

                    href = item.assets[band].href
                    # CDSE requires auth header on asset downloads
                    with rasterio.open(href, opener=lambda url, **kw: requests.get(
                        url, headers=headers, stream=True
                    ).raw) as src:
                        window = from_bounds(W, S, E, N, transform=src.transform)
                        data   = src.read(
                            1, window=window,
                            out_shape=(patch_sz, patch_sz),
                            resampling=Resampling.bilinear,
                        )
                    band_arrays.append(data)

                arr = np.stack(band_arrays, axis=0).astype(np.float32)
                save_patch(normalise(arr), out_path)
                save_empty_label((patch_sz, patch_sz), lbl_path)
                fetched += 1

            except Exception as e:
                tqdm.write(f"    ⚠️  {date_str} failed: {e}")

    print(f"\n✅ Copernicus: saved {fetched} patches")
    return fetched


# ───────────────────────────────────────────────────────────────────────────
# 5. VERIFICATION & SUMMARY
# ───────────────────────────────────────────────────────────────────────────
def verify_dataset(cfg):
    """
    After fetching, scan the output directory and print a summary.
    Catches: missing bands, wrong shapes, NaN/Inf values.
    """
    out_data  = cfg["output_data_dir"]
    out_label = cfg["output_label_dir"]
    issues    = []
    summary   = {}

    print("\n" + "="*55)
    print("📋 DATASET VERIFICATION")
    print("="*55)

    for loc_dir in sorted(Path(out_data).iterdir()):
        if not loc_dir.is_dir():
            continue
        loc_id = loc_dir.name
        files  = sorted(loc_dir.glob("*.npy"))
        summary[loc_id] = {"scenes": len(files), "bad": 0}

        for f in files:
            arr = np.load(f)
            date_str = f.stem

            # Shape
            if arr.ndim != 3 or arr.shape[0] != len(cfg["bands"]):
                issues.append(f"{loc_id}/{date_str}: bad shape {arr.shape}")
                summary[loc_id]["bad"] += 1
                continue

            # Range
            if arr.min() < -0.1 or arr.max() > 1.5:
                issues.append(f"{loc_id}/{date_str}: values out of range [{arr.min():.2f}, {arr.max():.2f}]")

            # NaN / Inf
            if not np.isfinite(arr).all():
                issues.append(f"{loc_id}/{date_str}: contains NaN/Inf")
                summary[loc_id]["bad"] += 1

            # Label exists
            lbl_path = Path(out_label) / loc_id / f"{date_str}.npy"
            if not lbl_path.exists():
                issues.append(f"{loc_id}/{date_str}: missing label file")

    for loc_id, s in summary.items():
        status = "✅" if s["bad"] == 0 else "⚠️ "
        print(f"  {status} {loc_id:30s}  {s['scenes']:3d} scenes  {s['bad']} errors")

    if issues:
        print(f"\n⚠️  {len(issues)} issues found:")
        for iss in issues[:20]:
            print(f"    • {iss}")
    else:
        print("\n✅ All patches look clean!")

    total = sum(s["scenes"] for s in summary.values())
    print(f"\n  Total patches: {total}")
    print(f"  Locations:     {len(summary)}")
    print("="*55)

    # Save summary
    summary_path = os.path.join(cfg["output_data_dir"], "..", "fetch_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now().isoformat(),
            "backend":    cfg["backend"],
            "locations":  summary,
            "issues":     issues,
        }, f, indent=2)
    print(f"  Summary saved → {summary_path}")
    return issues


# ───────────────────────────────────────────────────────────────────────────
# 6. LABELLING HELPER (CLI)
# ───────────────────────────────────────────────────────────────────────────
def print_labelling_instructions(cfg):
    """Print instructions for labelling the downloaded patches."""
    print("""
╔══════════════════════════════════════════════════════════╗
║              NEXT STEP: LABEL YOUR PATCHES               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  The fetcher created zero-filled label files.            ║
║  You need to annotate litter pixels before training.     ║
║                                                          ║
║  RECOMMENDED TOOLS:                                      ║
║                                                          ║
║  1. Label Studio (free, browser-based)                   ║
║     pip install label-studio                             ║
║     label-studio start                                   ║
║     → Import your .npy patches as images                 ║
║     → Use polygon/brush tool to mark litter              ║
║     → Export as masks → convert to .npy                  ║
║                                                          ║
║  2. QGIS (if you prefer GIS tools)                       ║
║     → Open .npy as raster, digitize polygons,            ║
║       rasterize to match patch grid                      ║
║                                                          ║
║  3. Quick auto-label with FDI threshold (noisy but fast) ║
║     python sentinel2_fetcher.py --autolabel              ║
║                                                          ║
║  Label values:                                           ║
║    0 = Background / water / land                         ║
║    1 = Litter / debris  ← what the model learns          ║
║    2 = Vegetation / ambiguous                            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ───────────────────────────────────────────────────────────────────────────
# 7. AUTO-LABEL WITH FDI (quick bootstrap, imperfect)
# ───────────────────────────────────────────────────────────────────────────
def autolabel_with_fdi(cfg, fdi_threshold=0.05):
    """
    Generates rough labels using the Floating Debris Index.
    High FDI → likely litter. Use as a starting point, then correct manually.

    FDI = B8 - (B6 + (B11-B6) * 10 * (833-740)/(1610-740))
    Band indices in our array: B5→idx3(NIR/B8), B4→idx4(RE/B6), B5→idx5(SWIR/B11)
    """
    print("\n🤖 Auto-labelling with FDI threshold...")
    count = 0
    for loc_dir in sorted(Path(cfg["output_data_dir"]).iterdir()):
        if not loc_dir.is_dir():
            continue
        for patch_path in loc_dir.glob("*.npy"):
            arr      = np.load(patch_path)        # (C, H, W)
            b6, b8, b11 = arr[4], arr[3], arr[5]
            fdi      = b8 - (b6 + (b11 - b6) * ((833 - 740) / (1610 - 740)) * 10)

            label    = np.zeros(fdi.shape, dtype=np.int8)
            label[fdi > fdi_threshold]  = 1   # litter candidate
            label[fdi < -0.02]          = 2   # likely vegetation/water

            lbl_path = Path(cfg["output_label_dir"]) / loc_dir.name / patch_path.name
            np.save(lbl_path, label)
            count += 1

    print(f"  ✅ Auto-labelled {count} patches (review and correct these!)")


# ───────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ───────────────────────────────────────────────────────────────────────────
def main():
    import sys
    cfg = FETCH_CONFIG

    os.makedirs(cfg["output_data_dir"],  exist_ok=True)
    os.makedirs(cfg["output_label_dir"], exist_ok=True)
    os.makedirs(cfg["cache_dir"],        exist_ok=True)

    autolabel = "--autolabel" in sys.argv

    print("╔══════════════════════════════════════════════╗")
    print("║  Sentinel-2 Fetcher — Chicago Litter Project ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Backend:   {cfg['backend']}")
    print(f"  Locations: {len(cfg['locations'])}")
    print(f"  Dates:     {cfg['start_date']}  →  {cfg['end_date']}")
    print(f"  Cloud <:   {cfg['max_cloud_pct']}%")
    print(f"  Patch:     {cfg['patch_size']}×{cfg['patch_size']} px @ {cfg['target_res_m']}m\n")

    # ── Fetch ─────────────────────────────────────────────────────────────
    t0 = time.time()
    if cfg["backend"] == "planetary_computer":
        fetch_planetary_computer(cfg)
    elif cfg["backend"] == "gee":
        fetch_gee(cfg)
    elif cfg["backend"] == "copernicus":
        fetch_copernicus(cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg['backend']}")
    print(f"\n⏱  Fetch time: {(time.time()-t0)/60:.1f} minutes")

    # ── Verify ────────────────────────────────────────────────────────────
    verify_dataset(cfg)

    # ── Auto-label or instructions ────────────────────────────────────────
    if autolabel:
        autolabel_with_fdi(cfg)
        print("\n⚠️  Auto-labels are approximate — review them before full training!")
    else:
        print_labelling_instructions(cfg)

    print("\n🎉 Done! You can now run:")
    print("   python train_litter_temporal.py")


if __name__ == "__main__":
    main()
