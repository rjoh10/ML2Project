# Fetch and clean data for hourly demand for Massachusetts subregions only from gov API
import os
import time
import requests
import pandas as pd

API_KEY = os.environ.get("EIA_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "EIA_API_KEY not set. Run: export EIA_API_KEY='your_key_here'"
    )

BASE_URL   = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
DATE_START = "2025-01-01T00"
DATE_END   = "2026-03-22T00"

MA_ZONES = {
    "4006": "W-C Massachusetts",
    "4007": "NE Massachusetts / Boston",
    "4008": "SE Massachusetts",
}

os.makedirs("data", exist_ok=True)
CACHE_PATH = "data/ma_hourly_raw.csv"

def fetch_zone(subba_code: str, label: str) -> pd.DataFrame:
    print(f"  Fetching {subba_code} — {label}")
    pages, offset, page_size = [], 0, 5000

    while True:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[subba][]": subba_code,
            "start": DATE_START,
            "end": DATE_END,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": page_size,
            "offset": offset,
        }, timeout=30)
        resp.raise_for_status()

        rows = resp.json().get("response", {}).get("data", [])
        if not rows:
            break

        pages.append(pd.DataFrame(rows))
        print(f"    {offset + len(rows)} rows so far...")

        if len(rows) < page_size:
            break
        offset += page_size
        time.sleep(0.3)

    if not pages:
        print(f"no data returned for {subba_code}")
        return pd.DataFrame()

    df = pd.concat(pages, ignore_index=True)
    df["subba_code"] = subba_code
    df["zone_name"]  = label
    return df


if os.path.exists(CACHE_PATH):
    raw_df = pd.read_csv(CACHE_PATH)
else:
    frames = [fetch_zone(code, label) for code, label in MA_ZONES.items()]
    raw_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    raw_df.to_csv(CACHE_PATH, index=False)


clean_df = (
    raw_df
    .rename(columns={"period": "period_str", "value": "demand_mwh"})
    .assign(
        datetime_utc = lambda d: pd.to_datetime(
            d["period_str"], format="%Y-%m-%dT%H", utc=True),
        datetime_et  = lambda d: d["datetime_utc"].dt.tz_convert(
            "America/New_York"),
        demand_mwh   = lambda d: pd.to_numeric(d["demand_mwh"], errors="coerce"),
    )

    .pipe(lambda d: d[d["demand_mwh"] > 0])
    [["datetime_utc", "datetime_et", "subba_code", "zone_name", "demand_mwh"]]
    .sort_values(["subba_code", "datetime_utc"])
    .reset_index(drop=True)
)

ma_total = (
    clean_df
    .groupby("datetime_utc")
    .agg(
        demand_mwh  = ("demand_mwh", "sum"),
        n_zones     = ("demand_mwh", "count"),
        datetime_et = ("datetime_et", "first"),
    )
    .reset_index()
    .query("n_zones == 3")
    .sort_values("datetime_utc")
    .reset_index(drop=True)
)


print(f"\nMassachusetts Total ({len(ma_total):,} hours)")
print(f"  Date range  : {ma_total['datetime_et'].min()} "
      f"to {ma_total['datetime_et'].max()}")
print(f"  Mean demand : {ma_total['demand_mwh'].mean():,.0f} MWh")
print(f"  Max demand  : {ma_total['demand_mwh'].max():,.0f} MWh")
print(f"  Min demand  : {ma_total['demand_mwh'].min():,.0f} MWh")

clean_df.to_csv("data/ma_subregion_clean.csv", index=False)
ma_total.to_csv("data/ma_total_hourly.csv", index=False)