"""
SARIMA Demand Forecasting — Northeast
DS 4420 Project | Rhea Johnson & Iba Baig
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")


EIA_API_KEY = os.environ.get("EIA_API_KEY", "rabrl0guSLeEilMwaUf8eQyg50dIGJ4wD6osFnTX")
OUTPUT_DIR  = "/Users/rheajohnson/Downloads/sarima_outputs"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "sarima_residuals.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE  = "2025-01-01"
END_DATE    = "2026-03-30"

ORDER    = (1, 1, 1)
SEASONAL = (1, 1, 1, 7)

STORM_WINDOWS = [
    ("2025-01-06", "2025-01-08"),
]

BASE_URL = "https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/"

def fetch_eia_demand(api_key, start, end, subba="4008", page_size=5000):
    all_records = []
    offset = 0

    params = {
        "api_key":            api_key,
        "frequency":          "daily",
        "data[0]":            "value",
        "facets[subba][]":    subba,
        "start":              start,
        "end":                end,
        "sort[0][column]":    "period",
        "sort[0][direction]": "asc",
        "length":             page_size,
    }

    print(f"Fetching daily EIA demand for subba {subba} ({start} → {end})...")

    while True:
        params["offset"] = offset
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

        records = payload.get("response", {}).get("data", [])
        if not records:
            break

        all_records.extend(records)
        total = int(payload.get("response", {}).get("total", 0))
        print(f"  fetched {len(all_records):,} / {total:,} rows", end="\r")

        if len(all_records) >= total:
            break

        offset += page_size
        time.sleep(0.2)

    print(f"\nDone — {len(all_records):,} daily records pulled.")
    return pd.DataFrame(all_records)

raw = fetch_eia_demand(EIA_API_KEY, START_DATE, END_DATE)

raw["datetime"] = pd.to_datetime(raw["period"], format="%Y-%m-%d", errors="coerce")
raw["value"]    = pd.to_numeric(raw["value"], errors="coerce")

raw = raw.dropna(subset=["datetime", "value"]).sort_values("datetime")
raw = raw.set_index("datetime")

daily = raw["value"].rename("demand_mwh")
daily = daily.replace(0, np.nan).dropna()

print(f"Date range: {daily.index.min().date()} → {daily.index.max().date()}")
print(f"N days: {len(daily)}")

adf_stat, adf_p, *_ = adfuller(daily.dropna())
print(f"\nADF stat: {adf_stat:.4f}  p-value: {adf_p:.4f}")
print("Non-stationary — d≥1 recommended." if adf_p > 0.05 else "Stationary.")

exog = pd.Series(0, index=daily.index, name="storm")
for start, end in STORM_WINDOWS:
    exog.loc[start:end] = 1
print(f"Storm days flagged: {exog.sum()}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(daily.diff().dropna(),  lags=40, ax=axes[0])
plot_pacf(daily.diff().dropna(), lags=40, ax=axes[1])
axes[0].set_title("ACF — First-differenced Daily Demand (Northeast Mass 4008)")
axes[1].set_title("PACF — First-differenced Daily Demand (Northeast Mass 4008)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "acf_pacf.png"), dpi=150)
plt.show()

print(f"\nFitting SARIMA{ORDER}x{SEASONAL} ...")
model  = SARIMAX(daily, exog=exog, order=ORDER, seasonal_order=SEASONAL,
                 enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(disp=False)
print(result.summary())

residuals = result.resid.rename("residual_mwh")

out = pd.DataFrame({
    "date":         daily.index.date,
    "demand_mwh":   daily.values,
    "fitted_mwh":   result.fittedvalues.values,
    "residual_mwh": residuals.values,
    "storm":        exog.values,
})
out.to_csv(OUTPUT_CSV, index=False)
print(f"\nResiduals saved → {OUTPUT_CSV}")

# 30 day forecast
FORECAST_STEPS = 30
exog_future = np.zeros(FORECAST_STEPS)
forecast = result.get_forecast(steps=FORECAST_STEPS, exog=exog_future)
forecast_mean = forecast.predicted_mean
forecast_ci   = forecast.conf_int(alpha=0.05)
forecast_index = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=FORECAST_STEPS, freq="D")
forecast_mean.index = forecast_index
forecast_ci.index   = forecast_index

# plots
def shade_winters(ax, index):
    for yr in range(index.year.min(), index.year.max() + 2):
        ax.axvspan(pd.Timestamp(f"{yr-1}-12-01"), pd.Timestamp(f"{yr}-02-28"),
                   alpha=0.07, color="steelblue", label="_nolegend_")

# plot 1: Actual vs fitted + 30-day Forecast with CI + residuals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

shade_winters(ax1, daily.index)
ax1.plot(daily.index, daily.values, color="steelblue", lw=0.9, label="Actual")
ax1.plot(daily.index, result.fittedvalues, color="orange", lw=0.9, alpha=0.8, label="Fitted")
ax1.plot(forecast_mean.index, forecast_mean.values, color="red", lw=1.2, label="30-day forecast")
ax1.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color="red", alpha=0.15, label="95% CI")
for s, e in STORM_WINDOWS:
    ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.25, color="crimson")
ax1.set_title("SARIMA Forecast — Northeast Mass (4008)")
ax1.set_ylabel("Demand (MWh)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)
ax1.legend()
ax1.set_facecolor("#fafafa")

ax2.plot(residuals.index, residuals.values, color="mediumpurple", lw=0.8)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title("Residuals — Northeast Mass (4008)")
ax2.set_ylabel("Residual (MWh)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
ax2.set_facecolor("#fafafa")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "forecast_and_residuals.png"), dpi=150)
plt.show()

# plot 2: Residuals by season (storm vs normal)
fig, ax = plt.subplots(figsize=(14, 4))
shade_winters(ax, residuals.index)
ax.axhline(0, color="gray", lw=0.8, linestyle="--")
ax.bar(residuals.index, residuals.values,
       color=["crimson" if s else "steelblue" for s in exog.values],
       width=1, alpha=0.7)
ax.legend(handles=[Patch(facecolor="steelblue", alpha=0.7, label="Normal day"),
                   Patch(facecolor="crimson",   alpha=0.7, label="Storm day")])
ax.set_title("SARIMA Residuals Across Seasons — Northeast Mass (4008)\n"
             "(positive = demand exceeded forecast)")
ax.set_ylabel("Residual (MWh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=30)
ax.set_facecolor("#fafafa")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_by_season.png"), dpi=150)
plt.show()

# plot 3: Residual distribution for storm vs normal
fig, ax = plt.subplots(figsize=(8, 4))
residuals[exog == 0].hist(ax=ax, bins=40, alpha=0.6, color="steelblue", label="Normal days")
residuals[exog == 1].hist(ax=ax, bins=15, alpha=0.7, color="crimson",   label="Storm days")
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("Residual Distribution: Storm vs Normal Days")
ax.set_xlabel("Residual (MWh)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_distribution.png"), dpi=150)
plt.show()

print("\nAll plots saved. Push sarima_residuals.csv to Git for Bayesian step.")
