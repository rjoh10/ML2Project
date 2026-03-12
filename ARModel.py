# Model 1 Proof of Concept
# DS 4420 Project Phase 1
# Iba Baig and Rhea Johnson

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#Clean Weather, Generation, and Winter Storm Data
#Filter Data to past One Year
#Merge Data Sets, include columns:
# EIA: datetime, region, demand_mw
# NOAA weather: date, station, tavg, tmin, tmax, prcp, snow, awnd
# Storm events: BEGIN_DATE_TIME, END_DATE_TIME, EVENT_TYPE, STATE, CZ_NAME


#Read and convert to Time format
df = pd.read_csv("MergedData.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")
df = df.set_index("datetime")

# Target series
y = df["demand"]

# Train/test split
train_size = int(len(y) * 0.8)
train, test = y.[:train_size], y.[train_size:]

# Fit ARmodel
model = AutoReg(train, lags=24).fit()

# Forecast on one year sample
predictions = model.predict(start=len(train), end=len(train) + len(test) - 1)

# Evaluate Results
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print("Autoregressive Proof-of-Concept Results")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Preview actual vs predicted
results = pd.DataFrame({
    "Actual": test.values,
    "Predicted": predictions.values
}, index=test.index)

print(results.head())