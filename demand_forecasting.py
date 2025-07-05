import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


warnings.filterwarnings("ignore")


df = pd.read_csv("retail_sales.csv")  

print(" Dataset Info:")
print(df.info())

print("\n First 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


df["Date"] = pd.to_datetime(df["Date"])


daily_demand = df.groupby("Date")["Units Sold"].sum()


plt.figure(figsize=(12, 5))
plt.plot(daily_demand, label="Daily Units Sold", color='navy')
plt.title("🗓️ Daily Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("daily_demand_timeseries.png")
plt.show()


model = SARIMAX(daily_demand, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
results = model.fit()


forecast = results.get_forecast(steps=30)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()


plt.figure(figsize=(12, 5))
plt.plot(daily_demand, label='Actual', color='blue')
plt.plot(forecast_values, label='Forecast', color='red')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3)
plt.title("🔮 Forecast for Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_30_days.png")
plt.show()
