import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Read data from CSV-file
df = pd.read_csv('dataset.csv')

# Rename columns to lowercase with underscores
df.columns = ['date', 'rate']

# Convert a date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Display basic information about the data
print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nОсновные статистики:")
print(df.describe())

# Check for missing values
print("Missing values:")
print(df.isnull().sum())
print(f"\nMissing value rate: {df.isnull().mean()}")

# Check for 'NaN' values
print(f"\nNaN values: {df.isna().sum().sum()}")

# Handle missing values ​​(if any)
if df.isnull().sum().sum() > 0:
    # Fill the gaps using the forward fill method (with the last known value)
    df = df.fillna(method='ffill')
    print("Missing values filled")
else:
    print("No missing values")

# Additional checking for data anomalies
print(f"\nMinimum rate value: {df['rate'].min()}")
print(f"Maximum rate value: {df['rate'].max()}")