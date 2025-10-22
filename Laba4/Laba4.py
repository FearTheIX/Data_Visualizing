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

# Calculating the median and mean
median_rate = df['rate'].median()
mean_rate = df['rate'].mean()

print(f"Median rate: {median_rate}")
print(f"Mean rate: {mean_rate}")

# Adding columns with deviations
df['deviation_from_median'] = df['rate'] - median_rate
df['deviation_from_mean'] = df['rate'] - mean_rate
df['abs_deviation_from_median'] = abs(df['rate'] - median_rate)
df['abs_deviation_from_mean'] = abs(df['rate'] - mean_rate)

print("\nData with added deviation columns:")
print(df.head())

# Statistical analysis of the main columns
columns_to_analyze = ['rate', 'deviation_from_median', 'deviation_from_mean', 
                     'abs_deviation_from_median', 'abs_deviation_from_mean']

print("Statistical info:")
print(df[columns_to_analyze].describe())

# Visualizing outliers with Boxplot
plt.figure(figsize=(15, 10))

for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot(2, 3, i)
    df[column].plot(kind='box')
    plt.title(f'Boxplot for {column}')
    plt.ylabel('Value')

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Outlier analysis using IQR
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

print("\nOutliers in main rate:")
outliers_rate = detect_outliers_iqr(df['rate'])
print(f"Outliers rate: {len(outliers_rate)}")
print(f"Outliers percentage: {len(outliers_rate)/len(df):.4f}")

def filter_by_deviation(dataframe, deviation_threshold):
    """
    Filter the DataFrame by the value of deviation from the mean
    
    Parameters:
    dataframe (pd.DataFrame): Original DataFrame
    deviation_threshold (float): Deviation threshold
    
    Returns:
    pd.DataFrame: Filtered DataFrame
    """
    filtered_df = dataframe[dataframe['abs_deviation_from_mean'] >= deviation_threshold]
    return filtered_df

def filter_by_date(dataframe, start_date, end_date):
    """
    Filters the DataFrame by a date range
    
    Parameters:
    dataframe (pd.DataFrame): Original DataFrame
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    pd.DataFrame: Filtered DataFrame
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    mask = (dataframe['date'] >= start_date) & (dataframe['date'] <= end_date)
    filtered_df = dataframe[mask]
    return filtered_df

# Function usage example
print("Example of filtering by deviation (threshold = 10):")
filtered_dev = filter_by_deviation(df, 10)
print(f"Number of rows after filtering: {len(filtered_dev)}")

print("\nExample of filtering by date (2000-2001):")
filtered_date = filter_by_date(df, '2000-01-01', '2001-12-31')
print(f"Number of rows after filtering: {len(filtered_date)}")