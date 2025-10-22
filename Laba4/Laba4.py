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
print("������ 5 ����� ������:")
print(df.head())
print("\n���������� � ������:")
print(df.info())
print("\n�������� ����������:")
print(df.describe())
