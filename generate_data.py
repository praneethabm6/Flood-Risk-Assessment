import pandas as pd
import numpy as np
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Generating realistic environmental features
rainfall = np.random.uniform(50, 400, n_samples)          # Rainfall in mm
river_discharge = np.random.uniform(100, 2000, n_samples)  # m³/s
elevation = np.random.uniform(5, 500, n_samples)           # meters above sea level
soil_moisture = np.random.uniform(20, 95, n_samples)       # percentage

# Calculate risk score based on features
risk_score = (
    (rainfall / 400) * 0.4 + 
    (river_discharge / 2000) * 0.4 + 
    (1 - (elevation / 500)) * 0.1 + 
    (soil_moisture / 100) * 0.1
)

# Map scores to categorical risk levels matching your project objectives
conditions = [
    (risk_score < 0.3),
    (risk_score >= 0.3) & (risk_score < 0.5),
    (risk_score >= 0.5) & (risk_score < 0.7),
    (risk_score >= 0.7)
]
choices = ['Low', 'Moderate', 'High', 'Very High']
risk_level = np.select(conditions, choices, default='Moderate')

# Create DataFrame and save to CSV
df = pd.DataFrame({
    'rainfall_mm': rainfall,
    'river_discharge': river_discharge,
    'elevation_m': elevation,
    'soil_moisture_%': soil_moisture,
    'flood_risk': risk_level
})

df.to_csv('data/flood_risk_data.csv', index=False)
print("Dataset generated successfully and saved to data/flood_risk_data.csv!")
print("\nFirst 5 rows:")
print(df.head())