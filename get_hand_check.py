import pandas as pd

year = 2016

# Read the CSV file using pandas
df = pd.read_csv(f"dataset_{year}.csv", sep=";", encoding="utf-8-sig")

# Print header and first row
print(df.columns.tolist())
print(df.iloc[0].tolist())

# Sample 100 rows with a fixed random seed for reproducibility
df = df.dropna(subset=["indoklas"])
sample_df = df.sample(n=100, random_state=42)

# Write to output file
sample_df.to_csv(f"dataset_{year}_sample.csv", index=False)
