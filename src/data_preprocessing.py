import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = "data/raw/housing.csv"
PROCESSED_PATH = "data/processed/housing_clean.csv"

def load_data(path):
    df = pd.read_csv(path)
    print("=== Data Loaded ===")
    print(f"Shape: {df.shape}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Stats:\n{df.describe().round(2)}")
    return df

def handle_missing_values(df):
    print("\n=== Handling Missing Values ===")
    before = df.isnull().sum().sum()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    after = df.isnull().sum().sum()
    print(f"Missing values: {before} → {after}")
    return df

def remove_outliers(df):
    print("\n=== Removing Outliers ===")
    initial = len(df)
    num_cols = ['median_house_value', 'median_income', 'total_rooms', 'population']
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"Rows removed: {initial - len(df)}")
    print(f"Remaining rows: {len(df)}")
    return df

def encode_categorical(df):
    print("\n=== Encoding Categorical Columns ===")
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    print(f"New shape after encoding: {df.shape}")
    return df

def feature_engineering(df):
    print("\n=== Feature Engineering ===")
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df['income_bracket'] = pd.cut(
        df['median_income'],
        bins=[0, 1.5, 3, 4.5, 6, 10],
        labels=[1, 2, 3, 4, 5]
    ).astype(float)
    print("New features: rooms_per_household, bedrooms_per_room,")
    print("              population_per_household, income_bracket")
    return df

def plot_correlation(df):
    print("\n=== Saving Correlation Heatmap ===")
    plt.figure(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    os.makedirs("data/processed", exist_ok=True)
    plt.savefig('data/processed/correlation_heatmap.png', dpi=150)
    print("Saved: data/processed/correlation_heatmap.png")

    print("\nTop correlations with median_house_value:")
    print(corr['median_house_value'].sort_values(ascending=False).round(3))

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"\n=== Saved Clean Data ===")
    print(f"Path: {path}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    df = load_data(RAW_PATH)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = encode_categorical(df)
    df = feature_engineering(df)
    plot_correlation(df)
    save_data(df, PROCESSED_PATH)
    print("\nStep 1 Complete! data/processed/housing_clean.csv ready.")