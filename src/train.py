import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "data/processed/housing_clean.csv"
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train, y_train,
                                 cv=5, scoring='r2')

    print(f"\n=== {name} ===")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {"name": name, "model": model,
            "rmse": rmse, "mae": mae, "r2": r2,
            "cv_r2": cv_scores.mean(), "y_pred": y_pred}

def plot_results(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, res in enumerate(results):
        axes[i].scatter(y_test, res['y_pred'], alpha=0.3, s=10, color='steelblue')
        axes[i].plot([y_test.min(), y_test.max()],
                     [y_test.min(), y_test.max()], 'r--', linewidth=1.5)
        axes[i].set_title(f"{res['name']}\nR² = {res['r2']:.4f}")
        axes[i].set_xlabel('Actual Price')
        axes[i].set_ylabel('Predicted Price')

    plt.tight_layout()
    plt.savefig('data/processed/model_comparison.png', dpi=150)
    plt.show()
    print("Saved: data/processed/model_comparison.png")

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(feat_df['feature'], feat_df['importance'], color='steelblue')
    plt.title('Feature Importance (Best Model)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png', dpi=150)
    plt.show()
    print("Saved: data/processed/feature_importance.png")

def save_model(model, scaler):
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")

if __name__ == "__main__":
    print("=== Loading Data ===")
    X, y = load_data()

    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("\n=== Scaling Features ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n=== Training Models ===")
    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBRegressor(
            n_estimators=100, learning_rate=0.1,
            random_state=42, verbosity=0)),
    ]

    results = []
    for name, model in models:
        if name == "Linear Regression":
            res = evaluate_model(name, model,
                                 X_train_scaled, X_test_scaled,
                                 y_train, y_test)
        else:
            res = evaluate_model(name, model,
                                 X_train, X_test,
                                 y_train, y_test)
        results.append(res)

    print("\n=== Plotting Results ===")
    plot_results(results, y_test)

    best = max(results, key=lambda x: x['r2'])
    print(f"\n=== Best Model: {best['name']} ===")
    print(f"R² Score: {best['r2']:.4f}")

    if best['name'] != "Linear Regression":
        plot_feature_importance(best['model'], X.columns)

    save_model(best['model'], scaler)
    print("\nStep 3 Complete! models/best_model.pkl ready.")