import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
SEED = 42
np.random.seed(SEED)
N = 5_000
sqft        = np.random.randint(500,  5_000, N)
bedrooms    = np.random.randint(1,    7,     N)
bathrooms   = np.random.randint(1,    5,     N)
year_built  = np.random.randint(1950, 2024,  N)
zip_code    = np.random.choice([10001,10002,90210,94102,60601,
                                 30301,77001,85001,98101,2101], N)
zip_multiplier = {10001:1.8, 10002:1.6, 90210:2.2, 94102:2.0, 60601:1.4,
                  30301:1.1, 77001:1.0, 85001:1.1, 98101:1.7, 2101:1.5}
multiplier = np.array([zip_multiplier[z] for z in zip_code])
price = (
    sqft       * 120 * multiplier
    + bedrooms * 15_000
    + bathrooms* 10_000
    + (year_built - 1950) * 500
    + np.random.normal(0, 25_000, N)
).clip(50_000, 5_000_000).astype(int)
df = pd.DataFrame({
    "sqft":       sqft,
    "bedrooms":   bedrooms,
    "bathrooms":  bathrooms,
    "year_built": year_built,
    "zip_code":   zip_code,
    "price":      price,
})
print(f"Dataset shape : {df.shape}")
print(df.describe().to_string())
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestRegressor(
        n_estimators=200, max_depth=15,
        random_state=SEED, n_jobs=-1)),
])
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"\nTest MAE : ${mae:,.0f}")
print(f"Test R²  : {r2:.4f}")
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/house_price_model.joblib")
print("\nModel saved → models/house_price_model.joblib")
