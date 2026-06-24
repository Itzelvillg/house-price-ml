import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── 1. Load the data ──────────────────────────────────────────
df = pd.read_csv("houses.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ── 2. Pick features (inputs) and target (what to predict) ────
features = ["Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
            "Year_Built", "Lot_Size"]

X = df[features]   # inputs
y = df["House_Price"]    # output we want to predict

# ── 3. Split into train and test ──────────────────────────────
# 80% to learn from, 20% to test honestly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining on {len(X_train)} houses, testing on {len(X_test)}")

# ── 4. Create and train the model ────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)   # this is where the magic happens 

# ── 5. Make predictions on the test set ──────────────────────
predictions = model.predict(X_test)

# ── 6. See how well we did ───────────────────────────────────
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error: ${mae:,.0f}")
print("(That's how many dollars off we are on average)")

# ── 7. Try predicting a brand new house ──────────────────────
# [Square_Footage, Bedrooms, Bathrooms, Year_Built, Lot_Size]
new_house = [[1800, 3, 2, 2005, 0.25]]

predicted_price = model.predict(new_house)
print(f"\n A 1800 sq ft house built in 2005:")
print(f"   Predicted price → ${predicted_price[0]:,.0f}")
