# 🏠 Predicting House Prices with Machine Learning

> **Codedex Challenge** · Python · scikit-learn · Kaggle Dataset

Hey! Welcome to your first machine learning project. In this challenge, you're going to train a model that predicts house prices based on real data — things like size, number of bedrooms, and how old the house is.

No worries if ML feels intimidating right now. By the end of this, you'll have a working model and a solid understanding of *why* it works.

Let's go! 

---

##  Wait... what even is machine learning?

Instead of writing rules like:

```python
# ❌ The old way (no fun)
if bedrooms == 3 and size > 1500:
    price = 300000
```

You give the computer **examples** and let it figure out the pattern:

```python
# ✅ The ML way
model.fit(X_train, y_train)   # "here are 800 houses, learn from them"
model.predict([[1500, 3, 2]]) # "ok, what would THIS house cost?"
```

That's it. That's the whole idea. 

---

## The Dataset

We're using the **[Home Value Insights](https://www.kaggle.com/datasets/prokshitha/home-value-insights)** dataset from Kaggle — 1,000 real houses with these columns:

| Column | What it means |
|--------|--------------|
| `Square_Footage` | Size of the house in sq ft |
| `Num_Bedrooms` | Number of bedrooms |
| `Num_Bathrooms` | Number of bathrooms |
| `Year_Built` | Year the house was built |
| `Lot_Size` | Size of the land (in acres) |
| `Price` |  What we're trying to predict |

Download it from Kaggle and save it as `houses.csv` inside the project folder.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Itzelvillg/house-price-ml.git
cd house-price-ml
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### 3. Download the dataset

Go to  [kaggle.com/datasets/prokshitha/home-value-insights](https://www.kaggle.com/datasets/prokshitha/home-value-insights), download `Housing.csv`, and put it in the project folder.

---

##  Project structure

```
house-price-ml/
│
├── README.md
├── houses.csv          ← dataset goes here
├── model.py            ← main script (train + predict)
└── explore.py          ← optional: visualize the data first
```

---

##  The Code

Here's the full `model.py` — read the comments, they explain everything:

```python
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
y = df["Price"]    # output we want to predict

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
```

---

##  Run it!

```bash
python model.py
```

You should see something like:

```
Dataset shape: (1000, 6)

   Square_Footage  Num_Bedrooms  ...     Price
0            2134             4  ...  432000.0
1            1560             3  ...  298500.0
...

Training on 800 houses, testing on 200

 Mean Absolute Error: $21,340
(That's how many dollars off we are on average)

 A 1800 sq ft house built in 2005:
   Predicted price → $347,821
```

---

##  Bonus: Explore the data first

Before training, it's always a good idea to *look* at your data. Run `explore.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("houses.csv")

# How does size relate to price?
plt.figure(figsize=(8, 5))
plt.scatter(df["Square_Footage"], df["Price"], alpha=0.4, color="#5C8DFF")
plt.xlabel("Square Footage")
plt.ylabel("Price ($)")
plt.title("Size vs Price")
plt.tight_layout()
plt.savefig("size_vs_price.png")
print("Chart saved as size_vs_price.png")

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="Blues")
plt.title("Feature Correlations")
plt.tight_layout()
plt.savefig("correlations.png")
print("Chart saved as correlations.png")
```

```bash
python explore.py
```

---

##  Key concepts 

| Term | Plain English |
|------|--------------|
| **Feature** | An input the model uses (size, bedrooms...) |
| **Target / Label** | What we're predicting (price) |
| **Training data** | Examples the model learns from |
| **Test data** | New examples to check if the model actually learned |
| **MAE** | Mean Absolute Error — average $ we're off by |
| **`.fit()`** | The learning step |
| **`.predict()`** | Using what was learned to make predictions |

---

##  Want to go further?

Try these upgrades and see if your MAE improves:

```python
# Swap LinearRegression for a more powerful model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## 🔗 Resources

- 📂 [Dataset on Kaggle](https://www.kaggle.com/datasets/prokshitha/home-value-insights)
- 📖 [scikit-learn docs](https://scikit-learn.org/stable/)


---

Made with 🤍 for the Codedex community by Itzel
