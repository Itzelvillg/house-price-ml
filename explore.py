import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("houses.csv")

# How does size relate to price?
plt.figure(figsize=(8, 5))
plt.scatter(df["Square_Footage"], df["House_Price"], alpha=0.4, color="#5C8DFF")
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
