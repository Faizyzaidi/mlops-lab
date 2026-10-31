"""Creates data/raw.csv (150 rows, iris) used by Lab 2."""
import os
from sklearn.datasets import load_iris

os.makedirs("data", exist_ok=True)
load_iris(as_frame=True).frame.to_csv("data/raw.csv", index=False)
print("Created data/raw.csv")
