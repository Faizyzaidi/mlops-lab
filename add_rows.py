"""Lab 2 Task 1: append 5 new rows to data/raw.csv."""
import pandas as pd

df = pd.read_csv("data/raw.csv")
new = df.sample(5, random_state=1)
pd.concat([df, new]).to_csv("data/raw.csv", index=False)
print("Rows now:", len(df) + 5)
