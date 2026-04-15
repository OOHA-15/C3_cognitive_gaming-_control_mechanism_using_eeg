import pandas as pd

files = [
    "mem1_dataset.csv",
    "mem2_dataset.csv",
    "mem3_dataset.csv",
    "mem4_dataset.csv"
]

data = []

for file in files:
    df = pd.read_csv(file)
    data.append(df)

final = pd.concat(data)

final.to_csv("final_dataset.csv", index=False)

print("✅ final_dataset.csv created")