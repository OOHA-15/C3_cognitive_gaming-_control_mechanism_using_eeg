import pandas as pd

# -----------------------------
# Attentive datasets
# -----------------------------
att_files = [
    "mem1_attentive.csv",
    "mem2_attentive.csv",
    "mem3_attentive.csv",
    "mem4_attentive.csv"
]

att_data = []

for file in att_files:
    df = pd.read_csv(file)
    df["label"] = 1   # 1 = attentive
    att_data.append(df)

att = pd.concat(att_data)


# -----------------------------
# Relaxed datasets
# -----------------------------
rel_files = [
    "mem1_relaxed.csv",
    "mem2_relaxed.csv",
    "mem3_relaxed.csv",
    "mem4_relaxed.csv"
]

rel_data = []

for file in rel_files:
    df = pd.read_csv(file)
    df["label"] = 0   # 0 = relaxed
    rel_data.append(df)

rel = pd.concat(rel_data)


# -----------------------------
# Combine attentive + relaxed
# -----------------------------
dataset = pd.concat([att, rel])

# Keep only value + label columns
dataset = dataset[["value","label"]]

# Save final dataset
dataset.to_csv("final_dataset.csv", index=False)

print("✅ Final dataset created successfully")