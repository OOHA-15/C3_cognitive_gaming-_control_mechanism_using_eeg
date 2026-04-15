import serial
import pandas as pd
import time

# -----------------------------
# SETTINGS
# -----------------------------
PORT = "COM9"
BAUD = 115200
DURATION = 30
CYCLES = 10

# -----------------------------
# ENTER MEMBER NAME HERE
# -----------------------------
member = input("Enter member name (mem1/mem2/mem3/mem4): ")

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

print(f"\nConnected... Collecting data for {member}")

# -----------------------------
def collect_data(label, duration):
    data = []
    start = time.time()

    while time.time() - start < duration:
        try:
            value = float(ser.readline().decode().strip())
            data.append([value, label])
        except:
            pass

    return data

# -----------------------------
member_data = []

for cycle in range(CYCLES):

    print(f"\nCycle {cycle+1}/{CYCLES}")

    # ATTENTIVE
    print("ATTENTIVE → Focus!")
    time.sleep(2)
    member_data.extend(collect_data(1, DURATION))

    print("Rest...")
    time.sleep(5)

    # RELAXED
    print("RELAXED → Close eyes")
    time.sleep(2)
    member_data.extend(collect_data(0, DURATION))

    print("Cycle complete")
    time.sleep(3)

# -----------------------------
df = pd.DataFrame(member_data, columns=["value", "label"])
df.to_csv(f"{member}_dataset.csv", index=False)

print(f"\n✅ {member}_dataset.csv saved successfully!")