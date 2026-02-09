import pickle
import matplotlib.pyplot as plt

with open("logs/run_log.pkl", "rb") as f:
    log = pickle.load(f)

steps = log["step"]

plt.figure(figsize=(12, 8))

# 1️⃣ Green duration
plt.subplot(2, 2, 1)
plt.plot(log["green"], marker='o')
plt.title("Green Duration per Decision")
plt.xlabel("Decision")
plt.ylabel("Seconds")

# 2️⃣ Phase timeline
plt.subplot(2, 2, 2)
plt.step(steps, log["phase"], where="post")
plt.title("Signal Phase Over Time")
plt.xlabel("Simulation Step")
plt.ylabel("Phase ID")

# 3️⃣ Queue length
plt.subplot(2, 2, 3)
plt.plot(steps, log["queue"])
plt.title("Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Vehicles")

# 4️⃣ Reward
plt.subplot(2, 2, 4)
plt.plot(log["reward"])
plt.title("Reward per Decision")
plt.xlabel("Decision")
plt.ylabel("Reward")

plt.tight_layout()
plt.show()
