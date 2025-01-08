import json

import matplotlib.pyplot as plt
import numpy as np

file_path = "./data/used2_ehullless20withchgnet_all_S05_Sx5/train.jsonl"
data_lines = []

with open(file_path, "r") as f:
    for line in f:
        data_point = json.loads(line)
        data_lines.append(data_point)

energies = []
for data_point in data_lines:
    energies.append(data_point["energy"])

print("Top 100 lowest energies:")
print(sorted(energies)[:100])
print("Top 100 highest energies:")
print(sorted(energies)[-100:])

# plot the histogram

plt.hist(energies, bins=100)
plt.xlabel("Energy (Hartree)")
plt.ylabel("Frequency")
plt.title("Histogram of Energy Distribution")
plt.savefig("histogram_train.png")
print("saved to histogram_train.png")

energies = np.asarray(energies)
print("Mean:", np.mean(energies))
print("Std:", np.std(energies))
energies = (energies - np.mean(energies)) / np.std(energies)
plt.hist(energies, bins=100)
plt.xlabel("Energy (Hartree)")
plt.ylabel("Frequency")
plt.title("Histogram of Energy Distribution")
plt.savefig("histogram_train_norm.png")
print("saved to histogram_train_norm.png")
