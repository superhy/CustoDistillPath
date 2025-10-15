'''
Created on 11 Mar 2025

@author: yang hu
'''

import numpy as np
import matplotlib.pyplot as plt

# Model names
models = ["conch_v15", "phikon_v2", "uni_v2", "hoptimus0", "gigapath", "virchow_v2", "ensemble"]

# # ACC data: order is Default, 512, 256, 128; for ensemble, Default is set to np.nan
# acc = np.array([
#     [83.45, 81.72, 81.38, 78.97],
#     [85.52, 84.83, 83.79, 80.69],
#     [84.83, 84.14, 82.76, 81.38],
#     [85.86, 84.48, 86.21, 80.69],
#     [83.79, 83.10, 84.14, 84.48],
#     [83.79, 86.55, 86.21, 82.41],
#     [np.nan, 85.17, 85.52, 86.55]
# ])
#
# # AUC data: order is Default, 512, 256, 128; for ensemble, Default is set to np.nan
# auc = np.array([
#     [85.75, 84.74, 84.69, 84.30],
#     [90.89, 89.06, 88.90, 85.87],
#     [90.17, 88.87, 86.56, 86.73],
#     [91.39, 90.29, 89.90, 87.14],
#     [88.05, 89.06, 87.91, 86.85],
#     [90.07, 90.79, 88.76, 87.76],
#     [np.nan, 90.94, 90.79, 90.43]
# ])

# # ACC data: order is Default, 512, 256, 128; for ensemble, Default is set to np.nan
# acc = np.array([
#     [75.52, 75.17, 73.45, 72.41],
#     [84.13, 82.07, 81.38, 80.34],
#     [82.07, 81.03, 79.66, 76.90],
#     [82.76, 82.76, 83.45, 77.93],
#     [78.62, 80.00, 78.62, 81.03],
#     [80.34, 79.66, 81.38, 77.59],
#     [np.nan, 84.48, 81.38, 80.61]
# ])
#
# # AUC data: order is Default, 512, 256, 128; for ensemble, Default is set to np.nan
# auc = np.array([
#     [84.18, 78.68, 80.58, 80.79],
#     [88.68, 88.29, 88.61, 86.06],
#     [88.77, 87.88, 86.06, 85.36],
#     [89.47, 88.77, 89.93, 85.38],
#     [87.74, 87.04, 85.75, 87.00],
#     [90.75, 88.68, 87.86, 86.35],
#     [np.nan, 91.32, 89.56, 89.66]
# ])

# ACC data: order is Default, 512, 256, 128; for ensemble, Default is set to np.nan
acc = np.array([
    [75.52, 75.17, 73.45, 72.41],
    [84.13, 82.07, 81.38, 80.34],
    [82.07, 81.03, 79.66, 76.90],
    [82.76, 82.76, 83.45, 77.93],
    [78.62, 80.00, 78.62, 81.03],
    [80.34, 79.66, 81.38, 77.59],
    [np.nan, 85.17, 85.52, 86.55]
])

# AUC data: order is Default, 512, 256, 128; for ensemble, Default is set to np.nan
auc = np.array([
    [84.18, 78.68, 80.58, 80.79],
    [88.68, 88.29, 88.61, 86.06],
    [88.77, 87.88, 86.06, 85.36],
    [89.47, 88.77, 89.93, 85.38],
    [87.74, 87.04, 85.75, 87.00],
    [90.75, 88.68, 87.86, 86.35],
    [np.nan, 90.94, 90.79, 90.43]
])

# Configuration names
configs = ["Default", "512", "256", "128"]

# Create a 2x2 subplot figure
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Define bar width for grouped bar charts
width = 0.18  

# X positions for grouped bar charts (one group per model)
x = np.arange(len(models))

# -------------------------
# Plot ACC grouped bar chart (Top-Left)
for i in range(len(configs)):
    axs[0, 0].bar(x + (i - (len(configs) - 1)/2) * width, acc[:, i], width, label=configs[i])
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(models)
axs[0, 0].set_ylabel("ACC (%)")
axs[0, 0].set_title("ACC Grouped Bar Chart")
axs[0, 0].legend()
axs[0, 0].set_ylim(70, 95)
axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

# -------------------------
# Plot ACC line chart (Top-Right)
x_line = np.arange(len(configs))  # x positions for line chart
for i in range(len(models)):
    axs[0, 1].plot(x_line, acc[i, :], marker='o', label=models[i])
axs[0, 1].set_xticks(x_line)
axs[0, 1].set_xticklabels(configs)
# Remove y-axis label for line chart
axs[0, 1].set_title("ACC Trend Across Configurations")
axs[0, 1].legend()
axs[0, 1].set_ylim(70, 95)  # modified y-axis range
axs[0, 1].grid(axis="both", linestyle="--", alpha=0.7)

# -------------------------
# Plot AUC grouped bar chart (Bottom-Left)
for i in range(len(configs)):
    axs[1, 0].bar(x + (i - (len(configs) - 1)/2) * width, auc[:, i], width, label=configs[i])
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(models)
axs[1, 0].set_ylabel("AUC (%)")
axs[1, 0].set_title("AUC Grouped Bar Chart")
axs[1, 0].legend()
axs[1, 0].set_ylim(70, 95)
axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

# -------------------------
# Plot AUC line chart (Bottom-Right)
for i in range(len(models)):
    axs[1, 1].plot(x_line, auc[i, :], marker='o', label=models[i])
axs[1, 1].set_xticks(x_line)
axs[1, 1].set_xticklabels(configs)
# Remove y-axis label for line chart
axs[1, 1].set_title("AUC Trend Across Configurations")
axs[1, 1].legend()
axs[1, 1].set_ylim(70, 95)  # modified y-axis range
axs[1, 1].grid(axis="both", linestyle="--", alpha=0.7)

# Adjust layout for better spacing
plt.tight_layout()
# plt.show()
# Save the figure to a file in the current directory
plt.savefig("ensemble_performance.png", dpi=900)


