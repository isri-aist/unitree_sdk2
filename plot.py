import numpy as np
from matplotlib import pyplot as plt
import glob

SAVEFIGS = False

"""log = np.genfromtxt("duild/2024-07-31_17-06-43.txt", delimiter=",", 
                    filling_values=np.nan, case_sensitive=True, deletechars='', 
                    replace_space=' ', usecols=range(21))"""

import csv

files = glob.glob("duild/2024-08-*.txt")
files.sort()
print("-- Plotting ", files[-1])
datafile = open(files[-1], "r")
datareader = csv.reader(datafile)
data = {}
for row in datareader:
    """print("-----")
    print(row)
    print(row[0])

    print(row[0].split(",")[1:])"""
    # data.setdefault(row[0].split(",", 0)[0], []).append([float(elem) for elem in row[0].split(",")[1:]])
    data.setdefault(row[0], []).append([float(elem) for elem in row[1:]])

for key in data.keys():
    data[key] = np.array(data[key])


joint_indices = [7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15, 9]
leg_indices = joint_indices[:10]
arm_indices = joint_indices[11:19]

def plot_leg_arm(ys, indices, lgd1, lgd2, lbls, title, yunit):
    mod = len(indices)
    mod2 = int(mod / 2)
    fig, axs = plt.subplots(
        2,
        mod2,
        figsize=(30, 12),
        sharex=True,
    )
    for i in range(mod):
        for j, y in enumerate(ys):
            axs[int(i / mod2), i % mod2].plot(
                data["time"][:ys[j].shape[0]],
                ys[j][:, indices[i]],
                linestyle="-",
                linewidth=3,
                label=lbls[j],
            )
        # axs[int(i / mod2), i % mod2].set_title(title)
        axs[int(i / mod2), i % mod2].set_xlabel("Time [s]")
        axs[int(i / mod2), i % mod2].set_ylabel(
            lgd1[i % mod2] + " " + lgd2[int(i / mod2)] + yunit
        )
        axs[int(i / mod2), i % mod2].grid(True)
    plt.legend()
    # plt.xlim([0.0, t_end])
    plt.grid(True)
    fig.canvas.manager.set_window_title(title)
    # plt.tight_layout()


### Display various quantities related to the legs
lgd1_leg = ["Hip Yaw", "Hip Roll", "Hip Pitch", "Knee", "Ankle"]
lgd1_arm = ["Shoulder Pitch", "Shoulder Roll", "Shoulder Yaw", "Elbow"]
lgd2 = ["Left", "Right"]

# Display leg joint positions
plot_leg_arm(
    [data["q"], data["q_ref"]],
    leg_indices,
    lgd1_leg,
    lgd2,
    ["Measured", "Desired"],
    "Leg joint positions",
    " [rad]",
)

plot_leg_arm(
    [data["q"], data["q_ref"]],
    arm_indices,
    lgd1_arm,
    lgd2,
    ["Measured", "Desired"],
    "Arm joint positions",
    " [rad]",
)

if SAVEFIGS:
    plt.savefig(self.checkpoint_name + "_joint_pos.png")

plt.show()
