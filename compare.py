import matplotlib.pyplot as plt

# Read csv
import numpy as np


bc_name = "logs/eval/bc/bc_cnn_eval.csv"
bc_al_name = "logs/eval/bc/bc_cnn_al_eval.csv"

# Read the csv as a numpy array
df_bc = np.genfromtxt(bc_name, delimiter=",")
df_bc = df_bc[1:, :]
df_bc = df_bc[~np.isnan(df_bc).any(axis=1)]
x_bc = df_bc[:, 0].astype(int)
y_bc = df_bc[:, 1]
print(x_bc)
print(y_bc)

df_bc_al = np.genfromtxt(bc_al_name, delimiter=",")
df_bc_al = df_bc_al[1:, :]
df_bc_al = df_bc_al[~np.isnan(df_bc_al).any(axis=1)]
x_bc_al = df_bc_al[:, 0].astype(int)
y_bc_al = df_bc_al[:, 1]
print(x_bc_al)
print(y_bc_al)

x_bc, y_bc = zip(*sorted(zip(x_bc, y_bc)))
x_bc_al, y_bc_al = zip(*sorted(zip(x_bc_al, y_bc_al)))

log_dir = "."

# Plot the BC results in blue
plt.plot(x_bc, y_bc, label="BC")
# Plot the BC-AL results in red
plt.plot(x_bc_al, y_bc_al, label="BC-AL")

plt.legend()


plt.title("Evaluation Reward vs. Number of Expert Samples")
plt.xlabel("Number of Expert Samples")
plt.ylabel("Reward")
plt.savefig(f"{log_dir}/reward_cnn_al.png")
plt.show()
