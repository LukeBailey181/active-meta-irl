import matplotlib.pyplot as plt
# Read csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description="Visualize an experiment's results")

parser.add_argument(
      "-y",
      "--config",
      type=str,
      required=True,
      help="Configuration file to run from. Any additional arguments will overwrite config.",
  )

parser.add_argument(
  "-n",
  "--log_name",
  type=str,
  default=None,
  help="The name of the log file to save to.",
)

args = parser.parse_args()

# Read the csv as a numpy array
df = np.genfromtxt(args.log_name, delimiter=",")
df = df[1:, :]
df = df[~np.isnan(df).any(axis=1)]
x = df[:, 0].astype(int)
y = df[:, 1]
print(x)
print(y)

# Sort by x
x, y = zip(*sorted(zip(x, y)))


log_dir = args.log_name.split("/")[0:-1]
log_dir = "/".join(log_dir)

# Plot the results
plt.plot(x, y)

plt.title("IL/IRL Reward vs. Number of Expert Samples")
plt.xlabel("Number of Expert Samples")
plt.ylabel("Reward")
plt.savefig(f"{log_dir}/reward.png")
plt.show()
