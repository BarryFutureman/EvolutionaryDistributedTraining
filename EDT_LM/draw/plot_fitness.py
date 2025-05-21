import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("wandb_export_2025-05-14T13_30_34.740-04_00.csv")

# Filter columns: keep only those with '@' in the run name
filtered_cols = [col for col in df.columns if "@" in col]
# Always keep the Step column for x-axis
cols_to_plot = ["Step"] + filtered_cols if "Step" in df.columns else filtered_cols

# Plot each filtered run
for col in filtered_cols:
    plt.plot(df["Step"], df[col], label=col)

plt.xlabel("Step")
plt.ylabel("Fitness")
plt.title("Filtered runs (with '@' in name)")
plt.legend()
plt.tight_layout()
plt.show()
