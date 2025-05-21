import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("wandb_export_2025-05-15T14_13_04.126-04_00.csv")

print(df.columns)

# Filter columns with '@' in the name and 'top_fitness' (not MIN/MAX)
fitness_cols = [col for col in df.columns[:40] if '@' in col and col.endswith('top_fitness') and '__' not in col and "evo" not in col]

# Pick the first 6 runs (or specify them manually if you want)
fitness_cols = fitness_cols[:6]

# Define your 6 custom colors
custom_colors = ['#E9762B', '#90D1CA', '#27548A', '#9467bd', ]

plt.figure(figsize=(10, 6))

for idx, col in enumerate(fitness_cols):
    steps = df['Step']
    fitness = pd.to_numeric(df[col], errors='coerce')
    # Drop NaNs for plotting
    mask = fitness.notna()
    steps = steps[mask]
    fitness = fitness[mask]
    if len(fitness) == 0:
        continue
    
    # If "pairwise" in column name, shift left by removing the first value
    if "pairwise" in col:
        steps = steps.iloc[:-1]
        fitness = fitness.iloc[1:]
        
    # If "4x" in column name, scale steps by 0.25
    if "4x" in col:
        steps = steps * 0.25
        
    perplexity = np.exp(1 / fitness)
    color = custom_colors[idx % len(custom_colors)]
    # Plot original (faint)
    plt.plot(steps, perplexity, label=None, alpha=0.3, color=color)
    # Smoothed line (bold)
    perplexity_smooth = perplexity # pd.Series(perplexity).rolling(window=4, min_periods=1).mean()
    plt.plot(steps, perplexity_smooth, label=col.split(' - ')[0], linewidth=2, color=color)

plt.xlabel('Step')
plt.ylabel('Perplexity')
plt.title('Perplexity over Outer Steps')
plt.legend()
plt.grid(True)
plt.ylim(15, 100)
plt.tight_layout()
plt.show()