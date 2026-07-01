import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set matplotlib style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8

# Read the dataset
data = pd.read_csv(r"C:\Users\105849\Grundfos\Simulation Driven Development - Industrial PhD - Kristian Hansen 2024-2027\05_Papers\JSV GTM\Data\SIMPLE_ACOUSTIC_AND_MODDED_RESULT.csv")

# Calculate RMSE between STM and both FEA variants
rmse_original = np.sqrt(np.mean((data['FEA'] - data['STM'])**2))
rmse_modified = np.sqrt(np.mean((data['moddedFEA'] - data['STM'])**2))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(16, 6))

# Plot all three datasets
ax.plot(data['Freq'], data['FEA'], linestyle='-', color='#bf0000', 
        label=r'FEA, $\Gamma_I^{\text{Original}}$')
ax.plot(data['Freq'], data['moddedFEA'], linestyle='-', color='#ff6600', 
        label=r'FEA, $\Gamma_I^{\text{Modified}}$')
ax.plot(data['Freq'], data['STM'], linestyle='-', color='#005b96', 
        label='STM')

# Customize the plot
ax.set_xlabel('Frequency (Hz)', fontsize=20)
ax.set_ylabel('Far-field sound power level [dB], ref=1pW', fontsize=20)
ax.set_title(f'Comparison of FEA variants and STM Acoustic Results for "Simple system"\n'
             f'RMSE Original: {rmse_original:.2f} dB, RMSE Modified: {rmse_modified:.2f} dB', 
             fontsize=22, pad=15)
ax.legend(fontsize=20, frameon=True, edgecolor='black', loc='lower right')
ax.grid(True, linestyle='-', alpha=0.7)

# Adjust axis ticks and limits
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim(min(data['Freq']) - 2, max(data['Freq']) + 2)

# Calculate y-limits considering all three datasets
y_min = min(min(data['FEA']), min(data['STM']), min(data['moddedFEA'])) - 1
y_max = max(max(data['FEA']), max(data['STM']), max(data['moddedFEA'])) + 1
ax.set_ylim(y_min, y_max)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the figure in high resolution for publication
plt.savefig('acoustic_results_plot_extended.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print RMSE values for reference
print(f"RMSE (STM vs FEA Original): {rmse_original:.3f} dB")
print(f"RMSE (STM vs FEA Modified): {rmse_modified:.3f} dB")
