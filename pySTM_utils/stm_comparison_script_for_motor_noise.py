import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming these are in your environment based on previous context
from pySTM.stm_result_handler import STMSynthesizer
from pySTM.excitation_handler import excitation_from_array

POWER_REF = 1e-12

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_db(power, ref=POWER_REF):
    """Convert linear power to dB."""
    p = np.asarray(power, float)
    out = np.full(p.shape, np.nan)
    pos = p > 0
    out[pos] = 10.0 * np.log10(p[pos] / ref)
    return out

def get_third_octave_bands(freqs, power):
    """
    Calculate 1/3 octave band levels. 
    Dynamically filters out bands that fall outside the provided frequency range.
    """
    # Standard nominal 1/3 octave center frequencies (Hz)
    nom_fc = np.array([
        10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 
        100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 
        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000
    ])
    
    levels = []
    valid_fc = []
    
    for fc in nom_fc:
        # Exact 1/3 octave band limits
        f_lower = fc / (2**(1/6))
        f_upper = fc * (2**(1/6))
        
        mask = (freqs >= f_lower) & (freqs < f_upper)
        if np.any(mask):
            band_power = np.sum(power[mask])
            levels.append(to_db(band_power))
            valid_fc.append(str(fc))
            
    # Filter out bands with no data (NaN)
    valid_indices = ~np.isnan(levels)
    return [f for f, v in zip(valid_fc, valid_indices) if v], np.array(levels)[valid_indices]


# =============================================================================
# USER CONFIGURATION
# =============================================================================

parameter_name = "Tooth Root Radius (mm)"

# 1. Define your designs (Name -> File Path & Parameter Value)
designs = {
    "7mm thinner": {"file": r"N:\PhD\motor_noise\stms\dIR_6mm_less_material.h5",  "param": 110.64},
    "4mm thinner": {"file": r"N:\PhD\motor_noise\stms\dIR_3mm_less_material.h5",  "param": 107.64},
    "As is":       {"file": r"N:\PhD\motor_noise\stms\base.h5",                   "param": 103.64},
    "2mm thicker": {"file": r"N:\PhD\motor_noise\stms\dIR_3mm_more_material.h5",  "param": 101.64}
}

# 2. Define your excitation vectors
excitation_vectors = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
]


# =============================================================================
# COMPUTATION
# =============================================================================

results = {}
print("Loading files and computing acoustic metrics...")

for name, info in designs.items():
    stm = STMSynthesizer.from_file(info["file"])
    results[name] = {"param": info["param"], "excitations": {}}
    
    for i, vec in enumerate(excitation_vectors):
        exc_name = f"Excitation {i+1}"
        exc = excitation_from_array(stm, vec)
        w = stm.radiated_power(exc)["total"]
        
        # Overall Level
        overall_db = to_db(np.nansum(w))
        
        # 1/3 Octave Bands
        fc_labels, bands_db = get_third_octave_bands(stm.frequencies, w)
        
        results[name]["excitations"][exc_name] = {
            "overall_db": overall_db,
            "bands_db": bands_db
        }
    # Store the frequency labels (assumes all designs share the same valid frequency grid)
    results[name]["fc_labels"] = fc_labels


# =============================================================================
# CONSOLE OUTPUT (TABLES)
# =============================================================================

print(f"\n{'='*70}")
print(f"OVERALL SOUND POWER LEVELS (dB re 1 pW)")
print(f"{'='*70}")

df_overall = pd.DataFrame(index=[f"Excitation {i+1}" for i in range(len(excitation_vectors))])
df_overall.index.name = "Excitation"

for name, res in results.items():
    col_name = f"{name}\n(param={res['param']})"
    df_overall[col_name] = [res["excitations"][f"Excitation {i+1}"]["overall_db"] for i in range(len(excitation_vectors))]
print(df_overall.round(1).to_string())


for i in range(len(excitation_vectors)):
    exc_name = f"Excitation {i+1}"
    print(f"\n{'='*70}")
    print(f"1/3 OCTAVE BANDS FOR {exc_name.upper()} (dB re 1 pW)")
    print(f"{'='*70}")
    
    fc_labels = results[list(results.keys())[0]]["fc_labels"]
    df_band = pd.DataFrame(index=fc_labels)
    df_band.index.name = "fc (Hz)"
    
    for name, res in results.items():
        col_name = f"{name}\n(param={res['param']})"
        df_band[col_name] = res["excitations"][exc_name]["bands_db"]
        
    print(df_band.round(1).to_string())


# =============================================================================
# SENSITIVITY ANALYSIS (dSPL / dPARAM)
# =============================================================================

print(f"\n{'='*70}")
print(f"SENSITIVITY ANALYSIS: Best-Fit Slope (dSPL / d{parameter_name})")
print(f"{'='*70}")

param_vals = np.array([res["param"] for res in results.values()])
slopes = {}

for i in range(len(excitation_vectors)):
    exc_name = f"Excitation {i+1}"
    levels = np.array([res["excitations"][exc_name]["overall_db"] for res in results.values()])
    
    # Linear regression: level = slope * param + intercept
    slope, intercept = np.polyfit(param_vals, levels, 1)
    slopes[exc_name] = {"slope": slope, "intercept": intercept}

df_sensitivity = pd.DataFrame(slopes).T
df_sensitivity.index.name = "Excitation"
df_sensitivity.columns = [f"Slope (dB/{parameter_name.split('(')[0].strip()})", "Intercept (dB)"]
print(df_sensitivity.round(3).to_string())


# =============================================================================
# PLOTTING (INTERACTIVE)
# =============================================================================

plt.rcParams.update({'figure.autolayout': True})

# PLOT 1: Overall Level vs Parameter Value WITH TRENDLINES
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(excitation_vectors)):
    exc_name = f"Excitation {i+1}"
    levels = [res["excitations"][exc_name]["overall_db"] for res in results.values()]
    
    # Plot actual data points
    ax.plot(param_vals, levels, 'o-', label=f"{exc_name} (Data)", markersize=8, linewidth=2)
    
    # Plot best-fit trendline
    slope = slopes[exc_name]["slope"]
    intercept = slopes[exc_name]["intercept"]
    trend_y = slope * param_vals + intercept
    
    # Format slope for legend
    sign = "+" if slope >= 0 else ""
    ax.plot(param_vals, trend_y, '--', alpha=0.6, linewidth=1.5,
            label=f"{exc_name} Trend ({sign}{slope:.3f} dB/mm)")

ax.set_xlabel(parameter_name, fontsize=12)
ax.set_ylabel("Overall Sound Power Level (dB re 1 pW)", fontsize=12)
ax.set_title(f"Overall Sound Power vs {parameter_name}", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(title="Excitations & Sensitivities")
plt.show()


# PLOT 2: 1/3 Octave Heatmaps vs Parameter Value (One per excitation)
for i in range(len(excitation_vectors)):
    exc_name = f"Excitation {i+1}"
    
    fc_labels = results[list(results.keys())[0]]["fc_labels"]
    fc_nums = [float(f) for f in fc_labels]
    
    # Build Z matrix (Rows = Freq bands, Cols = Designs/Parameters)
    Z = np.zeros((len(fc_nums), len(param_vals)))
    for j, (name, res) in enumerate(results.items()):
        Z[:, j] = res["excitations"][exc_name]["bands_db"]
        
    fig, ax = plt.subplots(figsize=(10, 7))
    X, Y = np.meshgrid(param_vals, fc_nums)
    c = ax.pcolormesh(X, Y, Z, cmap='turbo', shading='auto')
    
    ax.set_xscale('linear')
    ax.set_yscale('log') # Log scale for frequency axis is standard for acoustics
    
    # Format Y ticks
    ax.set_yticks(fc_nums)
    ax.set_yticklabels(fc_labels)
    
    ax.set_xlabel(parameter_name, fontsize=12)
    ax.set_ylabel("1/3 Octave Band Center Frequency (Hz)", fontsize=12)
    ax.set_title(f"1/3 Octave Spectrum vs {parameter_name}\n{exc_name}", fontsize=14)
    
    plt.colorbar(c, ax=ax, label="Sound Power Level (dB re 1 pW)")
    plt.show()
