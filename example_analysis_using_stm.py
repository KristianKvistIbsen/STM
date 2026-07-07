import numpy as np
from pySTM.stm_result_handler import STMSynthesizer
from pySTM.excitation_handler import frequency_independent_excitation_from_csv, excitation_from_array

# =============================================================================
# Configuration
# =============================================================================
STM_FILEPATH = r"C:/01_gitrepos/STM/STM_test.h5"
CSV_FILEPATH = r"N:\PhD\STM\TP\simple_flow_total_pressure.csv"

# =============================================================================
# Load the STM
# =============================================================================
print("Loading STM Synthesizer...")
STM = STMSynthesizer.from_file(STM_FILEPATH)

print("\n--- STM Summary ---")
for key, val in STM.summary().items():
    print(f"{key:>20}: {val}")

# =============================================================================
# Map CSV pressure data to Γ_I and project onto the basis
# =============================================================================
print("\nMapping CSV pressure data and projecting to basis...")

# New version: returns only (excitation, error_percent)
# Set plot=True to automatically show the VTK pressure reconstruction diagnostic
# excitation, error_percent = frequency_independent_excitation_from_csv(
#     stm_synthesizer=STM,
#     csv_filepath=CSV_FILEPATH,
#     num_neighbors=3,
#     p=2,
#     plot=True,              # ← New: automatically shows VTK reconstruction plot
#     plot_part="abs"
# )


excitation = excitation_from_array(STM,np.array([1+0j,1+0j,0+2j,1-1j]),export_csv_path="testpressure.csv")


# if isinstance(error_percent, float):
#     print(f"Bulk L2 Reconstruction Error: {error_percent:.4f} %")

# =============================================================================
# Compute radiated sound power
# =============================================================================
print("\nCalculating radiated sound power...")
power_data = STM.radiated_power(excitation)
total_power_db = power_data["total_db"]
# %%


valid_power = total_power_db[~np.isnan(total_power_db)]
print(f"Max total radiated power: {valid_power.max():.2f} dB re 1 pW")

# =============================================================================
# Visualization (all plot_* methods display immediately)
# =============================================================================
print("\nGenerating visualizations...")

# 1. Radiated Power Spectrum (Matplotlib)
STM.plot_power_spectrum(
    excitation,
    title="Far-field Sound Power Levels"
)

# 2. Error Spectrum (only if error data exists)
if STM.has_error_data:
    STM.plot_error_spectrum(
        use_relative=True,
        title="Surface Velocity Reconstruction Error (independent of excitation)"
    )
else:
    print("No internal error data found in this STM file. Skipping error spectrum.")

# 3. Surface Normal Velocity on External Mesh (PyVista / VTK)
freq_idx_to_plot = 0
freq_hz = STM.frequencies[freq_idx_to_plot]

print(f"Plotting 3D surface velocity at {freq_hz:.1f} Hz...")

STM.plot_surface_velocity(
    excitation,
    freq_index=freq_idx_to_plot,
    part="abs",                    # Options: "real", "imag", "abs", "phase"
    cmap="turbo",
    title=f"Surface Normal Velocity Magnitude @ {freq_hz:.1f} Hz"
)

print("\nDone.")
