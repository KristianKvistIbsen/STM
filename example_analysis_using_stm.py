import numpy as np
import plotly.io as pio
from pySTM.stm_result_handler import STMSynthesizer
from pySTM.excitation_handler import frequency_independent_excitation_from_csv

pio.renderers.default = "browser"

# =============================================================================
# I set the STM path and the filepath where i have a csv with constant pressure
# values in Pa for a number of XYZ coordinates.
# =============================================================================
STM_FILEPATH = r"produced_stms\STM_TP_MODAL.h5"
CSV_FILEPATH = r"N:\PhD\STM\TP\simple_flow_total_pressure.csv"

# =============================================================================
# I use the STMSynthesizer to load in the STM object
# =============================================================================
print("Loading STM Synthesizer...")
STM = STMSynthesizer.from_file(STM_FILEPATH)

# Print a summary of the loaded STM model
print("\n--- STM Summary ---")
for key, val in STM.summary().items():
    print(f"{key:>20}: {val}")

# =============================================================================
# The pressures are mapped to GammaI and decomposed to the excitation vector.
# UPDATED: Added return_figure=True and unpacking the third variable (fig_recon)
# =============================================================================
print("\nMapping CSV pressure data and projecting to basis...")
excitation, error_percent, fig_recon = frequency_independent_excitation_from_csv(
    stm_synthesizer=STM,
    csv_filepath=CSV_FILEPATH,
    num_neighbors=3,
    p=2,
    return_figure=True,  # Triggers the side-by-side plot generation
    plot_part="abs"      # Plots the magnitude of the pressure
)

if isinstance(error_percent, float):
    print(f"Bulk L2 Reconstruction Error: {error_percent:.4f} %")

# Show the 3D diagnostic figure immediately to inspect the error
print("Opening Pressure Reconstruction Diagnostic...")
fig_recon.show()


# =============================================================================
# Compute the radiated power by multiplying STM with excitation for all freqs.
# =============================================================================
power_data = STM.radiated_power(excitation)
total_power_db = power_data["total_db"]
print(f"\nMax total radiated power: {total_power_db[~np.isnan(total_power_db)].max():.2f} dB")


# =============================================================================
# Figures
# =============================================================================
print("Generating Acoustic and Surface Velocity Figures...")

fig_power = STM.figure_power_spectrum(excitation, title="Far field sound power levels")
fig_power.show()

if STM.has_error_data:
    fig_error = STM.figure_error_spectrum(use_relative=True, title="Surface velocity reconstruction error (independent of excitation!)")
    fig_error.show()
else:
    print("No internal error data found in this STM. Skipping error spectrum plot.")

freq_idx_to_plot = 0 
freq_hz = STM.frequencies[freq_idx_to_plot]
print(f"Plotting 3D surface velocity for frequency {freq_hz:.1f} Hz...")

fig_vel = STM.figure_surface_velocity(
    excitation, 
    freq_index=freq_idx_to_plot, 
    part="abs",           # Options: "real", "imag", "abs", "phase"
    colorscale="Turbo",
    title=f"Surface Normal Velocity Magnitude @ {freq_hz:.1f} Hz"
)
fig_vel.show()
