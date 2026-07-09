import numpy as np
from pySTM.stm_result_handler import STMSynthesizer
from pySTM.excitation_handler import frequency_independent_excitation_from_csv, excitation_from_array, evaluate_basis_truncation_error



stl = r"N:\PhD\STM\TP\TP_pumphousing_shrinkwrap_inner.stl"
pressure_csv = r"N:\PhD\STM\TP\simple_flow_total_pressure.csv"
evaluate_basis_truncation_error()
# %%


STM_FILEPATH = r"C:/01_gitrepos/STM/test.h5"

STM = STMSynthesizer.from_file(STM_FILEPATH)

excitation = excitation_from_array(STM,np.array([1+0j,1+0j,0+2j,1-1j]),export_csv_path="testpressure.csv")


print("\n--- STM Summary ---")
for key, val in STM.summary().items():
    print(f"{key:>20}: {val}")


# excitation, error_percent = frequency_independent_excitation_from_csv(
#     stm_synthesizer=STM,
#     csv_filepath=CSV_FILEPATH,
#     num_neighbors=3,
#     p=2,
#     plot=True,              
#     plot_part="abs"
# )



# plot_freq_index = 5
# NNI = STM.synthesize_intensity(excitation,method="NNI",plot=True,plot_freq_index=plot_freq_index)


power_data = STM.radiated_power(excitation)
total_power_db = power_data["total_db"]



# STM.plot_power_spectrum(
#     excitation,
#     title="Far-field Sound Power Levels"
# )

# if STM.has_error_data:
#     STM.plot_error_spectrum(
#         use_relative=True,
#         title="Surface Velocity Reconstruction Error (independent of excitation)"
#     )
# else:
#     print("No internal error data found in this STM file. Skipping error spectrum.")

STM.plot_surface_velocity(
    excitation,
    freq_index=1,
    part="imag",                    # Options: "real", "imag", "abs", "phase"
    cmap="turbo"
)
