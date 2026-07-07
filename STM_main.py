import os
import numpy as np
from scipy.spatial import cKDTree
import pyshtools as pysh
from ansys.dpf import core as dpf
from ansys.workbench.core import connect_workbench
from ansys.mechanical.core import connect_to_mechanical

import kdpf
import pySDEM
import pySTM

# =============================================================================
# USER SETTINGS
# =============================================================================
model_folder = None  # Defaults to dp0/MECH of system if None
pressure_export_folder = None  # Defaults to model_folder if None
INTERNAL_NS = 'GAMMA_I'
EXTERNAL_NS = 'GAMMA_E'

STM_NAME = "STM_test"

nCores = 8
lmax_I = 1
lmax_O = 60
workbench_server_port = 59361 # StartServer() to retrieve port
workbench_server_ip = None

SHRINK_WRAP_STL_INTERNAL = None
SHRINK_WRAP_STL_EXTERNAL = None
SHRINK_WRAP_MAP_FILTER_RADIUS = 0.005

# Pressure File Import Settings
systemName = "SYS"
DataExtension = "csv"
DelimiterIs = "Comma"
DelimiterStringIs = ","
StartImportAtLine = 2
LengthUnit = "m"
PressureUnit = "Pa"

# =============================================================================
# WORKBENCH & MECHANICAL CONNECTIONS
# =============================================================================
workbench = connect_workbench(
    port=workbench_server_port,
    host=workbench_server_ip if workbench_server_ip else None
)
mechPort = workbench.start_mechanical_server(systemName)
mechanical = connect_to_mechanical(ip='localhost', port=mechPort)

# Harmonic solver-files directory auto-detection
if model_folder is None:
    model_folder = mechanical.run_python_script("""
model = ExtAPI.DataModel.Project.Model
harmonics = [a for a in model.Analyses if "Harmonic" in a.AnalysisType.ToString()]
if len(harmonics) == 0:
    raise Exception("No Harmonic Response analysis found in the shared Mechanical model.")
h = harmonics[0]
_wd = None
for _attr in ["WorkingDir", "SolverFilesDirectory"]:
    if hasattr(h, _attr):
        _wd = getattr(h, _attr)
        break
if _wd is None or str(_wd) == "":
    raise Exception("Could not determine the harmonic solver files directory (no WorkingDir/SolverFilesDirectory).")
str(_wd)
""")
    model_folder = os.path.normpath(model_folder.strip())
    print(f"Harmonic solver files directory: {model_folder}")

if pressure_export_folder is None:
    pressure_export_folder = os.path.join(model_folder, "STM_pressures")
os.makedirs(pressure_export_folder, exist_ok=True)

# =============================================================================
# MONOPOLE SETUP & SOLVE
# =============================================================================
monopole_solve_command = f"""
model = ExtAPI.DataModel.Project.Model
harmonics = [a for a in model.Analyses if "Harmonic" in a.AnalysisType.ToString()]
if len(harmonics) == 0:
    raise Exception("No Harmonic Response analysis found in the shared Mechanical model.")
analysis = harmonics[0]
monopole_pressure = analysis.AddPressure()
named_selection = ExtAPI.DataModel.GetObjectsByName("{INTERNAL_NS}")[0]
monopole_pressure.Location = named_selection
magnitude_field = monopole_pressure.Magnitude
magnitude_field.Output.SetDiscreteValue(index=0,value=Ansys.Core.Units.Quantity("1 [Pa]"))
monopole_pressure.Name = "Y_0_0"
analysis.Solve(True)
analysis.Solution.GetResults()
"""

monopole_suppress_command = """
monopole_pressure.Suppressed = True
analysis.ClearGeneratedData()
"""

mechanical.run_python_script(monopole_solve_command)
mechanical.wait_till_mechanical_is_ready()

# =============================================================================
# MESH LOADING & PREPROCESSING (EXTERNAL & INTERNAL)
# =============================================================================
model = dpf.Model(model_folder + r"\file.rst")

# External Mesh
gammaO_from_ansys = kdpf.get_skin_mesh_from_ns(EXTERNAL_NS, model)
gammaO, mapping_workflow_external = pySTM.check_genus_zero_and_map_if_needed(
    gammaO_from_ansys, "EXTERNAL", SHRINK_WRAP_STL_EXTERNAL
)
original_points_gammaO = gammaO.grid.points.copy()
grid_gammaO = gammaO.grid.clean(remove_unused_points=True).compute_cell_sizes(length=False, area=True, volume=False)
v_gammaO = grid_gammaO.points
f_gammaO = grid_gammaO.cells_dict[list(grid_gammaO.cells_dict)[0]]
population_gammaO = grid_gammaO["Area"]

# SDEM External
S_gammaO = pySDEM.SphericalDensityEqualizingMap(v_gammaO, f_gammaO, population_gammaO)
R_gammaO, _ = pySDEM.optimal_rotation(v_gammaO, S_gammaO)
S_gammaO = S_gammaO @ R_gammaO
x_gammaO, y_gammaO, z_gammaO = S_gammaO[:, 0], S_gammaO[:, 1], S_gammaO[:, 2]
r_gammaO, lat_gammaO, lon_gammaO = pySDEM.cart_to_lat_lon(x_gammaO, y_gammaO, z_gammaO)

# Internal Mesh
gammaI_from_ansys = kdpf.get_skin_mesh_from_ns(INTERNAL_NS, model)
gammaI, mapping_workflow_internal = pySTM.check_genus_zero_and_map_if_needed(
    gammaI_from_ansys, "INTERNAL", SHRINK_WRAP_STL_INTERNAL
)
original_points_gammaI = gammaI.grid.points.copy()
grid_gammaI = gammaI.grid.clean(remove_unused_points=True).compute_cell_sizes(length=False, area=True, volume=False)
v_gammaI = grid_gammaI.points
f_gammaI = grid_gammaI.cells_dict[list(grid_gammaI.cells_dict)[0]]
population_gammaI = grid_gammaI["Area"]

# SDEM Internal
S_gammaI = pySDEM.SphericalDensityEqualizingMap(v_gammaI, f_gammaI, population_gammaI)
R_gammaI, _ = pySDEM.optimal_rotation(v_gammaI, S_gammaI)
S_gammaI = S_gammaI @ R_gammaI
x_gammaI, y_gammaI, z_gammaI = S_gammaI[:, 0], S_gammaI[:, 1], S_gammaI[:, 2]
r_gammaI, lat_gammaI, lon_gammaI = pySDEM.cart_to_lat_lon(x_gammaI, y_gammaI, z_gammaI)

# =============================================================================
# EXTRACT MONOPOLE VELOCITY
# =============================================================================
normals_O = kdpf.get_normals(gammaO_from_ansys)
tfreq = kdpf.get_tfreq(model)

vn = kdpf.get_normal_velocities(model, gammaO_from_ansys, tfreq, normals_O)
if mapping_workflow_external is not None:
    mapping_workflow_external.connect('source', vn)
    vn_with_potential_interpolated_values = mapping_workflow_external.get_output('target', output_type="fields_container")
    vn = pySTM.enforce_zero_outside_radius(
        mapped_fc=vn_with_potential_interpolated_values, 
        source_mesh=gammaO_from_ansys, 
        target_mesh=gammaO, 
        filter_radius=SHRINK_WRAP_MAP_FILTER_RADIUS
    )

vn_list = [vn]
model.metadata.release_streams()

# =============================================================================
# HARMONIC GENERATION & SOLVE LOOP
# =============================================================================
# Generate and export using pySTM module
spherical_harmonics_array, n_harmonics = pySTM.generate_spherical_harmonics(lmax_I, S_gammaI, lat_gammaI, lon_gammaI)
allfiles = pySTM.export_spherical_harmonics(spherical_harmonics_array, v_gammaI, pressure_export_folder, lmax_I)
print(f"Spherical harmonics exported: {len(allfiles)} files to {pressure_export_folder}")

mechanical.run_python_script(monopole_suppress_command)
mechanical.wait_till_mechanical_is_ready()

# Import data to Workbench
pySTM.setup_external_data(workbench, systemName, pressure_export_folder, allfiles,
                    StartImportAtLine=StartImportAtLine, DelimiterIs=DelimiterIs,
                    DelimiterStringIs=DelimiterStringIs, LengthUnit=LengthUnit,
                    PressureUnit=PressureUnit)

# Set core count once
set_cores_command = f"""
wbAnalysisName = "TARGET: HansenAutoImporter"
for item in ExtAPI.DataModel.AnalysisList:
    if item.SystemCaption == wbAnalysisName:
        analysis = item
analysis.SolveConfiguration.SolveProcessSettings.MaxNumberOfCores = {nCores}
"""
mechanical.run_python_script(set_cores_command)
mechanical.wait_till_mechanical_is_ready()

# Solve each harmonic
for fileid, filename in enumerate(allfiles, 1):
    vn, gammaO_ret, tfreq = pySTM.solve_model(mechanical, filename, fileid, INTERNAL_NS, EXTERNAL_NS, model_folder, gammaO_from_ansys, normals_O, tfreq)
    if mapping_workflow_external is not None:
        mapping_workflow_external.connect('source', vn)
        vn = mapping_workflow_external.get_output('target', output_type="fields_container")
        vn = pySTM.enforce_zero_outside_radius(
            mapped_fc=vn, 
            source_mesh=gammaO_from_ansys, 
            target_mesh=gammaO, 
            filter_radius=SHRINK_WRAP_MAP_FILTER_RADIUS
        )
    vn_list.append(vn)
    print(f"Solved for {filename}")

# =============================================================================
# STM MATRIX ASSEMBLY (LEAST SQUARES)
# =============================================================================
print("Calculating STM")
N_frequencies = len(tfreq.data)
N_points = len(lat_gammaO)

# Nearest-neighbour map
_, point_mapping_gammaO = cKDTree(original_points_gammaO).query(v_gammaO)
n_coeffs_I = (lmax_I + 1) ** 2
n_coeffs_O = (lmax_O + 1) ** 2
G = pysh.expand.LSQ_G(lat_gammaO, lon_gammaO, lmax_O)

# Construct large B matrix
B_large = np.zeros((N_points, n_coeffs_I * N_frequencies), dtype=complex)
for fileid in range(n_coeffs_I):
    vn = vn_list[fileid]
    field_data = np.array([vn[i].data for i in range(2 * N_frequencies)])
    block = field_data[0::2][:, point_mapping_gammaO] + 1j * field_data[1::2][:, point_mapping_gammaO]
    B_large[:, fileid * N_frequencies:(fileid + 1) * N_frequencies] = block.T

print("B_large constructed --> LSQ Solve started")
X_large, residuals_large, _, _ = np.linalg.lstsq(G, B_large, rcond=None)
print("Done --> Unpacking")

# Unpack X_large into STM
STM = X_large.T.reshape(n_coeffs_I, N_frequencies, n_coeffs_O).transpose(0, 2, 1)

# Errors
residual = G @ X_large - B_large
abs_error = np.linalg.norm(residual, axis=0).reshape(n_coeffs_I, N_frequencies)
b_norms = np.linalg.norm(B_large, axis=0).reshape(n_coeffs_I, N_frequencies)
rel_error = np.divide(abs_error, b_norms, out=np.zeros_like(abs_error), where=b_norms > 0)

print("Done")

# =============================================================================
# METADATA & RESULTS PACKAGING
# =============================================================================
mesh_data = {
    'INTERNAL': {
        'InternalGrid': grid_gammaI,
        'SDEM_Coordinates': S_gammaI,
        'mesh_metadata': {'nnodes': len(gammaI.grid.points), 'nelements': gammaI.grid.n_cells, 'areas': grid_gammaI["Area"]}
    },
    'EXTERNAL': {
        'ExternalGrid': grid_gammaO,
        'SDEM_Coordinates': S_gammaO,
        'mesh_metadata': {'nnodes': len(gammaO.grid.points), 'nelements': gammaO.grid.n_cells, 'areas': grid_gammaO["Area"]}
    },
    'FULL_MESH': {
        'FullMesh': model.metadata.meshed_region.grid,
        'mesh_metadata': {'nnodes': model.metadata.meshed_region.grid.n_points,
                          'nelements': model.metadata.meshed_region.grid.n_cells, 'areas': np.zeros([model.metadata.meshed_region.grid.n_cells])}
    }
}

metadata = {
    'user_settings': {
        'model_folder': model_folder, 'INTERNAL_NS': INTERNAL_NS, 'EXTERNAL_NS': EXTERNAL_NS,
        'lmax_I': lmax_I, 'lmax_O': lmax_O, 'nCores': nCores,
        'workbench_server_port': workbench_server_port, 'workbench_server_ip': workbench_server_ip,
        'solution_method': 'MSUP',
    },
    'pressure_file_settings': {
        'systemName': systemName, 'DataExtension': DataExtension, 'DelimiterIs': DelimiterIs,
        'DelimiterStringIs': DelimiterStringIs, 'StartImportAtLine': StartImportAtLine,
        'LengthUnit': LengthUnit, 'PressureUnit': PressureUnit, 'pressure_export_folder': pressure_export_folder
    },
    'computation_parameters': {
        'n_coeffs_I': n_coeffs_I,
        'n_coeffs_O': n_coeffs_O,
        'n_harmonics': n_harmonics,
        'n_points_internal': len(S_gammaI),
        'n_points_external': len(S_gammaO)
    },
    'frequency_data': {
        'n_frequencies': len(tfreq.data), 'frequency_unit': 'Hz',
        'frequency_range': [float(tfreq.data.min()), float(tfreq.data.max())]
    }
}

results_data = {
    'STM': STM,
    'G': G,
    'frequencies': tfreq.data,
    'export_files': {'harmonic_files': allfiles, 'n_files_exported': len(allfiles), 'file_pattern': 'Y_l_m.csv'},
    'point_mappings': {'point_mapping': np.array(point_mapping_gammaO), 'n_original_points': len(original_points_gammaO),
                       'n_cleaned_points': len(v_gammaO)},
    'error_data': {
        'abs_error': abs_error, 'rel_error': rel_error}
}

sh_labels = []
for l in range(lmax_I + 1):
    for m in range(-l, l + 1):
        sh_labels.append(f"Y_{l}_{m}")

monopole = np.ones((spherical_harmonics_array.shape[0], 1), dtype=np.complex128)
full_basis_array = np.hstack((monopole, spherical_harmonics_array))

results_data['input_basis'] = {
    'type': 'spherical_harmonics',
    'labels': sh_labels,                            
    'n_coeffs_I': n_coeffs_I,
    'basis_vectors': full_basis_array,
    'gammaI_points': v_gammaI,
    'lmax_I': lmax_I
}

# Package results
pySTM.package_stm_results(STM=STM, mesh_data=mesh_data, metadata=metadata, results_data=results_data,
                   output_file=STM_NAME+".h5")
