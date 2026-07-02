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
from stm_core import (
    check_genus_zero_and_map_if_needed,
    enforce_zero_outside_radius,
    export_named_fields,
    setup_external_data,
    solve_model,
)

model_folder = None  # Harmonic solver-files dir; auto-detected if None
pressure_export_folder = None # Pressure files are placed in ANSYS system folder if None
INTERNAL_NS = 'GI' # Named selection of pressure loaded surface
EXTERNAL_NS = 'GE' # Named selection of external sound radiating surface

STM_NAME = "STM_TP_MODAL" # Name of exported STM file

nCores = 8 # N_cores to use when solving
lmax_O = 60                      # output (GammaO) spherical-harmonic fitting degree
workbench_server_port = 1045    # StartServer() to retrieve port
workbench_server_ip = None 

SHRINK_WRAP_STL_INTERNAL = None
SHRINK_WRAP_STL_EXTERNAL = r"N:\PhD\STM\TP\TP_pumphousing_shrinkwrap.stl"
SHRINK_WRAP_MAP_FILTER_RADIUS = 0.005

# --- In-vacuo modal-basis settings --------------------------------------------
AREA_WEIGHTED_SVD = True   # orthonormalise in the surface-L2 (area-weighted) inner product
SVD_REL_TOL = 1e-6         # drop basis vectors with singular value < SVD_REL_TOL * largest
N_BASIS_MAX = 1         # optional hard cap on the number of retained basis vectors

# Pressure File Import Settings
systemName = "SYS 1"
DataExtension = "csv"
DelimiterIs = "Comma"
DelimiterStringIs = ","
StartImportAtLine = 2
LengthUnit = "m"
PressureUnit = "Pa"


workbench = connect_workbench(
    port=workbench_server_port,
    host=workbench_server_ip if workbench_server_ip else None
)
mechPort = workbench.start_mechanical_server(systemName)
mechanical = connect_to_mechanical(ip='localhost', port=mechPort)


# Harmonic solver-files directory (the Workbench system name may not match its dp0 folder).
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

# Modal solver-files directory (the modal analysis is assumed to be already solved).
modal_folder = mechanical.run_python_script("""
model = ExtAPI.DataModel.Project.Model
modals = [a for a in model.Analyses if "Modal" in a.AnalysisType.ToString()]
if len(modals) == 0:
    raise Exception("No Modal analysis found in the shared Mechanical model.")
modal = modals[0]
_wd = None
for _attr in ["WorkingDir", "SolverFilesDirectory"]:
    if hasattr(modal, _attr):
        _wd = getattr(modal, _attr)
        break
if _wd is None or str(_wd) == "":
    raise Exception("Could not determine the modal solver files directory (no WorkingDir/SolverFilesDirectory).")
str(_wd)
""")
modal_folder = os.path.normpath(modal_folder.strip())
print(f"Modal solver files directory: {modal_folder}")

if pressure_export_folder is None:
    pressure_export_folder = os.path.join(model_folder, "STM_pressures")
os.makedirs(pressure_export_folder, exist_ok=True)


# ============================================================================
# Read the shared model (via the solved modal result file) for both surfaces.
# ============================================================================
modal_model = dpf.Model(modal_folder + r"\file.rst")

# ---- GammaO (output/radiating surface): SDEM parameterisation for SH fitting ----
gammaO_from_ansys = kdpf.get_skin_mesh_from_ns(EXTERNAL_NS, modal_model)
gammaO, mapping_workflow_external = check_genus_zero_and_map_if_needed(
    gammaO_from_ansys, "EXTERNAL", SHRINK_WRAP_STL_EXTERNAL, filter_radius=SHRINK_WRAP_MAP_FILTER_RADIUS
)
original_points_gammaO = gammaO.grid.points.copy()
grid_gammaO = gammaO.grid.clean(remove_unused_points=True).compute_cell_sizes(length=False, area=True, volume=False)
v_gammaO = grid_gammaO.points
f_gammaO = grid_gammaO.cells_dict[list(grid_gammaO.cells_dict)[0]]
population_gammaO = grid_gammaO["Area"]

S_gammaO = pySDEM.SphericalDensityEqualizingMap(v_gammaO, f_gammaO, population_gammaO)
R_gammaO, _ = pySDEM.optimal_rotation(v_gammaO, S_gammaO)
S_gammaO = S_gammaO @ R_gammaO
x_gammaO, y_gammaO, z_gammaO = S_gammaO[:, 0], S_gammaO[:, 1], S_gammaO[:, 2]
r_gammaO, lat_gammaO, lon_gammaO = pySDEM.cart_to_lat_lon(x_gammaO, y_gammaO, z_gammaO)
normals_O = kdpf.get_normals(gammaO_from_ansys)

# ---- GammaI (interior/excitation surface): in-vacuo modal displacement basis ----
gammaI_skin = kdpf.get_skin_mesh_from_ns(INTERNAL_NS, modal_model)
normals_I = kdpf.get_normals(gammaI_skin)
modal_tfreq = kdpf.get_tfreq(modal_model)
# modal_tfreq.data = modal_tfreq.data[modal_tfreq.data>0]   #------------SCOPE TO POSITIVE NATURAL FREQUENCIES
nd_fc = kdpf.get_normal_displacements(modal_model, gammaI_skin, modal_tfreq, normals_I)
if len(nd_fc) == 0:
    raise ValueError("Modal analysis returned no mode-shape displacements; extract modes first.")
gammaI_points = np.array(gammaI_skin.nodes.coordinates_field.data)


# ---> NEW: Prepend a constant pressure field (1.0 Pa) to the modal basis
mode_cols = [np.array(nd_fc[i].data) for i in range(len(nd_fc))]
constant_col = np.ones(len(gammaI_points))

# Stack the constant column first, followed by the structural modes
M = np.column_stack([constant_col] + mode_cols)
n_modes = M.shape[1]
# <---

# Nodal areas for the surface-L2 inner product (weighted POD)
if AREA_WEIGHTED_SVD:
    areas_I = np.array(kdpf.get_areas(gammaI_skin, location="Nodal").data)
    w = np.maximum(areas_I, 1e-12 * np.max(areas_I))
else:
    w = np.ones(M.shape[0])
sqrt_w = np.sqrt(w)

modal_model.metadata.release_streams()

# Weighted SVD: A = W^{1/2} M = U S V^T ; basis phi = W^{-1/2} U_k is W-orthonormal
# and spans the same space as the modal normal displacements, ranked by energy.
Aw = M * sqrt_w[:, np.newaxis]
U, Svals, _ = np.linalg.svd(Aw, full_matrices=False)

tol = SVD_REL_TOL * Svals[0] if Svals.size else 0.0
keep_mask = Svals > tol
if N_BASIS_MAX is not None:
    kept_idx = np.where(keep_mask)[0]
    if len(kept_idx) > N_BASIS_MAX:
        keep_mask[kept_idx[N_BASIS_MAX:]] = False
n_basis = int(keep_mask.sum())
if n_basis == 0:
    raise ValueError("No basis vectors retained after SVD truncation; check modes / SVD_REL_TOL.")

singular_values = Svals[keep_mask]
basis = U[:, keep_mask] / sqrt_w[:, np.newaxis]   # (n_GammaI_nodes, n_basis), real or complex
print(f"In-vacuo modal basis (with constant shift): {n_modes} inputs -> {n_basis} orthogonal basis vectors "
      f"(area_weighted={AREA_WEIGHTED_SVD})")

basis_names = [f"IVMB_{k}" for k in range(n_basis)]
export_named_fields(basis_names, np.real(basis), np.imag(basis), gammaI_points, pressure_export_folder)
allfiles = basis_names
print(f"Exported {n_basis} basis pressure files to {pressure_export_folder}")


# ============================================================================
# Import basis to Workbench and solve one MSUP harmonic per basis vector.
# ============================================================================
setup_external_data(workbench, systemName, pressure_export_folder, allfiles,
                    StartImportAtLine=StartImportAtLine, DelimiterIs=DelimiterIs,
                    DelimiterStringIs=DelimiterStringIs, LengthUnit=LengthUnit,
                    PressureUnit=PressureUnit)

# Set the solve core count once; it persists across all harmonic solves
set_cores_command = f"""
wbAnalysisName = "TARGET: HansenAutoImporter"
for item in ExtAPI.DataModel.AnalysisList:
    if item.SystemCaption == wbAnalysisName:
        analysis = item
analysis.SolveConfiguration.SolveProcessSettings.MaxNumberOfCores = {nCores}
"""
mechanical.run_python_script(set_cores_command)
mechanical.wait_till_mechanical_is_ready()

vn_list = []
tfreq = None
for fileid, filename in enumerate(allfiles, 1):
    vn, gammaO_ret, tfreq = solve_model(
        mechanical, filename, fileid, INTERNAL_NS, EXTERNAL_NS, model_folder,
        gammaO_from_ansys, normals_O, tfreq
    )
    if mapping_workflow_external is not None:
        mapping_workflow_external.connect('source', vn)
        vn = mapping_workflow_external.get_output('target', output_type="fields_container")
        
        #----------------------------ATTEMPT AT ENFORCING ZERO VN WHERE THERE IS NO MESH UNDER MAPPING
        vn = enforce_zero_outside_radius(
            mapped_fc=vn, 
            source_mesh=gammaO_from_ansys, 
            target_mesh=gammaO, 
            filter_radius=SHRINK_WRAP_MAP_FILTER_RADIUS
        )
    
    vn_list.append(vn)
    print(f"Solved for {filename}")


print("Calculating STM")
N_frequencies = len(tfreq.data)
N_points = len(lat_gammaO)

# Nearest-neighbour map from cleaned/SDEM node order back to the original skin node order
_, point_mapping_gammaO = cKDTree(original_points_gammaO).query(v_gammaO)
n_coeffs_I = n_basis
n_coeffs_O = (lmax_O + 1) ** 2
G = pysh.expand.LSQ_G(lat_gammaO, lon_gammaO, lmax_O)


# Construct large B matrix: (N_points, n_coeffs_I * N_frequencies)
B_large = np.zeros((N_points, n_coeffs_I * N_frequencies), dtype=complex)
for fileid in range(n_coeffs_I):
    vn = vn_list[fileid]
    # Fetch all real/imag velocity fields once, then reorder to SDEM node order
    field_data = np.array([vn[i].data for i in range(2 * N_frequencies)])
    block = field_data[0::2][:, point_mapping_gammaO] + 1j * field_data[1::2][:, point_mapping_gammaO]
    B_large[:, fileid * N_frequencies:(fileid + 1) * N_frequencies] = block.T

print("B_large constructed --> LSQ Solve started")
X_large, residuals_large, _, _ = np.linalg.lstsq(G, B_large, rcond=None)
print("Done --> Unpacking")

# Unpack X_large into STM (columns are ordered basisid-major, freqid-minor)
STM = X_large.T.reshape(n_coeffs_I, N_frequencies, n_coeffs_O).transpose(0, 2, 1)

# Absolute/relative residual errors (computed directly; lstsq residuals are empty when
# G is rank-deficient, which is typical for the lmax_O fit).
residual = G @ X_large - B_large
abs_error = np.linalg.norm(residual, axis=0).reshape(n_coeffs_I, N_frequencies)
b_norms = np.linalg.norm(B_large, axis=0).reshape(n_coeffs_I, N_frequencies)
rel_error = np.divide(abs_error, b_norms, out=np.zeros_like(abs_error), where=b_norms > 0)

print("Done")


mesh_data = {
    'INTERNAL': {
        'InternalGrid': gammaI_skin.grid,
        'SDEM_Coordinates': None,  # <--- ADDED FOR SCHEMA PARITY
        'mesh_metadata': {'nnodes': gammaI_skin.grid.n_points, 'nelements': gammaI_skin.grid.n_cells, 'areas': w}
    },
    'EXTERNAL': {
        'ExternalGrid': grid_gammaO,
        'SDEM_Coordinates': S_gammaO,
        'mesh_metadata': {'nnodes': len(gammaO.grid.points), 'nelements': gammaO.grid.n_cells, 'areas': grid_gammaO["Area"]}
    },
    'FULL_MESH': {
        'FullMesh': modal_model.metadata.meshed_region.grid,
        'mesh_metadata': {'nnodes': modal_model.metadata.meshed_region.grid.n_points,
                          'nelements': modal_model.metadata.meshed_region.grid.n_cells,
                          'areas': np.zeros([modal_model.metadata.meshed_region.grid.n_cells])}
    }
}

metadata = {
    'user_settings': {
        'model_folder': model_folder, 'modal_folder': modal_folder,
        'INTERNAL_NS': INTERNAL_NS, 'EXTERNAL_NS': EXTERNAL_NS,
        'lmax_O': lmax_O, 'nCores': nCores,
        'workbench_server_port': workbench_server_port, 'workbench_server_ip': workbench_server_ip,
        'solution_method': 'MSUP', 'input_basis': 'in_vacuo_modal_svd',
        'area_weighted_svd': AREA_WEIGHTED_SVD, 'svd_rel_tol': SVD_REL_TOL,
        'n_modes': n_modes, 'n_basis': n_basis
    },
    'pressure_file_settings': {
        'systemName': systemName, 'DataExtension': DataExtension, 'DelimiterIs': DelimiterIs,
        'DelimiterStringIs': DelimiterStringIs, 'StartImportAtLine': StartImportAtLine,
        'LengthUnit': LengthUnit, 'PressureUnit': PressureUnit, 'pressure_export_folder': pressure_export_folder
    },
    'computation_parameters': {
        'n_coeffs_I': n_coeffs_I,
        'n_coeffs_O': n_coeffs_O,
        'n_basis': n_basis,
        'n_modes': n_modes,
        'n_points_internal': len(gammaI_points),
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
    'export_files': {'harmonic_files': allfiles, 'n_files_exported': len(allfiles), 'file_pattern': 'IVMB_k.csv'},
    'point_mappings': {'point_mapping': np.array(point_mapping_gammaO), 'n_original_points': len(original_points_gammaO),
                       'n_cleaned_points': len(v_gammaO)},
    'error_data': {
        'abs_error': abs_error, 'rel_error': rel_error}
}

results_data['input_basis'] = {
    'type': 'in_vacuo_modal_svd',
    'labels': basis_names,                          
    'n_coeffs_I': n_coeffs_I,
    'basis_vectors': basis,
    'gammaI_points': gammaI_points,
    'n_basis': n_basis,
    'n_modes': n_modes,
    'area_weighted': AREA_WEIGHTED_SVD
}

# Package results
pySTM.package_stm_results(STM=STM, mesh_data=mesh_data, metadata=metadata, results_data=results_data,
                          output_file=STM_NAME + ".h5")
