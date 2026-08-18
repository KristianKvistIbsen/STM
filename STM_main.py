import os
import json
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

# Workbench server port (and IPv4, if not on localhost 127.0.0.1)
# ---------------------------------------------------------------------- #
# In Workbench:                                                          #
#   File -> Scripting -> Open command window                             #
#       In command window: StartServer()                                 #
#           Returned number is inserted below as workbench_server_port   #
# ---------------------------------------------------------------------- #
workbench_server_port = 37021 # StartServer() to retrieve port
workbench_server_ip = None

# ANSYS settings
model_folder            = None  # Defaults to dp0/MECH of system if None
pressure_export_folder  = None  # Defaults to model_folder if None
nCores                  = 8
# Pressure File Import Settings
systemName          = "SYS" #Name of the system where data is imported
DataExtension       = "csv" # CSV format
DelimiterIs         = "Comma" # Comma seperated
DelimiterStringIs   = "," # Comma seperated
StartImportAtLine   = 2 # Assumes there is a header at line 1
LengthUnit          = "m" # Must be m! Implement unit conversion if not m
PressureUnit        = "Pa" # Must be Pa! Implement unit conversion if not Pa

# STM settings
STM_NAME    = "motor" # Name of outputted STM file
INTERNAL_NS = 'GI' # Named selection of inner (pressure loaded) surface
EXTERNAL_NS = 'GE' # Named selection of outer (sound radiating) surface
lmax_O      = 80   # Highest degree of SH used for decomposing outer surface

# .stl for representing inner and outer surfaces, if they are not water tight
SHRINK_WRAP_STL_INTERNAL = None
SHRINK_WRAP_STL_EXTERNAL = r"N:\PhD\motor_noise\shrinkwrap.stl"
SHRINK_WRAP_MAP_FILTER_RADIUS = 0.005 #Mapping radius between surface and stl

# Sphesrical harmonic load definition
# (Uses SH functions as pressure basis on internal surface, classical STM)
SH_basis = False # Toggles this excitation type
lmax_I   = 1    # Highest degre of SH used for excitation

# SVD Basis
# (Takes in .csv file of pressure field on inner surface and adds it to the
#  SH basis. Then performs SVD to retrieve an augmented SH basis)
SVD_basis        = False
lmax_I           = 1
pressure_for_svd = r"N:\PhD\STM\TP\simple_flow_total_pressure.csv" 

# Custom basis
# (Entirely custom basis. Must be made explicitly on mesh of analyzed geometry.
#  Find example code for producing such a basis in pySTM_utils)
CUSTOM_basis = True
custom_basis_csv = r"C:/01_gitrepos/STM/stator_harmonic_loads.csv"

# =============================================================================
# CONFIG VALIDATION
# =============================================================================
if not (SH_basis+SVD_basis+CUSTOM_basis) == 1:
    raise Exception("Please only define one excitation type")
elif not PressureUnit=="Pa":
    raise Exception("Pressure unit must be Pa (Son, In my house we use MKS)")
elif not LengthUnit=="m":
    raise Exception("Length unit must be m (Son, In my house we use MKS)")    
    

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
# MESH EXTRACTION (IRONPYTHON -> JSON)
# =============================================================================
print("Extracting mesh data from Mechanical memory...")
extraction_script = f"""
import json
mesh_data = ExtAPI.DataModel.MeshDataByName("Global")

# Extract Nodes
mesh_unit = mesh_data.Unit.lower()
scale = 1.0

# Determine conversion factor to meters
if mesh_unit == "mm":
    scale = 0.001
elif mesh_unit == "cm":
    scale = 0.01
elif mesh_unit == "in":
    scale = 0.0254

nodes = []
for node in mesh_data.Nodes:
    nodes.append([node.Id, node.X * scale, node.Y * scale, node.Z * scale])

# Extract Elements
elements = []
for elem in mesh_data.Elements:
    type_str = elem.Type.ToString().lower()
    if "shell" in type_str or "tri" in type_str or "quad" in type_str:
        shape = "shell"
    elif "beam" in type_str or "link" in type_str or "line" in type_str:
        shape = "beam"
    else:
        shape = "solid"
    elements.append([elem.Id, shape, [int(i) for i in elem.NodeIds]])

# Extract Named Selections
named_selections = {{}}
model = ExtAPI.DataModel.Project.Model
if model.NamedSelections is not None:
    for ns in model.NamedSelections.Children:
        if ns.Name not in ["{INTERNAL_NS}", "{EXTERNAL_NS}"]:
            continue
            
        loc = ns.Location
        sel_type = loc.SelectionType.ToString()  
        node_ids = []
        element_ids = []
        
        if sel_type == "MeshNodes":
            node_ids = [int(i) for i in loc.Ids]
        elif sel_type == "MeshElements":
            element_ids = [int(i) for i in loc.Ids]
        else:
            # Geometry-based: expand each entity to its mesh nodes/elements
            for gid in loc.Ids:
                region = mesh_data.MeshRegionById(gid)
                if region is not None:
                    node_ids.extend([int(i) for i in region.NodeIds])
                    element_ids.extend([int(i) for i in region.ElementIds])
                    
        named_selections[ns.Name] = {{
            "node_ids": sorted(list(set(node_ids))),
            "element_ids": sorted(list(set(element_ids))),
        }}

output = {{
    "num_nodes": mesh_data.NodeCount,
    "num_elements": mesh_data.ElementCount,
    "nodes": nodes,
    "elements": elements,
    "named_selections": named_selections,
}}
json.dumps(output)
"""

raw_json = mechanical.run_python_script(extraction_script)
data = json.loads(raw_json)
print(f"Extracted {data['num_nodes']} nodes and {data['num_elements']} elements.")

# =============================================================================
# BUILD GLOBAL DPF MESH
# =============================================================================
print("Constructing DPF MeshedRegion...")
mesh = dpf.MeshedRegion(
    num_nodes=data["num_nodes"],
    num_elements=data["num_elements"]
)

id_to_index = {}
for index, node_data in enumerate(data["nodes"]):
    node_id, x, y, z = node_data
    mesh.nodes.add_node(node_id, [x, y, z])
    id_to_index[node_id] = index

for elem_data in data["elements"]:
    elem_id, shape, node_ids = elem_data
    conn = [id_to_index[nid] for nid in node_ids if nid in id_to_index]
    
    if not conn:
        continue
    if shape == "solid":
        mesh.elements.add_solid_element(elem_id, conn)
    elif shape == "shell":
        mesh.elements.add_shell_element(elem_id, conn)
    elif shape == "beam":
        mesh.elements.add_beam_element(elem_id, conn)

mesh.unit = "m" 

# =============================================================================
# EXTRACT SKIN MESHES (INTERNAL & EXTERNAL)
# =============================================================================
def extract_skin_for_ns(ns_name, global_mesh, extraction_data):
    node_ids = extraction_data["named_selections"][ns_name]["node_ids"]
    nodal_scoping = dpf.Scoping(ids=node_ids, location=dpf.locations.nodal)
    
    skin_op = dpf.operators.mesh.skin()
    skin_op.inputs.mesh.connect(global_mesh)
    skin_op.inputs.mesh_scoping.connect(nodal_scoping)
    return skin_op.outputs.mesh()

gammaO_from_ansys = extract_skin_for_ns(EXTERNAL_NS, mesh, data)
gammaI_from_ansys = extract_skin_for_ns(INTERNAL_NS, mesh, data)

# =============================================================================
# SDEM MAPPING & PREPROCESSING
# =============================================================================
# Process External
gammaO, mapping_workflow_external = pySTM.check_genus_zero_and_map_if_needed(
    gammaO_from_ansys, "EXTERNAL", SHRINK_WRAP_STL_EXTERNAL
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

# Process Internal only if SH_basis or SVD_basis
if SH_basis or SVD_basis:
    gammaI, mapping_workflow_internal = pySTM.check_genus_zero_and_map_if_needed(
        gammaI_from_ansys, "INTERNAL", SHRINK_WRAP_STL_INTERNAL
    )
    original_points_gammaI = gammaI.grid.points.copy()
    grid_gammaI = gammaI.grid.clean(remove_unused_points=True).compute_cell_sizes(length=False, area=True, volume=False)
    v_gammaI = grid_gammaI.points
    f_gammaI = grid_gammaI.cells_dict[list(grid_gammaI.cells_dict)[0]]
    population_gammaI = grid_gammaI["Area"]

    S_gammaI = pySDEM.SphericalDensityEqualizingMap(v_gammaI, f_gammaI, population_gammaI)
    R_gammaI, _ = pySDEM.optimal_rotation(v_gammaI, S_gammaI)
    S_gammaI = S_gammaI @ R_gammaI
    x_gammaI, y_gammaI, z_gammaI = S_gammaI[:, 0], S_gammaI[:, 1], S_gammaI[:, 2]
    r_gammaI, lat_gammaI, lon_gammaI = pySDEM.cart_to_lat_lon(x_gammaI, y_gammaI, z_gammaI)

if CUSTOM_basis:
    v_gammaI = np.array(gammaI_from_ansys.nodes.coordinates_field.data)
    S_gammaI = v_gammaI
    grid_gammaI = gammaI_from_ansys.grid.compute_cell_sizes()
    gammaI = gammaI_from_ansys
    
    
# =============================================================================
# BASIS GENERATION
# =============================================================================
normals_O = kdpf.get_normals(gammaO_from_ansys)
tfreq = None  
vn_list = []

if SVD_basis:
    print("\nUsing Custom SVD Augmented Basis...")
    full_basis_array, n_coeffs_I = pySTM.generate_svd_augmented_basis(
        lmax_I, S_gammaI, lat_gammaI, lon_gammaI, v_gammaI, pressure_for_svd
    )
    n_harmonics = n_coeffs_I
    basis_labels = [f"SVD_Basis_{i}" for i in range(n_coeffs_I)]
    basis_type = 'svd_augmented'
    file_pattern = "SVD_Basis_{i}"
    
elif SH_basis:
    print("\nUsing Standard Spherical Harmonics Basis (Including Monopole)...")
    sh_array, n_harmonics = pySTM.generate_spherical_harmonics(lmax_I, S_gammaI, lat_gammaI, lon_gammaI)
    n_coeffs_I = (lmax_I + 1) ** 2
    monopole_array = np.ones((sh_array.shape[0], 1), dtype=np.complex128)
    full_basis_array = np.hstack((monopole_array, sh_array))
    
    basis_labels = [f"Y_{l}_{m}" for l in range(lmax_I + 1) for m in range(-l, l + 1)]
    basis_type = 'spherical_harmonics'
    file_pattern = "Y_{l}_{m}"
    
elif CUSTOM_basis:
    print(f"\nUsing Custom Basis from {custom_basis_csv}...")
    full_basis_array_with_xyz = np.genfromtxt(custom_basis_csv, delimiter=',',skip_header=1)
    full_basis_array = full_basis_array_with_xyz[:,3:]
    n_coeffs_I = np.shape(full_basis_array)[1]
    n_harmonics = n_coeffs_I
    
    basis_labels = [f"Custom_Basis_{i}" for i in range(n_coeffs_I)]
    basis_type = f'Custom Basis, {custom_basis_csv}'
    file_pattern = "Custom_Basis_{i}"
    
    # print(f"\nUsing Custom Basis from {custom_basis_csv}...")
    # full_basis_array_with_xyz = np.genfromtxt(custom_basis_csv, delimiter=',', skip_header=1)
    
    # csv_coords = full_basis_array_with_xyz[:, :3] 
    # raw_basis_array = full_basis_array_with_xyz[:, 3:]
    
    # tree = cKDTree(csv_coords)
    # distances, mapped_indices = tree.query(v_gammaI)
    
    # max_dist = np.max(distances)
    # if max_dist > 1e-4:
    #     print(f"WARNING: Max mapping distance is {max_dist:.6f}. Check if CSV coordinates are in mm while mesh is in m.")
        
    # full_basis_array = raw_basis_array[mapped_indices]
    
    # n_coeffs_I = np.shape(full_basis_array)[1]
    # n_harmonics = n_coeffs_I
    
    # basis_labels = [f"Custom_Basis_{i}" for i in range(n_coeffs_I)]
    # basis_type = f'Custom Basis, {custom_basis_csv}'
    # file_pattern = "Custom_Basis_{i}"
    
else:
    raise Exception("Excitation type not defined")


allfiles = pySTM.export_named_fields(
    basis_labels, np.real(full_basis_array), np.imag(full_basis_array), 
    v_gammaI, pressure_export_folder
)
print(f"Basis exported: {len(allfiles)} files to {pressure_export_folder}")

# =============================================================================
# IMPORT & SOLVE LOOP
# =============================================================================
pySTM.setup_external_data(workbench, systemName, pressure_export_folder, allfiles,
                    StartImportAtLine=StartImportAtLine, DelimiterIs=DelimiterIs,
                    DelimiterStringIs=DelimiterStringIs, LengthUnit=LengthUnit,
                    PressureUnit=PressureUnit)

set_cores_command = f"""
wbAnalysisName = "TARGET: HansenAutoImporter"
for item in ExtAPI.DataModel.AnalysisList:
    if item.SystemCaption == wbAnalysisName:
        analysis = item
analysis.SolveConfiguration.SolveProcessSettings.MaxNumberOfCores = {nCores}
"""
mechanical.run_python_script(set_cores_command)
mechanical.wait_till_mechanical_is_ready()

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

_, point_mapping_gammaO = cKDTree(original_points_gammaO).query(v_gammaO)
n_coeffs_O = (lmax_O + 1) ** 2
G = pysh.expand.LSQ_G(lat_gammaO, lon_gammaO, lmax_O)

B_large = np.zeros((N_points, n_coeffs_I * N_frequencies), dtype=complex)
for fileid in range(n_coeffs_I):
    vn = vn_list[fileid]
    field_data = np.array([vn[i].data for i in range(2 * N_frequencies)])
    block = field_data[0::2][:, point_mapping_gammaO] + 1j * field_data[1::2][:, point_mapping_gammaO]
    B_large[:, fileid * N_frequencies:(fileid + 1) * N_frequencies] = block.T

print("B_large constructed --> LSQ Solve started")
X_large, residuals_large, _, _ = np.linalg.lstsq(G, B_large, rcond=None)
print("Done --> Unpacking")

STM = X_large.T.reshape(n_coeffs_I, N_frequencies, n_coeffs_O).transpose(0, 2, 1)

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
        'FullMesh': mesh.grid,
        'mesh_metadata': {'nnodes': mesh.grid.n_points,
                          'nelements': mesh.grid.n_cells, 'areas': np.zeros([mesh.grid.n_cells])}
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
    'export_files': {'harmonic_files': allfiles, 'n_files_exported': len(allfiles), 'file_pattern': file_pattern},
    'point_mappings': {'point_mapping': np.array(point_mapping_gammaO), 'n_original_points': len(original_points_gammaO),
                       'n_cleaned_points': len(v_gammaO)},
    'error_data': {
        'abs_error': abs_error, 'rel_error': rel_error},
    'input_basis': {
        'type': basis_type,
        'labels': basis_labels,                            
        'n_coeffs_I': n_coeffs_I,
        'basis_vectors': full_basis_array,
        'gammaI_points': v_gammaI,
        'lmax_I': lmax_I
    }
}

pySTM.package_stm_results(STM=STM, mesh_data=mesh_data, metadata=metadata, results_data=results_data,
                   output_file=STM_NAME+".h5")
