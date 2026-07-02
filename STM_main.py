import kdpf
import pySDEM
import pySTM
import numpy as np
from scipy.spatial import cKDTree
import pyshtools as pysh
from ansys.dpf import core as dpf
import os
import trimesh
from ansys.workbench.core import connect_workbench
from ansys.mechanical.core import connect_to_mechanical

model_folder = None  # Defaults to dp0/MECH of system if None
pressure_export_folder = None  # Defaults to model_folder if None
INTERNAL_NS = 'GI'
EXTERNAL_NS = 'GE'

STM_NAME = "STM"

nCores = 6
lmax_I = 1
lmax_O = 60
workbench_server_port = 1045 # StartServer() to retrieve port
workbench_server_ip = None

SHRINK_WRAP_STL_INTERNAL = None
SHRINK_WRAP_STL_EXTERNAL = None
SHRINK_WRAP_MAP_FILTER_RADIUS = 0.005

# --- Mode-superposition (MSUP) settings ---------------------------------------
# The Modal + MSUP Harmonic systems are set up manually in Workbench (see module
# docstring). Nothing to configure here beyond pointing systemName at the harmonic.

# Pressure File Import Settings
# systemName = the HARMONIC (Mode-Superposition) system in Workbench. Mechanical connects
# here and imported pressures target it; set it to whichever of "SYS" / "SYS 1" is your
# Harmonic Response system. The Modal analysis is auto-detected in the shared model.
systemName = "SYS"
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


# The Workbench system name does not necessarily match its dp0 solver-files folder
# (e.g. system "SYS" can solve into dp0\SYS-1\MECH). Ask the harmonic analysis for its
# actual working directory instead of building the path from systemName.
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



def check_genus_zero_and_map_if_needed(mesh, mesh_name, stl_path=None):
    """Check if mesh is genus-0 and map to STL if not."""
    print(f"\nChecking {mesh_name} mesh topology...")

    mesh_grid = mesh.grid
    v = mesh_grid.points
    f = mesh_grid.cells_dict[list(mesh_grid.cells_dict.keys())[0]]

    v_used = np.unique(f)
    v_unused = np.setdiff1d(np.arange(len(v)), v_used)
    v_clean = v[v_used]

    v_map = np.zeros(len(v_used) + len(v_unused), dtype=int)
    v_map[v_used] = np.arange(len(v_used))
    f_clean = v_map[f]

    euler_characteristic = len(v_clean) - 3*len(f_clean)/2 + len(f_clean)

    print(f"{mesh_name} mesh - Vertices: {len(v_clean)}, Faces: {len(f_clean)}")
    print(f"Euler characteristic: {euler_characteristic} (should be 2 for genus-0)")

    is_genus_zero = int(abs(euler_characteristic - 2)) == 0

    if is_genus_zero:
        print(f"{mesh_name} mesh is genus-0. No mapping needed.")
        return mesh, None
    else:
        print(f"{mesh_name} mesh is NOT genus-0!")

        if stl_path is not None:
            print(f"Attempting projection to shrink-wrapped STL: {stl_path}")

            stl_mesh, _ = stl_to_dpf_mesh(stl_path)

            stl_grid = stl_mesh.grid
            stl_v = stl_grid.points
            stl_f = stl_grid.cells_dict[list(stl_grid.cells_dict.keys())[0]]
            stl_euler = len(stl_v) - 3*len(stl_f)/2 + len(stl_f)

            if int(abs(stl_euler - 2)) != 0:
                raise ValueError(f"STL mesh is also not genus-0! Euler characteristic: {stl_euler}")

            print("STL mesh is genus-0. Proceeding with mapping...")

            op_mapping_workflow = dpf.operators.mapping.prepare_mapping_workflow()
            op_mapping_workflow.inputs.input_support.connect(mesh)
            op_mapping_workflow.inputs.output_support.connect(stl_mesh)
            op_mapping_workflow.inputs.filter_radius.connect(SHRINK_WRAP_MAP_FILTER_RADIUS)
            mapping_workflow = op_mapping_workflow.outputs.mapping_workflow()
            mapping_workflow.progress_bar=False
            return stl_mesh, mapping_workflow
        else:
            raise ValueError(f'{mesh_name} mesh is not genus-0, and no STL file provided for shrink-wrapping.')


def stl_to_dpf_mesh(stl_path):
    mesh = trimesh.load(stl_path, file_type='stl')
    connectivity = mesh.faces
    coordinates = mesh.vertices
    meshed_region = dpf.MeshedRegion(
        num_nodes=coordinates.shape[0],
        num_elements=connectivity.shape[0]
    )
    print("Warning: scaling factor of 1000 used to convert from mm to m")
    scaled_coordinates = coordinates / 1000.0
    for i in range(len(scaled_coordinates)):
        meshed_region.nodes.add_node(i, scaled_coordinates[i])

    faces_list = connectivity.tolist()
    for i, face in enumerate(faces_list):
        meshed_region.elements.add_element(i, "shell", face)
    return meshed_region, mesh.area_faces


def generate_spherical_harmonics(lmax, points, lat, lon):
    """Generate spherical harmonics for given lmax and points."""
    n_harmonics = (lmax + 1) ** 2 - 1  # l=0 monopole is handled separately
    n_points = len(points)
    sh_array = np.zeros((n_points, n_harmonics), dtype=np.complex128)
    harm_idx = 0

    for l in range(1,lmax + 1):
        for m in range(-l, l + 1):
            coeffs = np.zeros((2, l + 1, l + 1), dtype=np.complex128)
            if m >= 0:
                coeffs[0, l, m] = 1.0 + 0j
            else:
                coeffs[1, l, abs(m)] = 1.0 + 0j
            sh_coeffs = pysh.SHCoeffs.from_array(coeffs)
            sh_array[:, harm_idx] = pysh.expand.MakeGridPointC(sh_coeffs.coeffs, lat, lon)
            harm_idx += 1
    return sh_array, n_harmonics

def export_spherical_harmonics(sh_array, points, export_folder, lmax):
    """Export spherical harmonics to CSV files."""
    allfiles = []
    harm_idx = 0
    for l in range(1,lmax + 1):
        for m in range(-l, l + 1):
            filename = f"Y_{l}_{m}.csv"
            export_path = os.path.join(export_folder, filename)
            sh_values = sh_array[:, harm_idx]
            data = np.column_stack((points[:, 0], points[:, 1], points[:, 2], np.real(sh_values), np.imag(sh_values)))
            header = ['x', 'y', 'z', 'real', 'imag']
            np.savetxt(export_path, data, delimiter=',', header=','.join(header), comments='', fmt='%.16e')
            allfiles.append(f"Y_{l}_{m}")
            harm_idx += 1
    return allfiles

def setup_external_data(workbench, system_name, export_folder, files):
    """Set up external data in Workbench."""
    init_commands = f"""
templateExternalData = GetTemplate(TemplateName="External Data")
system1 = templateExternalData.CreateSystem()
system1.DisplayText = "HansenAutoImporter"
setup1 = system1.GetContainer(ComponentName="Setup")
system2 = GetSystem(Name="{systemName}")
system2.DisplayText = "TARGET: HansenAutoImporter"

"""
    workbench.run_script_string(init_commands)

    external_commands = "setup1 = system1.GetContainer(ComponentName=\"Setup\")\n"
    for i, file in enumerate(files):
        filepath = os.path.join(export_folder, file).replace('\\', '\\\\') + ".csv"
        external_commands += f"""
externalLoadFileData{i} = setup1.AddDataFile(FilePath=r"{filepath}")
        """
        if i == 0:
            external_commands += f"""
externalLoadFileData{i}.SetAsMaster(Master=True)
externalLoadFileDataPropertyObj = externalLoadFileData{i}.GetDataProperty()
externalLoadFileData{i}.SetStartImportAtLine(FileDataProperty=externalLoadFileDataPropertyObj, LineNumber={StartImportAtLine})
externalLoadFileData{i}.SetDelimiterType(FileDataProperty=externalLoadFileDataPropertyObj, Delimiter="{DelimiterIs}", DelimiterString="{DelimiterStringIs}")
externalLoadFileDataPropertyObj.SetLengthUnit(Unit="{LengthUnit}")
externalLoadColumnData1 = externalLoadFileDataPropertyObj.GetColumnData(Name="ExternalLoadColumnData")
externalLoadFileDataPropertyObj.SetColumnDataType(ColumnData=externalLoadColumnData1, DataType="X Coordinate")
externalLoadColumnData2 = externalLoadFileDataPropertyObj.GetColumnData(Name="ExternalLoadColumnData 1")
externalLoadFileDataPropertyObj.SetColumnDataType(ColumnData=externalLoadColumnData2, DataType="Y Coordinate")
externalLoadColumnData3 = externalLoadFileDataPropertyObj.GetColumnData(Name="ExternalLoadColumnData 2")
externalLoadFileDataPropertyObj.SetColumnDataType(ColumnData=externalLoadColumnData3, DataType="Z Coordinate")
externalLoadColumnData4 = externalLoadFileDataPropertyObj.GetColumnData(Name="ExternalLoadColumnData 3")
externalLoadFileDataPropertyObj.SetColumnDataType(ColumnData=externalLoadColumnData4, DataType="Pressure")
externalLoadColumnData5 = externalLoadFileDataPropertyObj.GetColumnData(Name="ExternalLoadColumnData 4")
externalLoadFileDataPropertyObj.SetColumnDataType(ColumnData=externalLoadColumnData5, DataType="Pressure")
externalLoadColumnData4.Unit = "{PressureUnit}"
externalLoadColumnData4.Identifier = "{file}_real"
externalLoadColumnData5.Unit = "{PressureUnit}"
externalLoadColumnData5.Identifier = "{file}_imag"
"""
        else:
            external_commands += f"""
externalLoadFileDataPropertyObj = externalLoadFileData{i}.GetDataProperty()
externalLoadFileData{i}.SetStartImportAtLine(FileDataProperty=externalLoadFileDataPropertyObj, LineNumber={StartImportAtLine})
externalLoadFileData{i}.SetDelimiterType(FileDataProperty=externalLoadFileDataPropertyObj, Delimiter="{DelimiterIs}", DelimiterString="{DelimiterStringIs}")
externalLoadFileDataPropertyObj.SetLengthUnit(Unit="{LengthUnit}")
externalLoadColumnData4 = externalLoadFileDataPropertyObj.GetColumnData(Name="ExternalLoadColumnData {5*i+3}")
externalLoadFileDataPropertyObj.SetColumnDataType(ColumnData=externalLoadColumnData4, DataType="Pressure")
externalLoadColumnData5 = externalLoadFileDataPropertyObj.GetColumnData(Name="ExternalLoadColumnData {5*i+4}")
externalLoadFileDataPropertyObj.SetColumnDataType(ColumnData=externalLoadColumnData5, DataType="Pressure")
externalLoadColumnData4.Unit = "{PressureUnit}"
externalLoadColumnData4.Identifier = "{file}_real"
externalLoadColumnData5.Unit = "{PressureUnit}"
externalLoadColumnData5.Identifier = "{file}_imag"
"""
    external_commands += f"""
system2 = GetSystem(Name="{system_name}")
setupComponent2 = system2.GetComponent(Name="Setup")
setupComponent1 = system1.GetComponent(Name="Setup")
setupComponent1.TransferData(TargetComponent=setupComponent2)
setupComponent1.Update(AllDependencies=True)
setupComponent2.Refresh()
setup2 = system2.GetContainer(ComponentName="Setup")
setup2.Edit()
"""
    workbench.run_script_string(external_commands)


def solve_model(mechanical, filename, fileid, internal_ns, external_ns, model_folder, gammaO, normals, tfreq):
    """Solve one harmonic load case (now a cheap MSUP solve) and extract normal velocity.

    The harmonic analysis is mode-superposition, so ``analysis.Solve`` reuses the
    already-solved modal basis; clearing the harmonic's generated data afterwards does
    not invalidate the upstream modal. Geometry-dependent quantities (skin mesh
    ``gammaO``, ``normals`` and the frequency support ``tfreq``) are precomputed once
    and reused across all harmonics.
    """
    map_commands = f"""
import re
wbAnalysisName = "TARGET: HansenAutoImporter"
for item in ExtAPI.DataModel.AnalysisList:
    if item.SystemCaption == wbAnalysisName:
        analysis = item
with Transaction():
    importedloadobjects = [child for child in analysis.Children if child.DataModelObjectCategory.ToString() == "ImportedLoadGroup"]
    usedimportedloadobj = importedloadobjects[-1]
    namedsel_internal = ExtAPI.DataModel.GetObjectsByName("{internal_ns}")[0]
    namedsel_external = ExtAPI.DataModel.GetObjectsByName("{external_ns}")[0]
    importedPres = usedimportedloadobj.AddImportedPressure()
    importedPres.Location = namedsel_internal
    importedPres.Name = "{filename}"
    table = importedPres.GetTableByName("")
    table[0][0] = "File{fileid}:" + "{filename}" + "_real"
    table[0][1] = "File{fileid}:" + "{filename}" + "_imag"
    table[0][2] = 1
    importedPres.ImportLoad()
"""
    solve_commands = """
analysis.Solve(True)
analysis.Solution.GetResults()
"""
    clean_commands = f"""
with Transaction():
    importedloadobjects = [child for child in analysis.Children if child.DataModelObjectCategory.ToString() == "ImportedLoadGroup"]
    usedimportedloadobj = importedloadobjects[-1]
    children = usedimportedloadobj.Children
    for child in children:
        if child.Name == '{filename}':
            id2del = child.ObjectId
            imported_pressure = DataModel.GetObjectById(id2del)
            imported_pressure.Suppressed = True
    analysis.ClearGeneratedData()
"""
    mechanical.run_python_script(map_commands)
    mechanical.wait_till_mechanical_is_ready()
    mechanical.run_python_script(solve_commands)
    mechanical.wait_till_mechanical_is_ready()

    model = dpf.Model(model_folder + r"\file.rst")
    vn = kdpf.get_normal_velocities(model, gammaO, tfreq, normals)

    model.metadata.release_streams()
    mechanical.run_python_script(clean_commands)
    mechanical.wait_till_mechanical_is_ready()
    return vn, gammaO, tfreq

# The monopole (Y_0_0) is the first load case. The modal analysis must already be solved
# manually in Workbench. The harmonic is located by analysis type in the shared model.
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


# ============================================================================
# Solve the monopole (Y_0_0) as the first MSUP harmonic load case. The Modal + MSUP
# Harmonic systems must already be set up AND the modal solved manually in Workbench;
# this script only solves the harmonic load cases.
# ============================================================================
mechanical.run_python_script(monopole_solve_command)
mechanical.wait_till_mechanical_is_ready()


# Load and preprocess mesh
model = dpf.Model(model_folder + r"\file.rst")

gammaO_from_ansys = kdpf.get_skin_mesh_from_ns(EXTERNAL_NS, model)
gammaO, mapping_workflow_external = check_genus_zero_and_map_if_needed(
    gammaO_from_ansys, "EXTERNAL", SHRINK_WRAP_STL_EXTERNAL
)
original_points_gammaO = gammaO.grid.points.copy()
grid_gammaO = gammaO.grid.clean(remove_unused_points=True).compute_cell_sizes(length=False, area=True, volume=False)
v_gammaO = grid_gammaO.points
f_gammaO = grid_gammaO.cells_dict[list(grid_gammaO.cells_dict)[0]]
population_gammaO = grid_gammaO["Area"]

# Compute SDEM for internal mesh
S_gammaO = pySDEM.SphericalDensityEqualizingMap(v_gammaO, f_gammaO, population_gammaO)
R_gammaO, _ = pySDEM.optimal_rotation(v_gammaO, S_gammaO)
S_gammaO = S_gammaO @ R_gammaO
x_gammaO, y_gammaO, z_gammaO = S_gammaO[:, 0], S_gammaO[:, 1], S_gammaO[:, 2]
r_gammaO, lat_gammaO, lon_gammaO = pySDEM.cart_to_lat_lon(x_gammaO, y_gammaO, z_gammaO)


gammaI_from_ansys = kdpf.get_skin_mesh_from_ns(INTERNAL_NS, model)
gammaI, mapping_workflow_internal = check_genus_zero_and_map_if_needed(
    gammaI_from_ansys, "INTERNAL", SHRINK_WRAP_STL_INTERNAL
)
original_points_gammaI = gammaI.grid.points.copy()
grid_gammaI = gammaI.grid.clean(remove_unused_points=True).compute_cell_sizes(length=False, area=True, volume=False)
v_gammaI = grid_gammaI.points
f_gammaI = grid_gammaI.cells_dict[list(grid_gammaI.cells_dict)[0]]
population_gammaI = grid_gammaI["Area"]

# Compute SDEM for internal mesh
S_gammaI = pySDEM.SphericalDensityEqualizingMap(v_gammaI, f_gammaI, population_gammaI)
R_gammaI, _ = pySDEM.optimal_rotation(v_gammaI, S_gammaI)
S_gammaI = S_gammaI @ R_gammaI
x_gammaI, y_gammaI, z_gammaI = S_gammaI[:, 0], S_gammaI[:, 1], S_gammaI[:, 2]
r_gammaI, lat_gammaI, lon_gammaI = pySDEM.cart_to_lat_lon(x_gammaI, y_gammaI, z_gammaI)


normals = kdpf.get_normals(gammaO_from_ansys)
tfreq = kdpf.get_tfreq(model)
vn = kdpf.get_normal_velocities(model, gammaO_from_ansys, tfreq, normals)
if mapping_workflow_external is not None:
    mapping_workflow_external.connect('source',vn)
    vn = mapping_workflow_external.get_output('target', output_type="fields_container")
    
    
    #----------------------------ATTEMPT AT ENFORCING ZERO VN WHERE THERE IS NO MESH UNDER MAPPING
    vn = enforce_zero_outside_radius(
        mapped_fc=vn, 
        source_mesh=gammaO_from_ansys, 
        target_mesh=gammaO, 
        filter_radius=SHRINK_WRAP_MAP_FILTER_RADIUS
    )
    
    
    
vn_list = []
vn_list.append(vn)
model.metadata.release_streams()


# Generate and export spherical harmonics
spherical_harmonics_array, n_harmonics = generate_spherical_harmonics(lmax_I, S_gammaI, lat_gammaI, lon_gammaI)
allfiles = export_spherical_harmonics(spherical_harmonics_array, v_gammaI, pressure_export_folder, lmax_I)
print(f"Spherical harmonics exported: {len(allfiles)} files to {pressure_export_folder}")

mechanical.run_python_script(monopole_suppress_command)
mechanical.wait_till_mechanical_is_ready()


# Import data to Workbench
setup_external_data(workbench, systemName, pressure_export_folder, allfiles)

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

# Solve each harmonic as a cheap MSUP load case (modal basis is reused, skin mesh /
# normals / tfreq reused across solves)
for fileid, filename in enumerate(allfiles, 1):
    vn, gammaO, tfreq = solve_model(mechanical, filename, fileid, INTERNAL_NS, EXTERNAL_NS, model_folder, gammaO_from_ansys, normals, tfreq)
    if mapping_workflow_external is not None:
        mapping_workflow_external.connect('source',vn)
        vn = mapping_workflow_external.get_output('target', output_type="fields_container")
    vn_list.append(vn)
    print(f"Solved for {filename}")


print("Calculating STM")
N_frequencies = len(tfreq.data)
N_points = len(lat_gammaO)

# Nearest-neighbour map from cleaned/SDEM node order back to the original skin node order
_, point_mapping_gammaO = cKDTree(original_points_gammaO).query(v_gammaO)
n_coeffs_I = (lmax_I + 1) ** 2
n_coeffs_O = (lmax_O + 1) ** 2
STM = np.zeros([n_coeffs_I, n_coeffs_O, len(tfreq.data)], dtype=np.complex128)
G = pysh.expand.LSQ_G(lat_gammaO,lon_gammaO,lmax_O)


# Construct large B matrix: (N_points, n_coeffs_I * N_frequencies)
B_large = np.zeros((N_points, n_coeffs_I * N_frequencies), dtype=complex)
for fileid in range(n_coeffs_I):
    vn = vn_list[fileid]
    # Fetch all real/imag velocity fields once, then reorder to SDEM node order
    field_data = np.array([vn[i].data for i in range(2 * N_frequencies)])
    block = field_data[0::2][:, point_mapping_gammaO] + 1j * field_data[1::2][:, point_mapping_gammaO]
    B_large[:, fileid * N_frequencies:(fileid + 1) * N_frequencies] = block.T

print("B_large constructed --> LSQ Solve started")
# Solve the least-squares problem once
X_large, residuals_large, _, _ = np.linalg.lstsq(G, B_large, rcond=None)
print("Done --> Unpacking")


# Unpack X_large into STM (columns are ordered fileid-major, freqid-minor)
STM = X_large.T.reshape(n_coeffs_I, N_frequencies, n_coeffs_O).transpose(0, 2, 1)

# Compute absolute and relative residual errors directly. np.linalg.lstsq only returns
# residuals when G is full-rank AND overdetermined; the lmax_O fit is typically
# rank-deficient, which previously left the error arrays silently at zero.
residual = G @ X_large - B_large
abs_error = np.linalg.norm(residual, axis=0).reshape(n_coeffs_I, N_frequencies)
b_norms = np.linalg.norm(B_large, axis=0).reshape(n_coeffs_I, N_frequencies)
rel_error = np.divide(abs_error, b_norms, out=np.zeros_like(abs_error), where=b_norms > 0)

print("Done")


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
        'input_basis': 'spherical_harmonics'  # <--- ADDED FLAG
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
    'labels': sh_labels,                            # <--- ADDED
    'n_coeffs_I': n_coeffs_I,
    'basis_vectors': full_basis_array,
    'gammaI_points': v_gammaI,
    'lmax_I': lmax_I
}
# Package results
pySTM.package_stm_results(STM=STM, mesh_data=mesh_data, metadata=metadata, results_data=results_data,
                   output_file=STM_NAME+".h5")
