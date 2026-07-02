"""Shared helpers for the STM generators (`STM_main.py`, `STM_inVacuoModalBasis_main.py`).

These are the side-effect-free building blocks (mesh handling, CSV export, Workbench
External Data setup, per-load-case MSUP solve). Keeping them here lets both generator
scripts import the same code without triggering each other's module-level Workbench
connection.
"""

import os
import numpy as np
import trimesh
import pyshtools as pysh
from ansys.dpf import core as dpf

import kdpf


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


def check_genus_zero_and_map_if_needed(mesh, mesh_name, stl_path=None, filter_radius=0.005):
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
            op_mapping_workflow.inputs.filter_radius.connect(filter_radius)
            mapping_workflow = op_mapping_workflow.outputs.mapping_workflow()
            mapping_workflow.progress_bar = False
            return stl_mesh, mapping_workflow
        else:
            raise ValueError(f'{mesh_name} mesh is not genus-0, and no STL file provided for shrink-wrapping.')


def generate_spherical_harmonics(lmax, points, lat, lon):
    """Generate spherical harmonics for given lmax and points."""
    n_harmonics = (lmax + 1) ** 2 - 1  # l=0 monopole is handled separately
    n_points = len(points)
    sh_array = np.zeros((n_points, n_harmonics), dtype=np.complex128)
    harm_idx = 0

    for l in range(1, lmax + 1):
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
    for l in range(1, lmax + 1):
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


def export_named_fields(names, values_real, values_imag, points, export_folder):
    """Export a set of named nodal fields (real+imag) to per-field CSV files.

    Mirrors the spherical-harmonic CSV layout (columns x, y, z, real, imag) so the same
    External Data import path can be reused. ``values_real`` / ``values_imag`` are
    (n_points, n_fields) arrays; ``names`` has length n_fields. Returns list(names).
    """
    values_real = np.asarray(values_real)
    values_imag = np.asarray(values_imag)
    for k, name in enumerate(names):
        export_path = os.path.join(export_folder, f"{name}.csv")
        data = np.column_stack((points[:, 0], points[:, 1], points[:, 2],
                                values_real[:, k], values_imag[:, k]))
        header = ['x', 'y', 'z', 'real', 'imag']
        np.savetxt(export_path, data, delimiter=',', header=','.join(header), comments='', fmt='%.16e')
    return list(names)


def setup_external_data(workbench, system_name, export_folder, files, *,
                        StartImportAtLine, DelimiterIs, DelimiterStringIs,
                        LengthUnit, PressureUnit, systemName=None):
    """Set up external data in Workbench and transfer it into the target system's Setup."""
    if systemName is None:
        systemName = system_name

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


def solve_model(mechanical, filename, fileid, internal_ns, external_ns, model_folder, gammaO, normals, tfreq=None):
    """Solve one imported-pressure harmonic load case (MSUP) and extract normal velocity.

    Geometry-dependent quantities (skin mesh ``gammaO``, ``normals``) are precomputed
    once and reused across all load cases. ``tfreq`` may be ``None`` on the first call,
    in which case the harmonic frequency support is read from the result file and
    returned for reuse on subsequent calls.
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
    if tfreq is None:
        tfreq = kdpf.get_tfreq(model)
    vn = kdpf.get_normal_velocities(model, gammaO, tfreq, normals)

    model.metadata.release_streams()
    mechanical.run_python_script(clean_commands)
    mechanical.wait_till_mechanical_is_ready()
    return vn, gammaO, tfreq
