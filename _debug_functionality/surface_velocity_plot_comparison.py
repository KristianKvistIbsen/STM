import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import kdpf
import pySDEM
import pySTM
import pyshtools as pysh
from ansys.dpf import core as dpf


loaded_data = pySTM.load_stm_results(r"N:/PhD/GTM/h5 files/4paper/scala_l4_with_error_lo80.h5")
n_coeffs_I = loaded_data["metadata"]["computation_parameters"]["n_coeffs_I"]
STM = loaded_data["STM"]
G = loaded_data["results_data"]["G"]
lmax_O = int(loaded_data["metadata"]["user_settings"]["lmax_O"])
areas = loaded_data["mesh_data"]["EXTERNAL"]["mesh_metadata"]["areas"]   
frequencies = loaded_data["results_data"]["frequencies"]
S = loaded_data["mesh_data"]["EXTERNAL"]["SDEM_Coordinates"]
grid_gammaO = loaded_data["mesh_data"]["EXTERNAL"]["ExternalGrid"]
point_mapping = loaded_data["results_data"]["point_mappings"]["point_mapping"]
x, y, z = S[:, 0], S[:, 1], S[:, 2]
r, lat, lon = pySDEM.cart_to_lat_lon(x, y, z)


excitation_response = np.zeros(n_coeffs_I,dtype=np.complex128)
excitation_response[0] = 1 #Y00
excitation_response[1] = 1 #Y1-1
excitation_response[2] = 2j #Y10
excitation_response[3] = 1-1j #Y11

synthesized_response_clm1d = np.einsum('ijk,i->jk', STM, excitation_response)
synthesized_velocity = G @ synthesized_response_clm1d




external_ns = 'GAMMA_E'
model = dpf.Model(r"C:\Users\105849\Desktop\file.rst")
tfreq = kdpf.get_tfreq(model)
gammaO = kdpf.get_skin_mesh_from_ns(external_ns, model)
normals = kdpf.get_normals(gammaO)
vn = kdpf.get_normal_velocities(model, gammaO, tfreq, normals)
# %%

freq_idx = 150
plotter = pv.Plotter(shape=(1, 3))

# STM
plotter.subplot(0, 0)
field_synth = np.real(synthesized_velocity[:, freq_idx])
mesh_synth = grid_gammaO.copy()
mesh_synth.point_data['Velocity'] = field_synth
# plotter.add_mesh(mesh_synth, scalars='Velocity', cmap='turbo', show_edges=False)
# plotter.add_title(f'Synthesized Velocity at {frequencies[freq_idx]:.2f} Hz')

# FEA
plotter.subplot(0, 1)
field_vn = vn[2 * freq_idx].data + 1j*vn[2 * freq_idx+1].data
mesh_vn = mesh_synth.copy()
mesh_vn.point_data['Velocity'] = np.real(field_vn[point_mapping])
# plotter.add_mesh(mesh_vn, scalars='Velocity', cmap='turbo', show_edges=False)
# plotter.add_scalar_bar(title='Velocity')
# plotter.add_title(f'Surface Velocity at {frequencies[freq_idx]:.2f} Hz')

# Difference mesh
plotter.subplot(0, 2)
field_diff = field_synth - field_vn[point_mapping]
mesh_diff = grid_gammaO.copy()
mesh_diff.point_data['Velocity'] = field_diff
# plotter.add_mesh(mesh_diff, scalars='Velocity', cmap='turbo', show_edges=False)
# plotter.add_scalar_bar(title='Velocity')
# plotter.add_title(f'Velocity Difference at {frequencies[freq_idx]:.2f} Hz')

# Export meshes to VTU files
mesh_synth.save("N:/GTML80.vtu")
mesh_vn.save("N:/FEAL80.vtu")
mesh_diff.save("N:/DIFFL80.vtu")


# %%
import trimesh
def stl_to_dpf_mesh(stl_path):
    mesh = trimesh.load(stl_path, file_type='stl')
    connectivity = mesh.faces
    coordinates = mesh.vertices
    meshed_region = dpf.MeshedRegion(
        num_nodes=coordinates.shape[0],
        num_elements=connectivity.shape[0]
    )
    print("Warning: scaling factor of 1000 used to convert from mm to m")
    for i, coord in enumerate(coordinates, start=0):
        meshed_region.nodes.add_node(i, coord / 1000)

    for i, face in enumerate(connectivity, start=0):
        meshed_region.elements.add_element(i, "shell", face.tolist())
    return meshed_region, mesh.area_faces

op_mapping_workflow = dpf.operators.mapping.prepare_mapping_workflow()
op_mapping_workflow.inputs.input_support.connect(gammaO)
stl_mesh, _ = stl_to_dpf_mesh(r"N:/PhD/GTM/NEBULA_DF_24_STL_5.stl")
op_mapping_workflow.inputs.output_support.connect(stl_mesh)
op_mapping_workflow.inputs.filter_radius.connect(0.005)
mapping_workflow = op_mapping_workflow.outputs.mapping_workflow()
mapping_workflow.progress_bar=False


mapping_workflow.connect('source',vn)
vn = mapping_workflow.get_output('target', output_type="fields_container")
