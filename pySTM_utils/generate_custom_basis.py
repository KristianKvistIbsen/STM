import json
from ansys.dpf import core as dpf
from ansys.workbench.core import connect_workbench
from ansys.mechanical.core import connect_to_mechanical
import numpy as np
import os

workbench_server_port = 3732 # StartServer() to retrieve port
workbench_server_ip = None
systemName          = "SYS"

INTERNAL_NS = "GI"

# --- USER INPUTS FOR HARMONICS ---
number_of_stator_teeth = 48  # UPDATE THIS with your actual number of teeth
max_harmonic = 8             # UPDATE THIS with your desired highest harmonic
output_filename = r"C:\01_gitrepos\STM\stator_harmonic_loads.csv"
# ---------------------------------


workbench = connect_workbench(
    port=workbench_server_port,
    host=workbench_server_ip if workbench_server_ip else None
)
mechPort = workbench.start_mechanical_server(systemName)
mechanical = connect_to_mechanical(ip='localhost', port=mechPort)


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
        if ns.Name not in ["{INTERNAL_NS}"]:
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

gammaI_from_ansys = extract_skin_for_ns(INTERNAL_NS, mesh, data)

# =============================================================================
# EXTRACT COORDINATES AND IDENTIFY ROTATION AXIS
# =============================================================================
print("Processing node coordinates to identify rotation axis...")

# DPFArray can be safely cast to a numpy array for vectorized math
coords = np.asarray(gammaI_from_ansys.nodes.coordinates_field.data)

# Center the coordinates based on the bounding box center
centroid = np.mean(coords, axis=0)
centered_coords = coords - centroid

# Identify the rotation axis by finding the axis with the minimum radial variance.
# R = sqrt(u^2 + v^2) where u and v are the orthogonal axes.
var_r = []
for axis in range(3):
    ortho_axes = [i for i in range(3) if i != axis]
    r = np.sqrt(centered_coords[:, ortho_axes[0]]**2 + centered_coords[:, ortho_axes[1]]**2)
    var_r.append(np.var(r))

rot_axis_idx = np.argmin(var_r)
axis_names = ['X', 'Y', 'Z']
print(f"Identified motor rotation axis: {axis_names[rot_axis_idx]}")

# =============================================================================
# CALCULATE THETA AND SPATIAL HARMONICS
# =============================================================================
# Extract the orthogonal axes to calculate the angular position (theta)
ortho_axes = [i for i in range(3) if i != rot_axis_idx]
u = centered_coords[:, ortho_axes[0]]
v = centered_coords[:, ortho_axes[1]]

# Calculate theta from -pi to pi
theta = np.arctan2(v, u)



# Anti-aliasing check based on Nyquist limit
nyquist_limit = number_of_stator_teeth // 2
if max_harmonic > nyquist_limit:
    print(f"WARNING: max_harmonic ({max_harmonic}) exceeds the spatial Nyquist limit ({nyquist_limit}) for {number_of_stator_teeth} teeth.")
    print(f"Harmonics above {nyquist_limit} will alias. Clamping max_harmonic to {nyquist_limit}.")
    max_harmonic = nyquist_limit

# Initialize the output matrix with x, y, z coordinates
output_data = coords.copy()
header_cols = ["x", "y", "z"]

# p0: Constant force component (0th harmonic)
p0 = np.ones_like(theta)
output_data = np.column_stack((output_data, p0))
header_cols.append("p0")

# p1 to pk: sin(k*theta)
for k in range(1, max_harmonic + 1):
    pk = np.sin(k * theta)
    output_data = np.column_stack((output_data, pk))
    header_cols.append(f"p{k}")

# =============================================================================
# EXPORT TO CSV
# =============================================================================

header_str = ",".join(header_cols)

np.savetxt(
    output_filename, 
    output_data, 
    delimiter=",", 
    header=header_str, 
    comments="", 
    fmt="%.6e"
)

print(f"Successfully generated {max_harmonic} harmonics.")
print(f"Saved load profile to: {os.path.abspath(output_filename)}")

# gammaI_from_ansys.grid.plot(scalars=pk)
