import json
from ansys.dpf import core as dpf
from ansys.workbench.core import connect_workbench
from ansys.mechanical.core import connect_to_mechanical

workbench_server_port = 33086 # StartServer() to retrieve port
workbench_server_ip = None


systemName = "SYS 1"



# =============================================================================
# WORKBENCH & MECHANICAL CONNECTIONS
# =============================================================================
workbench = connect_workbench(
    port=workbench_server_port,
    host=workbench_server_ip if workbench_server_ip else None
)
mechPort = workbench.start_mechanical_server(systemName)
mechanical = connect_to_mechanical(ip='localhost', port=mechPort)

# =============================================================================
# 2. EXTRACT MESH DATA VIA IRONPYTHON
# =============================================================================
# We run this snippet inside Mechanical. It extracts the nodes and elements 
# into standard Python lists, then dumps it to a JSON string. This prevents 
# PyMechanical from having to make 100,000+ individual network calls for each node.

extraction_script = """
import json
mesh_data = ExtAPI.DataModel.MeshDataByName("Global")
 
# ...existing node/element extraction...
 
# Extract Named Selections
named_selections = {}
model = ExtAPI.DataModel.Project.Model
if model.NamedSelections is not None:
    for ns in model.NamedSelections.Children:
        loc = ns.Location
        sel_type = loc.SelectionType.ToString()  # MeshNodes / MeshElements / GeometryEntities
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
                node_ids.extend([int(i) for i in region.NodeIds])
                element_ids.extend([int(i) for i in region.ElementIds])
        named_selections[ns.Name] = {
            "node_ids": sorted(set(node_ids)),
            "element_ids": sorted(set(element_ids)),
        }
 
output = {
    "num_nodes": mesh_data.NodeCount,
    "num_elements": mesh_data.ElementCount,
    "nodes": nodes,
    "elements": elements,
    "named_selections": named_selections,
}
json.dumps(output)
"""

print("Extracting mesh data from Mechanical memory...")
raw_json = mechanical.run_python_script(extraction_script)
data = json.loads(raw_json)
print(f"Extracted {data['num_nodes']} nodes and {data['num_elements']} elements.")

# =============================================================================
# 3. BUILD THE DPF MESHED REGION (Your Logic)
# =============================================================================
print("Constructing DPF MeshedRegion...")

# Empty DPF MeshedRegion
mesh = dpf.MeshedRegion(
    num_nodes=data["num_nodes"],
    num_elements=data["num_elements"]
)

# Add nodes and build the ID -> Index map
id_to_index = {}
for index, node_data in enumerate(data["nodes"]):
    node_id, x, y, z = node_data
    mesh.nodes.add_node(node_id, [x, y, z])
    id_to_index[node_id] = index

# Add elements using the index mapping
for elem_data in data["elements"]:
    elem_id, shape, node_ids = elem_data
    
    # Convert Ansys Node IDs to DPF 0-based Node Indices
    conn = [id_to_index[nid] for nid in node_ids]
    
    if shape == "solid":
        mesh.elements.add_solid_element(elem_id, conn)
    elif shape == "shell":
        mesh.elements.add_shell_element(elem_id, conn)
    elif shape == "beam":
        mesh.elements.add_beam_element(elem_id, conn)
    else:
        # Note: DPF's add_element requires a specific element_type enum.
        # If it's unknown, it's safer to skip or log it, as DPF may reject it.
        pass

# Mechanical MeshData coordinates are always returned in standard SI (meters)
mesh.unit = "m" 

print("Mesh successfully built!")
print(mesh)
# =============================================================================
# 4. EXTRACT TRUE 2D SURFACE MESH FOR NAMED SELECTION "GE"
# =============================================================================
print("Extracting 2D surface skin mesh for GE...")

# 1. Grab the NODE IDs for "GE" (NOT the element IDs!)
ge_node_ids = data["named_selections"]["GI"]["node_ids"]

# 2. Create a DPF Nodal Scoping
ge_nodal_scoping = dpf.Scoping(ids=ge_node_ids, location=dpf.locations.nodal)

# 3. Use the DPF skin operator on the FULL mesh, filtered by the Nodal Scoping.
# This tells DPF: "Calculate the global skin, but only return the 2D faces 
# that are completely formed by these specific nodes."
skin_op = dpf.operators.mesh.skin()
skin_op.inputs.mesh.connect(mesh)
skin_op.inputs.mesh_scoping.connect(ge_nodal_scoping) # Pin 1 accepts Nodal scoping perfectly!

# 4. Generate the final 2D skin mesh
skin_mesh_ge = skin_op.outputs.mesh()

print("True surface mesh successfully extracted!")
print(skin_mesh_ge)

# Optional: Plot to verify or export to STL
pyvista_grid = skin_mesh_ge.grid
pyvista_grid.plot()
# pyvista_grid.save("GE_surface_only.stl")
