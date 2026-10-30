- Make the code work even if the exterior mesh is made of 4sided surface elements and not only triangular elements. Pyansys has the functionality to do this using the following method.
    op = dpf.operators.mesh.mesh_to_tetra() # operator instantiation
    op.inputs.mesh.connect(my_mesh)
    my_mesh = op.outputs.mesh()
    my_node_mapping = op.outputs.node_mapping()
    my_element_mapping = op.outputs.element_mapping()
    
    
- Update the motor basis to also include the moment contribution on the stator teeth.


- When it prints "Solved for ..." for each basis, it should also print the solve time for each itteration.

- Update the STM method, so instead of automatically generating the excitation basis in the main script, it simply assumes the pressure baiss is provided. 
    There will then be different "basis generation scripts" to generate basis for spherical harmonic excitation, one for harmoinc basis on electric motor, one which decomposes a provided pressure field and uses SVD etc. 
    The basis generation scripts will produce an STM basis, which is a h5 file or somehting with a mesh and a number of pressure basis vectors in a standard format, which the STM method will then import. 
    There must be a convenient way of visualizing a given pressure basis on the mesh of the excited surface
    There will similarly be a convenient way of providing a "real world" pressure field, and then there will be a way of decomposing and reconstructing this real world basis based on a given h5_stm_basis, so i can see if the basis is sufficient, and so i can decompose it and compute the response using the STM called something like "real world excitation basis decomposer (find better name)".
    In the decomposition and reconstruction, there should also be plots to visualize how the mesh in the basis h5 file and the realworld pressures (provided as a csv of x y z pressure_real pressure_imag) line up, so i do not 
    accidentily try to deompose a real world pressure field on meshA using a basis on meshB. (the meshes may be different, but they should be in the same place)
    
    The workflow is then essentially:
     - Generate basis by providing a named selection in the ansys model, (and other nessesary input depending on the specific basis...)
     - Export the generated basis as a h5 file or something including the mesh of the basis and nessesary metadata to identify it and understand what the basis represents.
     - When running the STM solver (STM_main.py) i then provide the basis file, and it computes the STM for that basis on the named selection specified in the STM_main.py file (the named selection name and mesh may be different from the basis generation pass, but the mesh location is of course the same.)
     - The resulting STM file (h5) ALSO has within it the excitation basis file.
     - When i want to compute the response from a real-world excitation, i will then give the STM file to the "real world excitation basis decomposer (find a better name)" and IT will act in place of the basis_h5_file, for decomposing, reconstructin, giving reconstruction error, plotting the original pressure and the reconstructed pressure for comparison etc.
     - I then have the decomposition, which i can use to compute responses using the STM.
     
     
       
