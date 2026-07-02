# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:31:16 2025

@author: 105849
"""

import numpy as np
import pySTM
import pySDEM
import pyshtools as pysh

# Load in data ================================================================
loaded_data = pySTM.load_stm_results(r"N:\PhD\GTM\upm10_stm\upm10_l4_60.h5")
n_coeffs_I = loaded_data["metadata"]["computation_parameters"]["n_coeffs_I"]
STM = loaded_data["STM"]
G = loaded_data["results_data"]["G"]
lmax_O = int(loaded_data["metadata"]["user_settings"]["lmax_O"])
areas = loaded_data["mesh_data"]["EXTERNAL"]["mesh_metadata"]["areas"]   
frequencies = loaded_data["results_data"]["frequencies"]
SE = loaded_data["mesh_data"]["EXTERNAL"]["SDEM_Coordinates"]
SI = loaded_data["mesh_data"]["INTERNAL"]["SDEM_Coordinates"]
grid_gammaO = loaded_data["mesh_data"]["EXTERNAL"]["ExternalGrid"]
grid_gammaI = loaded_data["mesh_data"]["INTERNAL"]["InternalGrid"]
point_mapping = loaded_data["results_data"]["point_mappings"]["point_mapping"]

excitation_response = np.zeros(n_coeffs_I,dtype=np.complex128)
excitation_response[0] = 1 #Y00
# excitation_response[1] = 1 #Y1-1
# excitation_response[2] = 2j #Y10
excitation_response[3] = 1-1j #Y11
# excitation_response[4] = 1 #Y2-2
# excitation_response[5] = 1 #Y2-1
# excitation_response[6] = 1 #Y20
excitation_response[7] = 2-3j #Y21
# excitation_response[8] = 1 #Y22
# excitation_response[9] = 1 #Y3-3
# excitation_response[10] = 1 #Y3-2
# excitation_response[11] = 1 #Y3-1
# excitation_response[12] = 1 #Y30
# excitation_response[13] = 1 #Y31
# excitation_response[14] = 1 #Y32
# excitation_response[15] = 1 #Y33
# excitation_response[16] = 1 #Y4-4
excitation_response[17] = 2j #Y4-3
# excitation_response[18] = 1 #Y4-2
# excitation_response[19] = 1 #Y4-1
# excitation_response[20] = 1 #Y40
# excitation_response[21] = 1 #Y41
# excitation_response[22] = 1 #Y42
# excitation_response[23] = 1 #Y43
# excitation_response[24] = 1 #Y44
# excitation_response[25] = 1 #Y5-5
# excitation_response[26] = 1 #Y5-4
# excitation_response[27] = 1 #Y5-3
# excitation_response[28] = 1 #Y5-2
# excitation_response[29] = 1 #Y5-1
# excitation_response[30] = 1 #Y50
# excitation_response[31] = 1 #Y51
# excitation_response[32] = 1 #Y52
# excitation_response[33] = 1 #Y53
# excitation_response[34] = 1 #Y54
# excitation_response[35] = 1 #Y55

points = SI
x, y, z = points[:, 0], points[:, 1], points[:, 2]
r, lat, lon = pySDEM.cart_to_lat_lon(x, y, z) 
def generate_spherical_harmonics(lmax, points, lat, lon):
    """Generate spherical harmonics for given lmax and points."""
    n_harmonics = (lmax + 1) ** 2
    n_points = len(points)
    sh_array = np.zeros((n_points, n_harmonics), dtype=np.complex128)
    harm_idx = 0

    for l in range(0,lmax + 1):
        for m in range(-l, l + 1):
            coeffs = np.zeros((2, l + 1, l + 1), dtype=np.complex128)
            if m >= 0:
                coeffs[0, l, m] = 1.0
            else:
                coeffs[1, l, abs(m)] = 1.0
            sh_coeffs = pysh.SHCoeffs.from_array(coeffs)
            sh_array[:, harm_idx] = pysh.expand.MakeGridPointC(sh_coeffs.coeffs, lat, lon)
            harm_idx += 1
    return sh_array, n_harmonics

sh_array, n_harmonics = generate_spherical_harmonics(int(np.sqrt(n_coeffs_I)-1),points,lat,lon)

pressureOut = sh_array@excitation_response


export_path = r"N:\upm10_pressure_for_acousticFEA.csv"
data = np.column_stack((
    points[:, 0],  # x
    points[:, 1],  # y
    points[:, 2],  # z
    np.real(pressureOut),  # real part
    np.imag(pressureOut)   # imaginary part
))
header = ['x', 'y', 'z', 'real', 'imag']
np.savetxt(export_path, data, delimiter=',', header=','.join(header), 
           comments='', fmt='%.16e')

# %%

grid_gammaI.plot(scalars=pressureOut.real,cmap="turbo")
