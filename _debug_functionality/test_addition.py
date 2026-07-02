import numpy as np
import pySTM
import pySDEM
import pyshtools as pysh
import pyvista as pv
from scipy.special import spherical_jn, spherical_yn
import time
import matplotlib.pyplot as plt


# Load in data ================================================================
loaded_data = pySTM.load_stm_results(r"N:/PhD/GTM/h5 files/4paper/simple.h5")
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



# %%

excitation_response1 = np.zeros(n_coeffs_I,dtype=np.complex128)
excitation_response1[0] = 1 #Y00
excitation_response1[1] = 2+3j #Y1-1
# excitation_response[2] = 2j #Y10
# excitation_response[3] = 1-1j #Y11

excitation_response2 = np.zeros(n_coeffs_I,dtype=np.complex128)
excitation_response2[0] = 1 #Y00
excitation_response2[1] = 2+3j #Y1-1
excitation_response1[2] = 2j #Y10
excitation_response1[3] = 1-1j #Y11


gammaI = loaded_data["mesh_data"]["INTERNAL"]["InternalGrid"]
points = gammaI.points
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

pressureOut1 = sh_array@excitation_response1
pressureOut2 = sh_array@excitation_response2
pressureOut12 = sh_array@(excitation_response1+excitation_response2)

# %%
X1, _, _, _ = np.linalg.lstsq(sh_array, pressureOut1, rcond=None)
X2, _, _, _ = np.linalg.lstsq(sh_array, pressureOut2, rcond=None)
X12, _, _, _ = np.linalg.lstsq(sh_array, pressureOut12, rcond=None)

aaa = X12-(X1+X2)
