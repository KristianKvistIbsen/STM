import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def excitation_from_array(stm_synthesizer, coeffs_array):
    """
    Build an excitation matrix from coefficients.
    
    Parameters:
    -----------
    stm_synthesizer : STMSynthesizer
        The instantiated STMSynthesizer object.
    coeffs_array : array-like
        1D array of shape (n_coeffs_I,) [will be tiled across all frequencies]
        OR 2D array of shape (n_coeffs_I, n_frequencies).
        
    Returns:
    --------
    ndarray
        The validated excitation array, shape (n_coeffs_I, n_frequencies).
    """
    e = np.asarray(coeffs_array, dtype=np.complex128)
    I = stm_synthesizer.n_coeffs_I
    F = stm_synthesizer.n_frequencies

    if e.ndim == 1:
        if e.shape[0] != I:
            raise ValueError(f"1D excitation must have length n_coeffs_I={I}, got {e.shape[0]}.")
        # Tile across all frequencies
        return np.tile(e[:, np.newaxis], (1, F))
    elif e.ndim == 2:
        if e.shape != (I, F):
            raise ValueError(f"2D excitation must have shape (n_coeffs_I={I}, n_frequencies={F}), got {e.shape}.")
        return e
    else:
        raise ValueError("Excitation array must be 1D or 2D.")

def excitation_from_pressure(stm_synthesizer, pressure_field, return_figure=False, plot_freq_index=0, plot_part="abs"):
    """
    Decompose a physical pressure field defined on Gamma_I into the STM basis coefficients.
    If return_figure=True, returns a 3rd argument containing a Plotly diagnostic figure.
    """
    if stm_synthesizer.input_basis_vectors.size == 0:
        raise ValueError("Basis vectors are not stored in this STM file. Cannot project pressure.")
        
    p = np.asarray(pressure_field, dtype=np.complex128)
    N = stm_synthesizer.input_basis_vectors.shape[0]
    F = stm_synthesizer.n_frequencies
    
    if p.ndim == 1:
        if p.shape[0] != N:
            raise ValueError(f"1D Pressure field size ({p.shape[0]}) does not match mesh ({N}).")
        p_lsq = p[:, np.newaxis] 
    elif p.ndim == 2:
        if p.shape != (N, F):
            raise ValueError(f"2D Pressure field shape {p.shape} must match (n_nodes={N}, n_freq={F}).")
        p_lsq = p
    else:
        raise ValueError("Pressure field must be 1D or 2D.")
        
    # Least squares projection
    coeffs, _, _, _ = np.linalg.lstsq(stm_synthesizer.input_basis_vectors, p_lsq, rcond=None)
    
    p_recon = stm_synthesizer.input_basis_vectors @ coeffs
    p_norm = np.linalg.norm(p_lsq)
    
    if p_norm == 0:
        error_percent = 0.0
    else:
        error_percent = (np.linalg.norm(p_recon - p_lsq) / p_norm) * 100.0
        
    # Generate diagnostic figure if requested
    fig = None
    if return_figure:
        fig = stm_synthesizer.figure_pressure_reconstruction(
            p_target=p_lsq, 
            p_recon=p_recon, 
            error_percent=error_percent, 
            freq_index=plot_freq_index, 
            part=plot_part
        )
        
    # Tile the result if the input pressure was purely 1D (static)
    if p.ndim == 1:
        coeffs = np.tile(coeffs, (1, F))
        
    if return_figure:
        return coeffs, error_percent, fig
    return coeffs, error_percent

def excitation_from_dict(stm_synthesizer, coeffs_dict):
    """
    Build an excitation matrix from a dictionary mapping labels/indices to values or arrays.

    Parameters:
    -----------
    stm_synthesizer : STMSynthesizer
        The instantiated STMSynthesizer object.
    coeffs_dict : dict
        Keys can be string labels ('Y_1_-1', 'IVMB_2'), tuples (l, m), or flat integer indices.
        Values can be scalar (applied to all frequencies) or 1D arrays of length n_frequencies.
        Example: {'IVMB_0': 1.0} or {'IVMB_1': np.array([...freqs...])}.
        
    Returns:
    --------
    ndarray
        The validated excitation matrix, shape (n_coeffs_I, n_frequencies).
    """
    I = stm_synthesizer.n_coeffs_I
    F = stm_synthesizer.n_frequencies
    e = np.zeros((I, F), dtype=np.complex128)
    
    for key, val in coeffs_dict.items():
        if isinstance(key, str):
            try:
                idx = stm_synthesizer.input_labels.index(key)
            except ValueError:
                raise KeyError(
                    f"Label '{key}' not found in STM basis labels. "
                    f"Available labels: {stm_synthesizer.input_labels}"
                )
        elif isinstance(key, (tuple, list)):
            if stm_synthesizer.input_basis_type != "spherical_harmonics":
                raise ValueError(
                    f"Tuple keys {key} are only for 'spherical_harmonics'. Use string labels or ints."
                )
            idx = stm_synthesizer.lm_to_index(int(key[0]), int(key[1]))
        else:
            idx = int(key)
            
        if not 0 <= idx < I:
            raise IndexError(f"Excitation index {idx} out of range [0, {I})")
            
        val_arr = np.asarray(val, dtype=np.complex128)
        
        # Broadcast scalar to entire row, or assign frequency array
        if val_arr.ndim == 0:
            e[idx, :] = val_arr 
        elif val_arr.ndim == 1 and val_arr.shape[0] == F:
            e[idx, :] = val_arr
        else:
            raise ValueError(f"Value for key '{key}' must be a scalar or an array of length n_frequencies={F}.")
        
    return e

def frequency_independent_excitation_from_csv(stm_synthesizer, csv_filepath, num_neighbors=3, p=2, return_figure=False, plot_freq_index=0, plot_part="abs"):
    """
    Load a raw CSV, interpolate it onto Gamma_I, and decompose it.
    If return_figure=True, returns a 3rd argument containing a Plotly diagnostic figure.
    """
    if stm_synthesizer.gammaI_points.size == 0:
        raise ValueError("Internal mesh coordinates (gammaI_points) are not stored in the STM.")
        
    df = pd.read_csv(csv_filepath, header=None, skiprows=1) 
    
    csv_coords = df.iloc[:, 1:4].values
    csv_pressures = np.asarray(df.iloc[:, 4].values, dtype=np.complex128)
    
    target_coords = stm_synthesizer.gammaI_points
    tree = cKDTree(csv_coords)
    distances, indices = tree.query(target_coords, k=num_neighbors)
    mapped_pressures = np.zeros(target_coords.shape[0], dtype=np.complex128)
    
    for i in range(target_coords.shape[0]):
        dist = distances[i]
        idx = indices[i]
        
        if dist[0] < 1e-12:
            mapped_pressures[i] = csv_pressures[idx[0]]
        else:
            weights = 1.0 / (dist ** p)
            mapped_pressures[i] = np.sum(weights * csv_pressures[idx]) / np.sum(weights)

    return excitation_from_pressure(
        stm_synthesizer, 
        mapped_pressures, 
        return_figure=return_figure,
        plot_freq_index=plot_freq_index,
        plot_part=plot_part
    )
