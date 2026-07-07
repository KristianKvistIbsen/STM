import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os

def excitation_from_array(stm_synthesizer, coeffs_array, export_csv_path=None, export_freq_index=0):
    """
    Build an excitation matrix from coefficients, with an option to synthesize 
    and export the physical pressure field to a CSV file.

    Parameters
    ----------
    stm_synthesizer : STMSynthesizer
        The instantiated STMSynthesizer object.
    coeffs_array : array-like
        1D array of shape (n_coeffs_I,) → will be tiled across all frequencies,
        or 2D array of shape (n_coeffs_I, n_frequencies).
    export_csv_path : str, optional
        If provided, the synthesized pressure field on the internal mesh will 
        be exported to this file path (e.g., 'excitation_field.csv').
    export_freq_index : int, optional
        The frequency index to use when exporting a 2D excitation array to CSV. 
        Defaults to 0.

    Returns
    -------
    ndarray
        Validated excitation array of shape (n_coeffs_I, n_frequencies).
    """
    e = np.asarray(coeffs_array, dtype=np.complex128)
    I = stm_synthesizer.n_coeffs_I
    F = stm_synthesizer.n_frequencies

    # 1. Validate and shape the excitation array
    if e.ndim == 1:
        if e.shape[0] != I:
            raise ValueError(f"1D excitation must have length n_coeffs_I={I}, got {e.shape[0]}.")
        e_full = np.tile(e[:, np.newaxis], (1, F))

    elif e.ndim == 2:
        if e.shape != (I, F):
            raise ValueError(f"2D excitation must have shape ({I}, {F}), got {e.shape}.")
        e_full = e

    else:
        raise ValueError("Excitation array must be 1D or 2D.")

    # 2. Export the physical pressure field to CSV if requested
    if export_csv_path is not None:
        if stm_synthesizer.input_basis_vectors.size == 0 or stm_synthesizer.gammaI_points.size == 0:
            raise ValueError("Cannot export pressure: STM file is missing basis vectors or internal coordinates.")
        
        # Synthesize the pressure field for the requested frequency bin
        # Shape: (n_points, n_coeffs_I) @ (n_coeffs_I,) -> (n_points,)
        p_recon = stm_synthesizer.input_basis_vectors @ e_full[:, export_freq_index]
        
        coords = stm_synthesizer.gammaI_points
        data = np.column_stack((
            coords[:, 0], 
            coords[:, 1], 
            coords[:, 2], 
            np.real(p_recon), 
            np.imag(p_recon)
        ))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(export_csv_path)), exist_ok=True)
        
        header = "x,y,z,real,imag"
        np.savetxt(export_csv_path, data, delimiter=',', header=header, comments='', fmt='%.16e')
        print(f"Exported synthesized pressure field to: {export_csv_path}")

    return e_full


def excitation_from_pressure(stm_synthesizer, pressure_field,
                             plot: bool = False,
                             plot_freq_index: int = 0,
                             plot_part: str = "abs"):
    """
    Decompose a physical pressure field on Γ_I into STM basis coefficients
    using least-squares projection.

    Parameters
    ----------
    stm_synthesizer : STMSynthesizer
        The instantiated STMSynthesizer object.
    pressure_field : array-like
        Pressure values on the internal mesh nodes (1D or 2D).
    plot : bool, optional
        If True, automatically calls the VTK-based pressure reconstruction plot.
    plot_freq_index, plot_part : 
        Passed to the plot method when `plot=True`.

    Returns
    -------
    coeffs : ndarray
        Basis coefficients of shape (n_coeffs_I, n_frequencies).
    error_percent : float
        Relative L2 reconstruction error in percent.
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
            raise ValueError(f"2D Pressure field shape {p.shape} must match ({N}, {F}).")
        p_lsq = p
    else:
        raise ValueError("Pressure field must be 1D or 2D.")

    # Least-squares projection onto the basis
    coeffs, _, _, _ = np.linalg.lstsq(stm_synthesizer.input_basis_vectors, p_lsq, rcond=None)

    # Reconstruction and error
    p_recon = stm_synthesizer.input_basis_vectors @ coeffs
    p_norm = np.linalg.norm(p_lsq)

    error_percent = 0.0 if p_norm == 0 else (np.linalg.norm(p_recon - p_lsq) / p_norm) * 100.0

    # Tile if input was static (1D)
    if p.ndim == 1:
        coeffs = np.tile(coeffs, (1, F))

    # Optional visualization using new VTK plotter
    if plot:
        stm_synthesizer.plot_pressure_reconstruction(
            p_target=p_lsq,
            p_recon=p_recon,
            error_percent=error_percent,
            freq_index=plot_freq_index,
            part=plot_part
        )

    return coeffs, error_percent


def excitation_from_dict(stm_synthesizer, coeffs_dict):
    """
    Build an excitation matrix from a dictionary of labels or indices.

    Parameters
    ----------
    stm_synthesizer : STMSynthesizer
    coeffs_dict : dict
        Keys can be:
            - string labels (e.g. 'IVMB_0', 'Y_2_1')
            - tuples (l, m) for spherical harmonics
            - integer indices
        Values can be scalars or 1D arrays of length n_frequencies.

    Returns
    -------
    ndarray
        Excitation matrix of shape (n_coeffs_I, n_frequencies).
    """
    I = stm_synthesizer.n_coeffs_I
    F = stm_synthesizer.n_frequencies
    e = np.zeros((I, F), dtype=np.complex128)

    for key, val in coeffs_dict.items():
        if isinstance(key, str):
            try:
                idx = stm_synthesizer.input_labels.index(key)
            except ValueError:
                raise KeyError(f"Label '{key}' not found. Available: {stm_synthesizer.input_labels}")
        elif isinstance(key, (tuple, list)):
            if stm_synthesizer.input_basis_type != "spherical_harmonics":
                raise ValueError("Tuple keys (l, m) are only valid for spherical_harmonics basis.")
            idx = stm_synthesizer.lm_to_index(int(key[0]), int(key[1]))
        else:
            idx = int(key)

        if not (0 <= idx < I):
            raise IndexError(f"Excitation index {idx} out of range [0, {I})")

        val_arr = np.asarray(val, dtype=np.complex128)

        if val_arr.ndim == 0:
            e[idx, :] = val_arr
        elif val_arr.ndim == 1 and val_arr.shape[0] == F:
            e[idx, :] = val_arr
        else:
            raise ValueError(f"Value for key '{key}' must be scalar or array of length {F}.")

    return e


def frequency_independent_excitation_from_csv(stm_synthesizer, csv_filepath,
                                              num_neighbors=3, p=2,
                                              plot: bool = False,
                                              plot_freq_index: int = 0,
                                              plot_part: str = "abs"):
    """
    Load pressure data from CSV, interpolate onto Γ_I using IDW, and project to basis.

    Parameters
    ----------
    stm_synthesizer : STMSynthesizer
    csv_filepath : str
        Path to CSV file (columns: index, x, y, z, pressure).
    num_neighbors, p : int, float
        Parameters for Inverse Distance Weighting (IDW).
    plot : bool
        If True, automatically shows the VTK pressure reconstruction plot.

    Returns
    -------
    coeffs, error_percent
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
        plot=plot,
        plot_freq_index=plot_freq_index,
        plot_part=plot_part
    )
