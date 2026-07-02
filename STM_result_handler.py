from __future__ import annotations

import ast
import numpy as np
from scipy.special import spherical_jn, spherical_yn

import pySTM

# Physical defaults (air)
RHO_AIR = 1.3       # kg/m^3
C_AIR = 343.0       # m/s
POWER_REF = 1e-12   # W, reference for dB (re 1 pW)

VTK_TRIANGLE = 5


def _import_go():
    """Lazily import plotly.graph_objects so compute methods work without plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the figure_* methods. Install it with 'pip install plotly'."
        ) from exc
    return go


class STMSynthesizer:
    """Load an STM result file and synthesise responses, spectra and fields."""

    # ------------------------------------------------------------------ loading
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self._loaded = pySTM.load_stm_results(self.filepath)
        self._Z_table = None
        self._Z_cache_key = None
        self._extract()

    @classmethod
    def from_file(cls, filepath) -> "STMSynthesizer":
        return cls(filepath)

    def _extract(self):
        d = self._loaded
        rd = d.get("results_data", {}) or {}

        self.metadata = d.get("metadata", {}) or {}
        self.file_info = d.get("file_info", {}) or {}
        self.mesh_data = d.get("mesh_data", {}) or {}
        
        # Load the core STM matrix first so dimensions are available
        self.STM = np.asarray(d["STM"])                       # (n_coeffs_I, n_coeffs_O, n_freq)
        self.G = np.asarray(rd["G"])                          # (n_points, n_coeffs_O)
        self.frequencies = np.real(np.asarray(rd["frequencies"]).ravel()).astype(float)

        self.n_coeffs_I, self.n_coeffs_O, self.n_frequencies = self.STM.shape
        self.lmax_O = int(round(np.sqrt(self.n_coeffs_O))) - 1

        # ---------------- Unified Input Basis Handling ----------------
        basis_info = rd.get("input_basis", {})
        self.input_basis_type = basis_info.get("type", "spherical_harmonics")
        
        # Load explicit labels and basis vectors for projection safely
        raw_labels = basis_info.get("labels", self._generate_legacy_labels())
        
        if isinstance(raw_labels, bytes):
            raw_labels = raw_labels.decode('utf-8')
            
        if isinstance(raw_labels, str):
            if raw_labels.startswith('[') and raw_labels.endswith(']'):
                self.input_labels = ast.literal_eval(raw_labels)
            else:
                self.input_labels = [raw_labels]
        elif isinstance(raw_labels, (list, np.ndarray, tuple)):
            self.input_labels = [
                lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                for lbl in raw_labels
            ]
        else:
            self.input_labels = list(raw_labels)
            
        self.input_basis_vectors = np.asarray(basis_info.get("basis_vectors", []))
        self.gammaI_points = np.asarray(basis_info.get("gammaI_points", []))
        
        if self.input_basis_type == "spherical_harmonics":
            self.lmax_I = int(round(np.sqrt(self.n_coeffs_I))) - 1
        else:
            self.lmax_I = None 

        err = rd.get("error_data", {}) or {}
        self.abs_error = np.asarray(err["abs_error"]) if "abs_error" in err else None
        self.rel_error = np.asarray(err["rel_error"]) if "rel_error" in err else None

        ext = self.mesh_data.get("EXTERNAL", {}) or {}
        self.external_grid = ext.get("ExternalGrid", None)
        self.sdem_coordinates = (
            np.asarray(ext["SDEM_Coordinates"]) if "SDEM_Coordinates" in ext else None
        )
        mm = ext.get("mesh_metadata", {}) or {}
        self.areas = np.asarray(mm["areas"]) if "areas" in mm else None
        self._external_points, self._external_faces = self._grid_to_arrays(self.external_grid)

        self.equivalent_radius = (
            float(np.sqrt(np.sum(self.areas) / (4.0 * np.pi))) if self.areas is not None else None
        )

        j = np.arange(self.n_coeffs_O)
        l = np.floor(np.sqrt(j)).astype(int)
        m = j - l * l - l
        self._out_l = l
        self._out_sign = np.where(m >= 0, 0, 1)
        self._out_absm = np.abs(m)

    @staticmethod
    def _grid_to_arrays(grid):
        if grid is None:
            return None, None
        points = np.asarray(grid.points)
        faces = None
        cells_dict = getattr(grid, "cells_dict", None)
        if cells_dict:
            if VTK_TRIANGLE in cells_dict:
                faces = np.asarray(cells_dict[VTK_TRIANGLE])
            else:
                faces = np.asarray(next(iter(cells_dict.values())))
        return points, faces

    def _generate_legacy_labels(self):
        """Fallback for older files that didn't explicitly store 'labels'."""
        if self.input_basis_type == "spherical_harmonics":
            return [f"Y_{l}_{m}" for l, m in (self.index_to_lm(i) for i in range(self.n_coeffs_I))]
        return [f"Basis_{k}" for k in range(self.n_coeffs_I)]

    # ------------------------------------------------------------- introspection
    def summary(self) -> dict:
        fmin = float(self.frequencies.min()) if self.n_frequencies else None
        fmax = float(self.frequencies.max()) if self.n_frequencies else None
        summary_dict = {
            "filepath": self.filepath,
            "input_basis_type": self.input_basis_type,
            "lmax_O": self.lmax_O,
            "n_coeffs_I": self.n_coeffs_I,
            "n_coeffs_O": self.n_coeffs_O,
            "n_frequencies": self.n_frequencies,
            "frequency_range_hz": (fmin, fmax),
            "n_internal_points": self.gammaI_points.shape[0] if self.gammaI_points.size else None,
            "n_external_points": self._external_points.shape[0] if self._external_points is not None else None,
            "has_error_data": self.has_error_data,
        }
        if self.lmax_I is not None:
            summary_dict["lmax_I"] = self.lmax_I
        return summary_dict

    def __repr__(self):
        return (
            f"STMSynthesizer(basis='{self.input_basis_type}', n_coeffs_I={self.n_coeffs_I}, "
            f"lmax_O={self.lmax_O}, n_frequencies={self.n_frequencies}, file='{self.filepath}')"
        )

    @property
    def has_error_data(self) -> bool:
        return self.rel_error is not None and bool(np.any(self.rel_error != 0))

    # ------------------------------------------------------------- index helpers
    @staticmethod
    def lm_to_index(l: int, m: int) -> int:
        return l * l + l + m

    @staticmethod
    def index_to_lm(idx: int):
        l = int(np.floor(np.sqrt(idx)))
        return l, idx - l * l - l

    # -------------------------------------------------------- excitation builders
    def zero_excitation(self):
        return np.zeros(self.n_coeffs_I, dtype=np.complex128)

    def excitation_from_array(self, coeffs_array):
        """
        Build an excitation vector directly from an array of coefficients.
        
        Parameters:
        -----------
        coeffs_array : array-like, shape (n_coeffs_I,)
            The complex coefficients for each mode in order.
            E.g. np.array([1+1j, 0, 0, 2])
        """
        return self._as_excitation(coeffs_array)

    def excitation_from_pressure(self, pressure_field):
        """
        Decompose a physical pressure field defined on Gamma_I into the STM basis coefficients.
        
        Parameters:
        -----------
        pressure_field : array-like, shape (n_internal_points,)
            The complex pressure values matching the node ordering of `self.gammaI_points`.
            
        Returns:
        --------
        excitation : ndarray, shape (n_coeffs_I,)
            The input excitation vector ready to be passed to `radiated_power()` or `synthesize_velocity()`.
        """
        if self.input_basis_vectors.size == 0:
            raise ValueError("Basis vectors are not stored in this STM file. Cannot project pressure.")
            
        p = np.asarray(pressure_field, dtype=np.complex128).ravel()
        if p.shape[0] != self.input_basis_vectors.shape[0]:
            raise ValueError(
                f"Pressure field size ({p.shape[0]}) does not match the internal "
                f"mesh point count ({self.input_basis_vectors.shape[0]})."
            )
            
        # Least squares projection onto the basis: [Basis] * c = p
        coeffs, residuals, rank, s = np.linalg.lstsq(self.input_basis_vectors, p, rcond=None)
        return coeffs

    def _as_excitation(self, excitation):
        e = np.asarray(excitation, dtype=np.complex128).ravel()
        if e.shape[0] != self.n_coeffs_I:
            raise ValueError(f"excitation must have length n_coeffs_I={self.n_coeffs_I}, got {e.shape[0]}")
        return e

    # -------------------------------------------------------------- synthesis
    def synthesize_coeffs(self, excitation):
        """Output surface-velocity SH coefficients (1D layout), shape (n_coeffs_O, n_freq)."""
        e = self._as_excitation(excitation)
        return np.einsum("ijk,i->jk", self.STM, e)

    def synthesize_coeffs_2d(self, excitation):
        """Output SH coefficients in pyshtools layout: (2, lmax_O+1, lmax_O+1, n_freq)."""
        clm1d = self.synthesize_coeffs(excitation)
        clm = np.zeros((2, self.lmax_O + 1, self.lmax_O + 1, self.n_frequencies), dtype=np.complex128)
        clm[self._out_sign, self._out_l, self._out_absm, :] = clm1d
        return clm

    def synthesize_velocity(self, excitation):
        """Surface normal velocity on the external mesh, shape (n_external_points, n_freq)."""
        return self.G @ self.synthesize_coeffs(excitation)

    # --------------------------------------------------------------- radiated power
    @staticmethod
    def _impedance_Z(l, ka, rho, c):
        d_jn = spherical_jn(l, ka, derivative=True)
        d_yn = spherical_yn(l, ka, derivative=True)
        dh1 = d_jn + 1j * d_yn
        return 1j * rho * c / dh1

    def _impedance_table(self, rho, c):
        key = (rho, c)
        if self._Z_cache_key == key and self._Z_table is not None:
            return self._Z_table
        if self.equivalent_radius is None:
            raise ValueError("No external mesh areas available; cannot form the impedance table.")
        L = self.lmax_O
        Z = np.zeros((L + 1, self.n_frequencies), dtype=np.complex128)
        a = self.equivalent_radius
        for fi, freq in enumerate(self.frequencies):
            if freq <= 0:
                continue
            ka = (2.0 * np.pi * freq / c) * a
            for l in range(L + 1):
                Z[l, fi] = self._impedance_Z(l, ka, rho, c)
        self._Z_table = Z
        self._Z_cache_key = key
        return Z

    def radiated_power(self, excitation, rho: float = RHO_AIR, c: float = C_AIR) -> dict:
        clm = self.synthesize_coeffs_2d(excitation)          
        Z = self._impedance_table(rho, c)                    

        v_abs_sqr = np.sum(np.abs(clm) ** 2, axis=(0, 2))    
        p_abs_sqr = v_abs_sqr * np.abs(Z) ** 2               

        k = 2.0 * np.pi * self.frequencies / c
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(k > 0, 1.0 / (2.0 * rho * c * k ** 2) / np.sqrt(4.0 * np.pi), 0.0)

        per_order = p_abs_sqr * factor[np.newaxis, :]        
        total = per_order.sum(axis=0)
        return {
            "total": total,
            "total_db": self._to_db(total),
            "per_order": per_order,
            "frequencies": self.frequencies,
        }

    @staticmethod
    def _to_db(power, ref: float = POWER_REF):
        power = np.asarray(power, dtype=float)
        out = np.full(power.shape, np.nan)
        pos = power > 0
        out[pos] = 10.0 * np.log10(power[pos] / ref)
        return out

    @staticmethod
    def _scalar_part(field, part: str):
        part = part.lower()
        if part == "real":
            return np.real(field)
        if part in ("imag", "imaginary"):
            return np.imag(field)
        if part in ("abs", "mag", "magnitude"):
            return np.abs(field)
        if part in ("phase", "angle"):
            return np.angle(field)
        raise ValueError(f"Unknown part '{part}' (use real / imag / abs / phase)")

    # ------------------------------------------------------------------- figures
    def figure_power_spectrum(self, excitation, rho: float = RHO_AIR, c: float = C_AIR, title=None):
        go = _import_go()
        p = self.radiated_power(excitation, rho=rho, c=c)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.frequencies, y=p["total_db"], mode="lines", name="Total"))
        fig.update_layout(title=title or "Radiated sound power", xaxis_title="Frequency (Hz)", yaxis_title="Power (dB re 1 pW)", template="plotly_white")
        return fig

    def figure_order_power(self, excitation, rho: float = RHO_AIR, c: float = C_AIR, max_order=None, title=None):
        go = _import_go()
        p = self.radiated_power(excitation, rho=rho, c=c)
        per_order_db = self._to_db(p["per_order"])
        L = self.lmax_O if max_order is None else min(int(max_order), self.lmax_O)
        fig = go.Figure(data=go.Heatmap(x=self.frequencies, y=np.arange(L + 1), z=per_order_db[: L + 1, :], colorscale="Turbo", colorbar=dict(title="dB")))
        fig.update_layout(title=title or "Radiated power by spherical-harmonic degree", xaxis_title="Frequency (Hz)", yaxis_title="Degree l", template="plotly_white")
        return fig

    def figure_surface_velocity(self, excitation, freq_index: int = 0, part: str = "real", colorscale: str = "Turbo", title=None):
        go = _import_go()
        if self._external_points is None or self._external_faces is None:
            raise ValueError("No external mesh geometry available for 3D plotting.")
        field = self.synthesize_velocity(excitation)[:, freq_index]
        scalar = self._scalar_part(field, part)
        pts, faces = self._external_points, self._external_faces
        fig = go.Figure(data=go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=scalar, intensitymode="vertex", colorscale=colorscale, colorbar=dict(title=f"v ({part})"), flatshading=True, showscale=True))
        fig.update_layout(title=title or f"Surface normal velocity ({part}) @ {self.frequencies[freq_index]:.1f} Hz", scene=dict(aspectmode="data"), template="plotly_white")
        return fig

    def figure_error_spectrum(self, use_relative: bool = True, title=None):
        go = _import_go()
        err = self.rel_error if use_relative else self.abs_error
        fig = go.Figure()
        if err is None:
            fig.add_annotation(text="No error data in file", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        err = np.asarray(err)
        
        for i in range(err.shape[0]):
            label = self.input_labels[i]
            # Try to group by 'l' if it's a spherical harmonic, else group by the label itself
            group = f"l{label.split('_')[1]}" if label.startswith("Y_") else "basis"
            fig.add_trace(go.Scatter(x=self.frequencies, y=err[i, :], mode="lines", name=label, legendgroup=group))
            
        if not self.has_error_data:
            fig.add_annotation(text="Error data is all zero - regenerate the STM with the updated generator", xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False)
        fig.update_layout(title=title or ("Relative fit error" if use_relative else "Absolute fit error"), xaxis_title="Frequency (Hz)", yaxis_title="Relative error" if use_relative else "Absolute error", yaxis_type="log", template="plotly_white")
        return fig


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"C:/01_gitrepos/STM/STM_inVacuoModalBasis.h5"
    stm = STMSynthesizer(path)
    print(stm)
    
    # 2. Test Array Excitation (using explicit numpy array)
    mock_coeffs = np.zeros(stm.n_coeffs_I, dtype=np.complex128)
    mock_coeffs[0] = 100.0
    
    exc_from_arr = stm.excitation_from_array(mock_coeffs)
    power_arr = stm.radiated_power(exc_from_arr)
    print(f"Power via Array Excitation (first 5 freqs, dB): {np.round(power_arr['total_db'][:5], 2)}")

    # 3. Test Pressure Field Projection
    if stm.gammaI_points.size > 0:
        # Create a mock pressure field on GammaI (e.g. static pressure of 100 Pa)
        mock_pressure = np.full(stm.gammaI_points.shape[0], 100.0 + 0j)
        exc_from_p = stm.excitation_from_pressure(mock_pressure)
        power_p = stm.radiated_power(exc_from_p)
        print(f"Power via Least Squares Projection (first 5 freqs, dB): {np.round(power_p['total_db'][:5], 2)}")
