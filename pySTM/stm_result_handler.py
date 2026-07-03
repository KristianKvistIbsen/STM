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

def _import_make_subplots():
    """Lazily import make_subplots."""
    try:
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required. Install it with 'pip install plotly'.") from exc
    return make_subplots

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

    # -------------------------------------------------------------- synthesis
    def synthesize_coeffs(self, excitation):
        """Output surface-velocity SH coefficients, shape (n_coeffs_O, n_freq)."""
        exc = np.asarray(excitation, dtype=np.complex128)
        
        # Broadcast 1D excitations automatically for backward compatibility
        if exc.ndim == 1:
            if exc.shape[0] != self.n_coeffs_I:
                raise ValueError(f"1D excitation length must be {self.n_coeffs_I}, got {exc.shape[0]}")
            exc = np.tile(exc[:, np.newaxis], (1, self.n_frequencies))
        elif exc.ndim == 2:
            if exc.shape != (self.n_coeffs_I, self.n_frequencies):
                raise ValueError(f"2D excitation shape must be ({self.n_coeffs_I}, {self.n_frequencies}), got {exc.shape}")
        else:
            raise ValueError("Excitation must be 1D or 2D.")

        # Tensor Contraction:
        # STM is (I, O, F)  -> 'ijk'
        # exc is (I, F)     -> 'ik'
        # Out is (O, F)     -> 'jk'
        return np.einsum("ijk,ik->jk", self.STM, exc)

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
    # (Plotly visualization methods remain structurally identical, passing exc downward)
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
            group = f"l{label.split('_')[1]}" if label.startswith("Y_") else "basis"
            fig.add_trace(go.Scatter(x=self.frequencies, y=err[i, :], mode="lines", name=label, legendgroup=group))
            
        if not self.has_error_data:
            fig.add_annotation(text="Error data is all zero - regenerate the STM with the updated generator", xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False)
        fig.update_layout(title=title or ("Relative fit error" if use_relative else "Absolute fit error"), xaxis_title="Frequency (Hz)", yaxis_title="Relative error" if use_relative else "Absolute error", yaxis_type="log", template="plotly_white")
        return fig
    
    def figure_pressure_reconstruction(self, p_target, p_recon, error_percent, freq_index=0, part="abs"):
        """Generate a side-by-side 3D point cloud comparing original and reconstructed pressures."""
        go = _import_go()
        make_subplots = _import_make_subplots()
        
        pts = self.gammaI_points
        if pts.size == 0:
            raise ValueError("No internal mesh geometry available for plotting.")
            
        # Handle 1D (static) vs 2D (frequency-dependent) arrays
        p_t = p_target[:, freq_index] if p_target.ndim == 2 else p_target
        p_r = p_recon[:, freq_index] if p_recon.ndim == 2 else p_recon
        freq_label = f" @ {self.frequencies[freq_index]:.1f} Hz" if p_target.ndim == 2 else " (Static)"
            
        val_t = self._scalar_part(p_t, part)
        val_r = self._scalar_part(p_r, part)
        
        # Lock the color scale so visually identical colors mean identical values
        cmin = min(np.min(val_t), np.min(val_r))
        cmax = max(np.max(val_t), np.max(val_r))
        
        fig = make_subplots(
            rows=1, cols=2, 
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=(f"Original Mapped CFD{freq_label}", f"SVD Basis Reconstruction (Error: {error_percent:.1f}%)")
        )
        
        # Original Pressure
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=val_t, colorscale='Turbo', cmin=cmin, cmax=cmax, showscale=False),
            name="Original"
        ), row=1, col=1)
        
        # Reconstructed Pressure
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=val_r, colorscale='Turbo', cmin=cmin, cmax=cmax, colorbar=dict(title=f"Pressure ({part})")),
            name="Reconstructed"
        ), row=1, col=2)
        
        fig.update_layout(
            title=f"Internal Pressure Field Reconstruction Diagnostic ({part})",
            scene=dict(aspectmode='data'),
            scene2=dict(aspectmode='data'),
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=50) # Tighter margins for better 3D viewing
        )
        return fig
