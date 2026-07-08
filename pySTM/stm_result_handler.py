from __future__ import annotations
import ast
import numpy as np
from scipy.special import spherical_jn, spherical_yn
import pySTM

# Physical defaults (air)
RHO_AIR = 1.3
C_AIR = 343.0
POWER_REF = 1e-12
VTK_TRIANGLE = 5


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

        self.STM = np.asarray(d["STM"])
        self.G = np.asarray(rd["G"])
        self.frequencies = np.real(np.asarray(rd["frequencies"]).ravel()).astype(float)
        self.n_coeffs_I, self.n_coeffs_O, self.n_frequencies = self.STM.shape
        self.lmax_O = int(round(np.sqrt(self.n_coeffs_O))) - 1

        basis_info = rd.get("input_basis", {})
        self.input_basis_type = basis_info.get("type", "spherical_harmonics")

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
        self.sdem_coordinates = np.asarray(ext["SDEM_Coordinates"]) if "SDEM_Coordinates" in ext else None
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
        return (f"STMSynthesizer(basis='{self.input_basis_type}', n_coeffs_I={self.n_coeffs_I}, "
                f"lmax_O={self.lmax_O}, n_frequencies={self.n_frequencies}, file='{self.filepath}')")

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
        exc = np.asarray(excitation, dtype=np.complex128)
        if exc.ndim == 1:
            if exc.shape[0] != self.n_coeffs_I:
                raise ValueError(f"1D excitation length must be {self.n_coeffs_I}, got {exc.shape[0]}")
            exc = np.tile(exc[:, np.newaxis], (1, self.n_frequencies))
        elif exc.ndim == 2:
            if exc.shape != (self.n_coeffs_I, self.n_frequencies):
                raise ValueError(f"2D excitation shape must be ({self.n_coeffs_I}, {self.n_frequencies}), got {exc.shape}")
        else:
            raise ValueError("Excitation must be 1D or 2D.")
        return np.einsum("ijk,ik->jk", self.STM, exc)

    def synthesize_coeffs_2d(self, excitation):
        clm1d = self.synthesize_coeffs(excitation)
        clm = np.zeros((2, self.lmax_O + 1, self.lmax_O + 1, self.n_frequencies), dtype=np.complex128)
        clm[self._out_sign, self._out_l, self._out_absm, :] = clm1d
        return clm

    def synthesize_velocity(self, excitation):
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
            factor = np.where(k > 0, 1.0 / (2.0 * rho * c * k ** 2) * 4.0 * np.pi, 0.0)
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

    # ======================================================================
    #                           VTK / MATPLOTLIB PLOTTING
    # ======================================================================

    def plot_power_spectrum(self, excitation, rho: float = RHO_AIR, c: float = C_AIR, title: str = None):
        """Plot radiated sound power spectrum using Matplotlib."""
        import matplotlib.pyplot as plt

        p = self.radiated_power(excitation, rho=rho, c=c)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.frequencies, p["total_db"], linewidth=2, label="Total Radiated Power")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Sound Power Level (dB re 1 pW)")
        ax.set_title(title or "Radiated Sound Power Spectrum")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_order_power(self, excitation, rho: float = RHO_AIR, c: float = C_AIR,
                         max_order: int = None, title: str = None):
        """Plot radiated power per spherical harmonic degree (heatmap)."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        p = self.radiated_power(excitation, rho=rho, c=c)
        per_order_db = self._to_db(p["per_order"])

        L = self.lmax_O if max_order is None else min(int(max_order), self.lmax_O)
        data = per_order_db[:L + 1, :]

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data, aspect='auto', origin='lower',
                       extent=[self.frequencies[0], self.frequencies[-1], 0, L],
                       cmap='turbo', norm=mcolors.Normalize())
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Spherical Harmonic Degree l")
        ax.set_title(title or "Radiated Power by Spherical Harmonic Order")
        plt.colorbar(im, ax=ax, label="Power Level (dB re 1 pW)")
        plt.tight_layout()
        plt.show()

    def plot_surface_velocity(self, excitation, freq_index: int = 0, part: str = "real",
                              cmap: str = "turbo", title: str = None):
        """Plot surface normal velocity on external mesh using PyVista (VTK)."""
        import pyvista as pv

        if self._external_points is None or self._external_faces is None:
            raise ValueError("No external mesh geometry available for 3D plotting.")

        field = self.synthesize_velocity(excitation)[:, freq_index]
        scalar = self._scalar_part(field, part)

        mesh = pv.PolyData(self._external_points)
        if self._external_faces is not None:
            mesh.faces = np.hstack([np.full((len(self._external_faces), 1), 3), self._external_faces]).astype(np.int32)

        mesh.point_data[f"Velocity ({part})"] = scalar

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars=f"Velocity ({part})", cmap=cmap, show_edges=False,
                         scalar_bar_args={"title": f"v ({part})"})
        plotter.add_text(f"Surface Normal Velocity ({part}) @ {self.frequencies[freq_index]:.1f} Hz",
                         position="upper_edge", font_size=12)
        plotter.show()

    def plot_error_spectrum(self, use_relative: bool = True, title: str = None):
        """Plot reconstruction error vs frequency using Matplotlib."""
        import matplotlib.pyplot as plt

        err = self.rel_error if use_relative else self.abs_error
        if err is None:
            print("No error data available in the STM file.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(err.shape[0]):
            label = self.input_labels[i]
            ax.semilogy(self.frequencies, err[i, :], label=label, alpha=0.8)

        if not self.has_error_data:
            ax.text(0.5, 0.95, "Error data is all zero — regenerate STM with updated generator",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10, color="red")

        ylabel = "Relative Error" if use_relative else "Absolute Error"
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"{'Relative' if use_relative else 'Absolute'} Fit Error Spectrum")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        plt.tight_layout()
        plt.show()

    def plot_pressure_reconstruction(self, p_target, p_recon, error_percent: float,
                                     freq_index: int = 0, part: str = "abs"):
        """
        Plot original vs reconstructed pressure field + absolute difference using PyVista.
        Shows three separate interactive VTK windows.
        """
        import pyvista as pv

        pts = self.gammaI_points
        if pts.size == 0:
            raise ValueError("No internal mesh geometry available for plotting.")

        # Handle 1D vs 2D input
        p_t = p_target[:, freq_index] if p_target.ndim == 2 else p_target
        p_r = p_recon[:, freq_index] if p_recon.ndim == 2 else p_recon

        val_t = self._scalar_part(p_t, part)
        val_r = self._scalar_part(p_r, part)
        val_diff = np.abs(val_t - val_r)

        freq_label = f" @ {self.frequencies[freq_index]:.1f} Hz" if p_target.ndim == 2 else " (Static)"

        def _make_plotter(data, title, cmap, cmin=None, cmax=None):
            pl = pv.Plotter()
            cloud = pv.PolyData(pts)
            cloud.point_data[title] = data
            pl.add_mesh(cloud, scalars=title, cmap=cmap,
                        clim=(cmin, cmax) if cmin is not None else None,
                        point_size=5, render_points_as_spheres=True)
            pl.add_text(title, position="upper_edge", font_size=12)
            return pl

        cmin = min(np.min(val_t), np.min(val_r))
        cmax = max(np.max(val_t), np.max(val_r))

        # 1. Original
        p1 = _make_plotter(val_t, f"Original CFD{freq_label}", "turbo", cmin, cmax)
        p1.show()

        # 2. Reconstructed
        p2 = _make_plotter(val_r, "SVD Reconstruction", "turbo", cmin, cmax)
        p2.show()

        # 3. Absolute Difference
        p3 = _make_plotter(val_diff, f"Abs Diff (Error: {error_percent:.1f}%)", "Reds")
        p3.show()
        
        
    @staticmethod
    def _surface_impedance_Z(l, ka, rho, c):
        """True modal specific acoustic impedance on the surface of a sphere."""
        # Spherical Bessel (jn) and Neumann (yn) functions
        jn = spherical_jn(l, ka)
        yn = spherical_yn(l, ka)
        h1 = jn + 1j * yn  # Spherical Hankel function of the first kind
        
        # Derivatives
        d_jn = spherical_jn(l, ka, derivative=True)
        d_yn = spherical_yn(l, ka, derivative=True)
        dh1 = d_jn + 1j * d_yn
        
        # True specific acoustic impedance: Z = i * rho * c * (h1 / dh1)
        return 1j * rho * c * h1 / dh1

    def _surface_impedance_table(self, rho, c):
        """Precompute true surface impedance for all l and frequencies."""
        key = (rho, c, "surface")
        if getattr(self, '_Z_surf_cache_key', None) == key and getattr(self, '_Z_surf_table', None) is not None:
            return self._Z_surf_table
            
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
                Z[l, fi] = self._surface_impedance_Z(l, ka, rho, c)
                
        self._Z_surf_table = Z
        self._Z_surf_cache_key = key
        return Z
    
    
# =============================================================================
#     Non-negative intensity calculation
# =============================================================================
    def synthesize_intensity(self, excitation, rho: float = RHO_AIR, c: float = C_AIR, 
                             method: str = "NNI", plot: bool = False, 
                             plot_freq_index: int = 0, cmap: str = "inferno", title: str = None):
        # 1. Get Velocity Coefficients
        c_eta = self.synthesize_coeffs(excitation)  # Shape: (n_coeffs_O, n_frequencies)
        
        # 2. Extract TRUE Surface Impedance (other impedance function is missing the hankel function to compensate for the definition in FourierAcoustics book)
        Z_table = self._surface_impedance_table(rho, c)
        Z_eta = Z_table[self._out_l, :]             # Expand to shape (n_coeffs_O, n_frequencies)

        method = method.upper()
        
        if method == "NNI":
            # True modal radiation resistance at the surface
            R_eta = np.maximum(np.real(Z_eta), 0.0)
            
            # Construct the beta parameter
            beta_eta = c_eta * np.sqrt(R_eta)
            beta_x = self.G @ beta_eta
            
            # Non-Negative Intensity (strictly positive, tracks symmetric structural radiation)
            intensity_data = 0.5 * np.abs(beta_x)**2

        elif method == "SSI":
            # Compute true surface pressure coefficients (p_eta = c_eta * Z_eta)
            p_eta = c_eta * Z_eta
            
            # Reconstruct spatial pressure and velocity fields
            p_x = self.G @ p_eta
            v_x = self.G @ c_eta
            
            # True active surface intensity
            intensity_data = 0.5 * np.real(p_x * np.conj(v_x))

        else:
            raise ValueError("Method must be 'NNI' or 'SSI'")

        # 3. Optional VTK Plotting
        if plot:
            import pyvista as pv

            if self._external_points is None or self._external_faces is None:
                print("Warning: No external mesh geometry available for 3D plotting. Skipping plot.")
            else:
                field = intensity_data[:, plot_freq_index]

                # Construct the physical PyVista mesh
                mesh = pv.PolyData(self._external_points)
                if self._external_faces is not None:
                    mesh.faces = np.hstack([
                        np.full((len(self._external_faces), 1), 3), 
                        self._external_faces
                    ]).astype(np.int32)

                # Apply scalar data
                mesh.point_data[f"Intensity ({method})"] = field

                plotter = pv.Plotter()
                plotter.add_mesh(mesh, scalars=f"Intensity ({method})", cmap=cmap, show_edges=False,
                                 scalar_bar_args={"title": f"Intensity [W/m²]"})
                
                plot_title = title or f"Surface {method} @ {self.frequencies[plot_freq_index]:.1f} Hz"
                plotter.add_text(plot_title, position="upper_edge", font_size=12)
                plotter.show()

        return intensity_data
