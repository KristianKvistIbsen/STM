from __future__ import annotations

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

        self.STM = np.asarray(d["STM"])                       # (n_coeffs_I, n_coeffs_O, n_freq)
        self.G = np.asarray(rd["G"])                          # (n_points, n_coeffs_O)
        self.frequencies = np.real(np.asarray(rd["frequencies"]).ravel()).astype(float)

        self.n_coeffs_I, self.n_coeffs_O, self.n_frequencies = self.STM.shape
        self.lmax_I = int(round(np.sqrt(self.n_coeffs_I))) - 1
        self.lmax_O = int(round(np.sqrt(self.n_coeffs_O))) - 1

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
        """Extract (points, triangle_faces) numpy arrays from a PyVista grid."""
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

    # ------------------------------------------------------------- introspection
    def summary(self) -> dict:
        """Return a small dict of key facts (handy for a GUI status panel)."""
        fmin = float(self.frequencies.min()) if self.n_frequencies else None
        fmax = float(self.frequencies.max()) if self.n_frequencies else None
        return {
            "filepath": self.filepath,
            "lmax_I": self.lmax_I,
            "lmax_O": self.lmax_O,
            "n_coeffs_I": self.n_coeffs_I,
            "n_coeffs_O": self.n_coeffs_O,
            "n_frequencies": self.n_frequencies,
            "frequency_range_hz": (fmin, fmax),
            "n_external_points": (
                int(self._external_points.shape[0]) if self._external_points is not None else None
            ),
            "n_external_faces": (
                int(self._external_faces.shape[0]) if self._external_faces is not None else None
            ),
            "equivalent_radius_m": self.equivalent_radius,
            "has_error_data": self.has_error_data,
            "solution_method": self.metadata.get("user_settings", {}).get("solution_method", "unknown"),
            "creation_date": self.file_info.get("creation_date", ""),
        }

    def __repr__(self):
        return (
            f"STMSynthesizer(lmax_I={self.lmax_I}, lmax_O={self.lmax_O}, "
            f"n_frequencies={self.n_frequencies}, file='{self.filepath}')"
        )

    @property
    def has_error_data(self) -> bool:
        """True if the file carries non-zero fit-error data."""
        return self.rel_error is not None and bool(np.any(self.rel_error != 0))

    @property
    def external_points(self):
        return self._external_points

    @property
    def external_faces(self):
        return self._external_faces

    # ------------------------------------------------------------- index helpers
    @staticmethod
    def lm_to_index(l: int, m: int) -> int:
        return l * l + l + m

    @staticmethod
    def index_to_lm(idx: int):
        l = int(np.floor(np.sqrt(idx)))
        return l, idx - l * l - l

    @property
    def input_harmonic_labels(self):
        """['Y_0_0', 'Y_1_-1', ...] for every input excitation coefficient."""
        return [f"Y_{l}_{m}" for l, m in (self.index_to_lm(i) for i in range(self.n_coeffs_I))]

    def nearest_frequency_index(self, freq_hz: float) -> int:
        """Index of the stored frequency closest to ``freq_hz``."""
        return int(np.argmin(np.abs(self.frequencies - freq_hz)))

    # -------------------------------------------------------- excitation builders
    def zero_excitation(self):
        return np.zeros(self.n_coeffs_I, dtype=np.complex128)

    def excitation_single(self, l: int, m: int, amplitude: complex = 1.0):
        """Excitation vector with a single interior harmonic ``Y_l_m`` active."""
        e = self.zero_excitation()
        e[self.lm_to_index(l, m)] = amplitude
        return e

    def excitation_from_dict(self, coeffs: dict):
        """Build an excitation vector from a dict.

        Keys may be ``(l, m)`` tuples or flat integer indices; values are complex
        amplitudes. Example: ``{(0, 0): 1.0, (1, 1): 1 - 1j}``.
        """
        e = self.zero_excitation()
        for key, val in coeffs.items():
            if isinstance(key, (tuple, list)):
                idx = self.lm_to_index(int(key[0]), int(key[1]))
            else:
                idx = int(key)
            if not 0 <= idx < self.n_coeffs_I:
                raise IndexError(f"excitation index {idx} out of range [0, {self.n_coeffs_I})")
            e[idx] = val
        return e

    def _as_excitation(self, excitation):
        e = np.asarray(excitation, dtype=np.complex128).ravel()
        if e.shape[0] != self.n_coeffs_I:
            raise ValueError(
                f"excitation must have length n_coeffs_I={self.n_coeffs_I}, got {e.shape[0]}"
            )
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
        """Surface normal velocity on the external mesh, shape (n_external_points, n_freq).

        Point ordering matches :pyattr:`external_points` / :pyattr:`external_grid`.
        """
        return self.G @ self.synthesize_coeffs(excitation)

    # --------------------------------------------------------------- radiated power
    @staticmethod
    def _impedance_Z(l, ka, rho, c):
        # Follows the definition used in the generator/original synthesiser (see
        # E. G. Williams, Fourier Acoustics, eq. 6.93): a modified modal impedance.
        d_jn = spherical_jn(l, ka, derivative=True)
        d_yn = spherical_yn(l, ka, derivative=True)
        dh1 = d_jn + 1j * d_yn
        return 1j * rho * c / dh1

    def _impedance_table(self, rho, c):
        """(lmax_O+1, n_freq) modal-impedance table, cached (independent of excitation)."""
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
        """Time-averaged radiated sound power vs frequency.

        Returns a dict with:
        ``total`` (n_freq,), ``total_db`` (n_freq,), ``per_order`` (lmax_O+1, n_freq)
        and ``frequencies`` (n_freq,). Each spherical-harmonic degree is summed exactly
        once (corrects the running-partial-sum in the original script).
        """
        clm = self.synthesize_coeffs_2d(excitation)          # (2, L+1, L+1, n_freq)
        Z = self._impedance_table(rho, c)                    # (L+1, n_freq)

        # |v_lm|^2 summed over sign(+/-) and |m| for every degree l:
        v_abs_sqr = np.sum(np.abs(clm) ** 2, axis=(0, 2))    # (L+1, n_freq)
        p_abs_sqr = v_abs_sqr * np.abs(Z) ** 2               # |p_lm|^2 grouped by degree

        k = 2.0 * np.pi * self.frequencies / c
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(k > 0, 1.0 / (2.0 * rho * c * k ** 2) / np.sqrt(4.0 * np.pi), 0.0)

        per_order = p_abs_sqr * factor[np.newaxis, :]        # (L+1, n_freq)
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
        """Total radiated power (dB) vs frequency as a Plotly figure."""
        go = _import_go()
        p = self.radiated_power(excitation, rho=rho, c=c)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.frequencies, y=p["total_db"], mode="lines", name="Total"))
        fig.update_layout(
            title=title or "Radiated sound power",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB re 1 pW)",
            template="plotly_white",
        )
        return fig

    def figure_order_power(self, excitation, rho: float = RHO_AIR, c: float = C_AIR,
                           max_order=None, title=None):
        """Radiated power per spherical-harmonic degree (dB) as a frequency/degree heatmap."""
        go = _import_go()
        p = self.radiated_power(excitation, rho=rho, c=c)
        per_order_db = self._to_db(p["per_order"])
        L = self.lmax_O if max_order is None else min(int(max_order), self.lmax_O)
        fig = go.Figure(
            data=go.Heatmap(
                x=self.frequencies,
                y=np.arange(L + 1),
                z=per_order_db[: L + 1, :],
                colorscale="Turbo",
                colorbar=dict(title="dB"),
            )
        )
        fig.update_layout(
            title=title or "Radiated power by spherical-harmonic degree",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Degree l",
            template="plotly_white",
        )
        return fig

    def figure_surface_velocity(self, excitation, freq_index: int = 0, part: str = "real",
                                colorscale: str = "Turbo", title=None):
        """3D surface normal-velocity field on the external mesh as a Plotly Mesh3d figure."""
        go = _import_go()
        if self._external_points is None or self._external_faces is None:
            raise ValueError("No external mesh geometry available for 3D plotting.")
        field = self.synthesize_velocity(excitation)[:, freq_index]
        scalar = self._scalar_part(field, part)
        pts, faces = self._external_points, self._external_faces
        fig = go.Figure(
            data=go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=scalar, intensitymode="vertex",
                colorscale=colorscale, colorbar=dict(title=f"v ({part})"),
                flatshading=True, showscale=True,
            )
        )
        freq = self.frequencies[freq_index]
        fig.update_layout(
            title=title or f"Surface normal velocity ({part}) @ {freq:.1f} Hz",
            scene=dict(aspectmode="data"),
            template="plotly_white",
        )
        return fig

    def figure_error_spectrum(self, use_relative: bool = True, title=None):
        """Fit-error vs frequency for each interior harmonic (log y) as a Plotly figure."""
        go = _import_go()
        err = self.rel_error if use_relative else self.abs_error
        fig = go.Figure()
        if err is None:
            fig.add_annotation(text="No error data in file", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig
        err = np.asarray(err)
        for i in range(err.shape[0]):
            l, m = self.index_to_lm(i)
            fig.add_trace(
                go.Scatter(x=self.frequencies, y=err[i, :], mode="lines",
                           name=f"Y_{l}_{m}", legendgroup=f"l{l}")
            )
        if not self.has_error_data:
            fig.add_annotation(
                text="Error data is all zero - regenerate the STM with the updated generator",
                xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False,
            )
        fig.update_layout(
            title=title or ("Relative fit error" if use_relative else "Absolute fit error"),
            xaxis_title="Frequency (Hz)",
            yaxis_title="Relative error" if use_relative else "Absolute error",
            yaxis_type="log",
            template="plotly_white",
        )
        return fig


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"C:/01_gitrepos/STM/STM.h5"
    stm = STMSynthesizer(path)
    print(stm)
    for key, value in stm.summary().items():
        print(f"  {key}: {value}")

    exc = stm.excitation_from_dict({(0, 0): 1.0, (1, 1): 1 - 1j})
    power = stm.radiated_power(exc)
    print(f"Total power (first 5 freqs, dB): {np.round(power['total_db'][:5], 2)}")
