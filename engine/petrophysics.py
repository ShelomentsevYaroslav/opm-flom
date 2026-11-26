"""
Petrophysical Co-Simulation Module.
Ensures physically consistent relationships between Porosity (PORO),
Permeability (PERMX, PERMY, PERMZ), and Net-to-Gross (NTG).
"""

from typing import Dict, Optional, Tuple
import numpy as np


class PetrophysicsEngine:
    """
    Calibrates and co-simulates reservoir petrophysical properties
    preserving log-linear trends, cross-correlation coefficients,
    and directional anisotropy (Kx, Ky, Kz).
    """

    @classmethod
    def calibrate_from_model(cls, properties: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extracts empirical petrophysical relationships from a baseline reservoir model:
        log10(Kx) = slope * PORO + intercept + residual_std
        anisotropy_y = mean(PERMY / PERMX)
        anisotropy_z = mean(PERMZ / PERMX)
        """
        poro = properties.get("PORO")
        permx = properties.get("PERMX")
        permy = properties.get("PERMY")
        permz = properties.get("PERMZ")
        ntg = properties.get("NTG")

        calib = {
            "slope": 5.0,
            "intercept": 1.0,
            "residual_std": 0.15,
            "anisotropy_y": 0.8,
            "anisotropy_z": 0.2,
            "ntg_mean": 0.85,
            "ntg_poro_corr": 0.6
        }

        if poro is not None and permx is not None:
            p_flat = poro.flatten()
            k_flat = np.maximum(permx.flatten(), 1e-3)
            log_k = np.log10(k_flat)

            # Linear regression: log10(K) ~ slope * P + intercept
            if np.std(p_flat) > 1e-5:
                cov = np.cov(p_flat, log_k)
                slope = cov[0, 1] / cov[0, 0]
                intercept = float(np.mean(log_k) - slope * np.mean(p_flat))
                pred = slope * p_flat + intercept
                residual_std = float(np.std(log_k - pred))

                calib["slope"] = float(slope)
                calib["intercept"] = float(intercept)
                calib["residual_std"] = max(0.05, residual_std)

            if permy is not None:
                calib["anisotropy_y"] = float(np.clip(np.mean(permy / np.maximum(permx, 1e-3)), 0.1, 2.0))
            if permz is not None:
                calib["anisotropy_z"] = float(np.clip(np.mean(permz / np.maximum(permx, 1e-3)), 0.001, 1.0))
            if ntg is not None:
                calib["ntg_mean"] = float(np.mean(ntg))

        return calib

    @classmethod
    def cosimulate_properties(
        cls,
        synthetic_poro: np.ndarray,
        calibration: Optional[Dict[str, float]] = None,
        corr_factor: float = 0.90,       # correlation coefficient (0.5 to 0.99)
        aniso_z: Optional[float] = None, # custom vertical anisotropy ratio Kz/Kh
        min_perm: float = 0.01,
        max_perm: float = 10000.0,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Co-simulates PERMX, PERMY, PERMZ, and NTG conditioned on synthetic PORO.
        """
        if seed is not None:
            np.random.seed(seed)

        calib = calibration or {
            "slope": 6.0,
            "intercept": 0.8,
            "residual_std": 0.2,
            "anisotropy_y": 0.7,
            "anisotropy_z": 0.2,
            "ntg_mean": 0.85
        }

        slope = calib.get("slope", 6.0)
        intercept = calib.get("intercept", 0.8)
        base_std = calib.get("residual_std", 0.2)
        aniso_y = calib.get("anisotropy_y", 0.7)
        ratio_z = aniso_z if aniso_z is not None else calib.get("anisotropy_z", 0.2)

        # 1. Generate correlated log10(PERMX)
        # log10(Kx) = slope * Phi + intercept + random_scatter
        shape = synthetic_poro.shape
        # Deterministic component
        mean_log_k = slope * synthetic_poro + intercept

        # Match the exact requested Pearson correlation R between PORO and log10(PERMX):
        # R = signal_std / sqrt(signal_std^2 + scatter_std^2)
        signal_std = max(1e-4, float(np.std(mean_log_k)))
        r_clamped = float(np.clip(corr_factor, 0.10, 0.999))
        scatter_std = signal_std * np.sqrt(1.0 - r_clamped ** 2) / r_clamped
        noise_k = np.random.normal(0.0, scatter_std, size=shape)

        log_kx = mean_log_k + noise_k
        permx = 10.0 ** log_kx
        permx = np.clip(permx, min_perm, max_perm)

        # 2. Horizontal permeability PERMY (with mild random variance)
        noise_y = np.random.normal(0.0, 0.05, size=shape)
        permy = permx * aniso_y * np.exp(noise_y)
        permy = np.clip(permy, min_perm, max_perm)

        # 3. Vertical permeability PERMZ (Kz/Kh anisotropy with geological layering)
        noise_z = np.random.normal(0.0, 0.08, size=shape)
        permz = permx * ratio_z * np.exp(noise_z)
        permz = np.clip(permz, min_perm * 0.1, max_perm)

        # 4. Net-to-Gross (NTG)
        # NTG is higher for higher porosity, bounded in [0.05, 1.0]
        poro_norm = (synthetic_poro - synthetic_poro.min()) / (synthetic_poro.max() - synthetic_poro.min() + 1e-6)
        ntg = np.clip(0.3 + 0.65 * poro_norm + np.random.normal(0.0, 0.03, size=shape), 0.05, 1.0)

        return {
            "PORO": synthetic_poro,
            "PERMX": permx,
            "PERMY": permy,
            "PERMZ": permz,
            "NTG": ntg
        }
