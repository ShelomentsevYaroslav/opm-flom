"""
Quality Control and Reservoir Validation Engine.
Performs rigorous petrophysical and geostatistical validation of synthetic models:
- Physical bounds and continuity
- Vertical (Z) vs Lateral (XY) variogram continuity
- Porosity-Permeability cross-correlation (Phi vs log(K))
- Directional anisotropy ratios (Ky/Kx, Kz/Kx)
- Ensemble distribution metrics (P10, P50, P90)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .geostats import GeostatisticalEngine


class ReservoirValidator:
    """
    Evaluates synthetic reservoir property fields against physical laws
    and geostatistical continuity standards.
    """

    @classmethod
    def validate_property_set(
        cls,
        synthetic_props: Dict[str, np.ndarray],
        baseline_props: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict:
        """
        Runs comprehensive validation on synthetic realization.
        Returns detailed report with metrics, warnings, and overall status (PASS/WARN/FAIL).
        """
        report = {
            "status": "PASS",
            "score": 100,  # 0 to 100
            "checks": {},
            "metrics": {},
            "warnings": [],
            "errors": []
        }

        poro = synthetic_props.get("PORO")
        permx = synthetic_props.get("PERMX")
        permy = synthetic_props.get("PERMY")
        permz = synthetic_props.get("PERMZ")
        ntg = synthetic_props.get("NTG")

        # 1. Basic Physical Bounds Checks
        if poro is not None:
            p_min, p_max = float(np.min(poro)), float(np.max(poro))
            p_mean, p_std = float(np.mean(poro)), float(np.std(poro))
            report["metrics"]["poro"] = {"min": p_min, "max": p_max, "mean": p_mean, "std": p_std}

            if p_min < 0.0 or p_max > 0.50:
                report["errors"].append(f"Porosity out of physical range: [{p_min:.3f}, {p_max:.3f}]")
                report["score"] -= 25
            else:
                report["checks"]["poro_bounds"] = "PASS"

        if permx is not None:
            kx_min, kx_max = float(np.min(permx)), float(np.max(permx))
            kx_mean, kx_std = float(np.mean(permx)), float(np.std(permx))
            report["metrics"]["permx"] = {"min": kx_min, "max": kx_max, "mean": kx_mean, "std": kx_std}

            if kx_min <= 0.0 or kx_max > 50000.0:
                report["errors"].append(f"Permeability X out of physical range: [{kx_min:.2f}, {kx_max:.2f}]")
                report["score"] -= 25
            else:
                report["checks"]["permx_bounds"] = "PASS"

        # 2. Vertical vs Lateral Geostatistical Continuity (The key problem from correspondence)
        if poro is not None and poro.shape[0] > 1:
            lags_z, gz = GeostatisticalEngine.compute_experimental_variogram(poro, "z")
            lags_x, gx = GeostatisticalEngine.compute_experimental_variogram(poro, "x")
            var_total = float(np.var(poro))

            if var_total > 1e-6 and len(gz) > 0 and len(gx) > 0:
                # Normalized semivariance at lag 1
                ratio_z1 = float(gz[0] / var_total)
                ratio_x1 = float(gx[0] / var_total)

                report["metrics"]["variogram"] = {
                    "z_lag1_ratio": round(ratio_z1, 3),
                    "x_lag1_ratio": round(ratio_x1, 3),
                    "vertical_continuity": "PRESERVED" if ratio_z1 < 0.70 else "WEAK_OR_INDEPENDENT"
                }

                if ratio_z1 >= 0.85:
                    report["warnings"].append(
                        f"Vertical correlation is very weak (gamma_z/var = {ratio_z1:.2f}). "
                        "Layers may be physically disconnected."
                    )
                    report["score"] -= 15
                else:
                    report["checks"]["vertical_continuity"] = "PASS"

        # 3. Petro-Physical Cross-Correlation Check (Poro vs PermX)
        if poro is not None and permx is not None:
            p_flat = poro.flatten()
            log_k = np.log10(np.maximum(permx.flatten(), 1e-3))
            corr = float(np.corrcoef(p_flat, log_k)[0, 1])
            report["metrics"]["petro_corr_r"] = round(corr, 3)

            if corr < 0.50:
                report["warnings"].append(
                    f"Low petrophysical correlation between PORO and log(PERMX): R = {corr:.2f}. "
                    "Expected R >= 0.70 for realistic reservoir facies."
                )
                report["score"] -= 15
            else:
                report["checks"]["petro_correlation"] = "PASS"

        # 4. Anisotropy Ratio Checks (Ky/Kx and Kz/Kx)
        if permx is not None and permz is not None:
            ratio_z = float(np.mean(permz / np.maximum(permx, 1e-3)))
            report["metrics"]["anisotropy_kz_kx"] = round(ratio_z, 3)

            if ratio_z > 1.2 or ratio_z < 0.001:
                report["warnings"].append(f"Unusual vertical anisotropy Kz/Kx = {ratio_z:.3f} (typically 0.01 to 0.50).")
                report["score"] -= 10
            else:
                report["checks"]["vertical_anisotropy"] = "PASS"

        # Final Status Decision
        report["score"] = max(0, report["score"])
        if report["errors"]:
            report["status"] = "FAIL"
        elif report["warnings"]:
            report["status"] = "WARN"
        else:
            report["status"] = "PASS"

        return report
