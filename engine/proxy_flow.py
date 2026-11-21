"""
Fast 3D Reservoir Flow Proxy Simulator.
Simulates multiphase reservoir dynamics (pressure diffusion and water-oil displacement)
to generate realistic well production profiles (WOPR, WWPR, WWCT, WBHP)
directly in Python without needing external OPM Flow binaries.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class ReservoirProxySimulator:
    """
    Lightweight 3D 2-phase (Oil-Water) flow solver based on
    IMPES (Implicit Pressure, Explicit Saturation) formulation
    and Peaceman well models.
    """

    DEFAULT_WELLS = [
        {"name": "G1", "type": "PROD", "i": 14, "j": 2, "k1": 1, "k2": 4, "target_rate": 35.0},
        {"name": "G2", "type": "PROD", "i": 12, "j": 4, "k1": 3, "k2": 5, "target_rate": 40.0},
        {"name": "G3", "type": "PROD", "i": 14, "j": 6, "k1": 2, "k2": 5, "target_rate": 30.0},
        {"name": "G4", "type": "PROD", "i": 10, "j": 2, "k1": 3, "k2": 5, "target_rate": 25.0},
        {"name": "P1:01", "type": "PROD", "i": 9, "j": 5, "k1": 1, "k2": 3, "target_rate": 45.0},
        {"name": "P2", "type": "PROD", "i": 12, "j": 8, "k1": 1, "k2": 4, "target_rate": 50.0},
        {"name": "P3", "type": "PROD", "i": 8, "j": 8, "k1": 1, "k2": 3, "target_rate": 35.0},
        {"name": "P4", "type": "PROD", "i": 6, "j": 2, "k1": 2, "k2": 4, "target_rate": 30.0},
        {"name": "P5", "type": "PROD", "i": 5, "j": 5, "k1": 1, "k2": 4, "target_rate": 40.0},
        {"name": "P6", "type": "PROD", "i": 4, "j": 2, "k1": 1, "k2": 3, "target_rate": 30.0},
        {"name": "D1", "type": "INJ", "i": 3, "j": 5, "k1": 1, "k2": 6, "target_rate": 120.0},
        {"name": "D2", "type": "INJ", "i": 5, "j": 11, "k1": 1, "k2": 6, "target_rate": 140.0},
        {"name": "D3", "type": "INJ", "i": 13, "j": 11, "k1": 1, "k2": 6, "target_rate": 130.0},
    ]

    def __init__(
        self,
        dimens: Tuple[int, int, int] = (15, 15, 6),
        dx: float = 50.0,  # meters
        dy: float = 50.0,
        dz: float = 5.0,
        initial_pressure: float = 250.0,  # bars
        water_viscosity: float = 0.5,     # cP
        oil_viscosity: float = 2.0,       # cP
        sw_initial: float = 0.20,         # connate water saturation
        sor: float = 0.25                 # residual oil saturation
    ):
        self.nx, self.ny, self.nz = dimens
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.n_cells = self.nx * self.ny * self.nz
        self.p_init = initial_pressure
        self.mu_w = water_viscosity
        self.mu_o = oil_viscosity
        self.sw_init = sw_initial
        self.sor = sor

    def _index(self, k: int, j: int, i: int) -> int:
        """Flattens 3D index (k, j, i) to 1D."""
        return (k * self.ny + j) * self.nx + i

    def simulate(
        self,
        poro: np.ndarray,      # (Nz, Ny, Nx)
        permx: np.ndarray,     # (Nz, Ny, Nx) in mD
        permy: np.ndarray,
        permz: np.ndarray,
        timesteps_days: Optional[List[float]] = None,
        wells: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Runs dynamic multiphase simulation.
        Returns time series for each well: WOPR, WWPR, WWCT, WBHP, and Field Totals.
        """
        nz, ny, nx = self.nz, self.ny, self.nx
        if timesteps_days is None:
            # 24 monthly steps (2 years)
            timesteps_days = [float(30 * m) for m in range(1, 25)]

        wells_config = wells or self.DEFAULT_WELLS

        # Initial fields
        sw = np.full((nz, ny, nx), self.sw_init, dtype=np.float64)
        p = np.full((nz, ny, nx), self.p_init, dtype=np.float64)

        # Transmissibility conversion factor (mD to m3/day/bar)
        conv = 0.008527

        # Results dictionary
        results = {
            "time_days": timesteps_days,
            "wells": {},
            "field": {
                "FOPR": [],
                "FWPR": [],
                "FWCT": [],
                "FPR": []
            }
        }
        for w in wells_config:
            results["wells"][w["name"]] = {
                "WOPR": [],
                "WWPR": [],
                "WWCT": [],
                "WBHP": []
            }

        prev_t = 0.0

        for t_day in timesteps_days:
            dt = t_day - prev_t
            prev_t = t_day

            # 1. Compute relative permeabilities
            # Brooks-Corey type relative perm
            sw_eff = np.clip((sw - self.sw_init) / (1.0 - self.sw_init - self.sor + 1e-6), 0.0, 1.0)
            krw = sw_eff ** 2.0
            kro = (1.0 - sw_eff) ** 2.0

            total_mobility = (krw / self.mu_w) + (kro / self.mu_o)
            fw_fraction = (krw / self.mu_w) / (total_mobility + 1e-9)

            # 2. Build Pressure Matrix
            # A * P = q
            entries = []
            rows = []
            cols = []
            rhs = np.zeros(self.n_cells)

            # Transmissibility calculation
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        idx = self._index(k, j, i)
                        diag = 0.0

                        # X neighbor
                        if i < nx - 1:
                            idx_adj = self._index(k, j, i + 1)
                            kx_harm = 2.0 / (1.0 / max(permx[k, j, i], 0.01) + 1.0 / max(permx[k, j, i + 1], 0.01))
                            tx = conv * kx_harm * (self.dy * self.dz) / self.dx * 0.5 * (total_mobility[k, j, i] + total_mobility[k, j, i + 1])
                            rows.extend([idx, idx_adj])
                            cols.extend([idx_adj, idx])
                            entries.extend([-tx, -tx])
                            diag += tx

                        # Y neighbor
                        if j < ny - 1:
                            idx_adj = self._index(k, j + 1, i)
                            ky_harm = 2.0 / (1.0 / max(permy[k, j, i], 0.01) + 1.0 / max(permy[k, j + 1, i], 0.01))
                            ty = conv * ky_harm * (self.dx * self.dz) / self.dy * 0.5 * (total_mobility[k, j, i] + total_mobility[k, j + 1, i])
                            rows.extend([idx, idx_adj])
                            cols.extend([idx_adj, idx])
                            entries.extend([-ty, -ty])
                            diag += ty

                        # Z neighbor
                        if k < nz - 1:
                            idx_adj = self._index(k + 1, j, i)
                            kz_harm = 2.0 / (1.0 / max(permz[k, j, i], 0.01) + 1.0 / max(permz[k + 1, j, i], 0.01))
                            tz = conv * kz_harm * (self.dx * self.dy) / self.dz * 0.5 * (total_mobility[k, j, i] + total_mobility[k + 1, j, i])
                            rows.extend([idx, idx_adj])
                            cols.extend([idx_adj, idx])
                            entries.extend([-tz, -tz])
                            diag += tz

                        # Diagonal
                        rows.append(idx)
                        cols.append(idx)
                        entries.append(diag + 1e-4)

            # Add Well source/sink terms
            well_water_rates = {}
            well_oil_rates = {}
            well_bhps = {}

            for w in wells_config:
                wi = int(np.clip(w["i"] - 1, 0, nx - 1))
                wj = int(np.clip(w["j"] - 1, 0, ny - 1))
                k1 = int(np.clip(w.get("k1", 1) - 1, 0, nz - 1))
                k2 = int(np.clip(w.get("k2", nz) - 1, 0, nz - 1))
                target = w.get("target_rate", 30.0)

                # Peaceman Well Index (WI)
                n_layers = max(1, k2 - k1 + 1)
                layer_rate = target / n_layers

                total_q_oil = 0.0
                total_q_water = 0.0
                bhp_samples = []

                for k in range(k1, k2 + 1):
                    idx = self._index(k, wj, wi)
                    kh = np.sqrt(permx[k, wj, wi] * permy[k, wj, wi])
                    wi_factor = conv * 2.0 * np.pi * kh * self.dz / (np.log(0.14 * np.sqrt(self.dx**2 + self.dy**2) / 0.1) + 1e-3)

                    if w["type"] == "INJ":
                        # Injection well adds fluid
                        rhs[idx] += layer_rate
                        total_q_water += layer_rate
                        p_well = p[k, wj, wi] + layer_rate / max(wi_factor * (krw[k, wj, wi] / self.mu_w), 1e-4)
                        bhp_samples.append(p_well)
                    else:
                        # Production well removes fluid
                        rhs[idx] -= layer_rate
                        fw_cell = fw_fraction[k, wj, wi]
                        total_q_water += layer_rate * fw_cell
                        total_q_oil += layer_rate * (1.0 - fw_cell)
                        p_well = max(50.0, p[k, wj, wi] - layer_rate / max(wi_factor * total_mobility[k, wj, wi], 1e-4))
                        bhp_samples.append(p_well)

                well_oil_rates[w["name"]] = total_q_oil
                well_water_rates[w["name"]] = total_q_water
                well_bhps[w["name"]] = float(np.mean(bhp_samples)) if bhp_samples else 150.0

            # Solve for updated pressure
            A = sp.csr_matrix((entries, (rows, cols)), shape=(self.n_cells, self.n_cells))
            try:
                p_vec = spla.spsolve(A, rhs)
                p = p_vec.reshape((nz, ny, nx))
            except Exception:
                pass  # Keep previous pressure on singular step

            # 3. Saturation update via water front advance
            cell_pv = (self.dx * self.dy * self.dz) * np.maximum(poro, 0.01)
            for w in wells_config:
                wi = int(np.clip(w["i"] - 1, 0, nx - 1))
                wj = int(np.clip(w["j"] - 1, 0, ny - 1))
                k1 = int(np.clip(w.get("k1", 1) - 1, 0, nz - 1))
                k2 = int(np.clip(w.get("k2", nz) - 1, 0, nz - 1))
                n_layers = max(1, k2 - k1 + 1)
                layer_rate = w.get("target_rate", 30.0) / n_layers

                if w["type"] == "INJ":
                    # Increase water saturation around injection wells
                    for k in range(k1, k2 + 1):
                        d_sw = (layer_rate * dt) / cell_pv[k, wj, wi]
                        sw[k, wj, wi] = min(1.0 - self.sor, sw[k, wj, wi] + d_sw * 0.4)
                        # Disperse to adjacent cells
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = wi + di, wj + dj
                            if 0 <= ni < nx and 0 <= nj < ny:
                                sw[k, nj, ni] = min(1.0 - self.sor, sw[k, nj, ni] + d_sw * 0.15)

            # Record step metrics
            f_oil = sum(well_oil_rates.values())
            f_water = sum(well_water_rates.values())
            f_wct = f_water / (f_oil + f_water + 1e-9)

            results["field"]["FOPR"].append(round(f_oil, 2))
            results["field"]["FWPR"].append(round(f_water, 2))
            results["field"]["FWCT"].append(round(f_wct, 4))
            results["field"]["FPR"].append(round(float(np.mean(p)), 2))

            for w in wells_config:
                name = w["name"]
                qo = well_oil_rates.get(name, 0.0)
                qw = well_water_rates.get(name, 0.0)
                wct = qw / (qo + qw + 1e-9) if qo + qw > 0 else 0.0
                results["wells"][name]["WOPR"].append(round(qo, 2))
                results["wells"][name]["WWPR"].append(round(qw, 2))
                results["wells"][name]["WWCT"].append(round(wct, 4))
                results["wells"][name]["WBHP"].append(round(well_bhps.get(name, 150.0), 2))

        return results
