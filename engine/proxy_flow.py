"""
Fast 3D Reservoir Flow Proxy Simulator.
Simulates multiphase reservoir dynamics (pressure diffusion and water-oil displacement)
using a vectorized 3D IMPES (Implicit Pressure, Explicit Saturation) solver
and Peaceman well models, directly in Python without external binaries.
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

    def _relperm(self, sw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes Brooks-Corey relative permeabilities, total mobility, and water fractional flow."""
        sw_eff = np.clip((sw - self.sw_init) / (1.0 - self.sw_init - self.sor + 1e-6), 0.0, 1.0)
        krw = 0.65 * (sw_eff ** 2.0)
        kro = 0.85 * ((1.0 - sw_eff) ** 2.0)
        lam_w = krw / self.mu_w
        lam_o = kro / self.mu_o
        total_mob = lam_w + lam_o + 1e-9
        fw = lam_w / total_mob
        return krw, kro, total_mob, fw

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
        Runs dynamic multiphase 3D IMPES simulation.
        Returns time series for each well: WOPR, WWPR, WWCT, WBHP, and Field Totals.
        """
        nz, ny, nx = self.nz, self.ny, self.nx
        if timesteps_days is None:
            timesteps_days = [float(30 * m) for m in range(1, 25)]

        wells_config = wells or self.DEFAULT_WELLS

        sw = np.full((nz, ny, nx), self.sw_init, dtype=np.float64)
        p = np.full((nz, ny, nx), self.p_init, dtype=np.float64)

        conv = 0.008527  # Darcy unit conversion (mD*m / (cP*bar) -> m3/day)
        kx_safe = np.maximum(permx, 0.01)
        ky_safe = np.maximum(permy, 0.01)
        kz_safe = np.maximum(permz, 0.005)
        cell_pv = (self.dx * self.dy * self.dz) * np.maximum(poro, 0.01)

        # Static geometric transmissibilities along X, Y, Z faces
        geom_tx = conv * (2.0 / (1.0 / kx_safe[:, :, :-1] + 1.0 / kx_safe[:, :, 1:])) * (self.dy * self.dz) / self.dx
        geom_ty = conv * (2.0 / (1.0 / ky_safe[:, :-1, :] + 1.0 / ky_safe[:, 1:, :])) * (self.dx * self.dz) / self.dy
        geom_tz = conv * (2.0 / (1.0 / kz_safe[:-1, :, :] + 1.0 / kz_safe[1:, :, :])) * (self.dx * self.dy) / self.dz

        # Precompute flat neighbor index pairs for fast vectorized sparse matrix assembly
        grid_idx = np.arange(self.n_cells).reshape((nz, ny, nx))
        x_l, x_r = grid_idx[:, :, :-1].ravel(), grid_idx[:, :, 1:].ravel()
        y_l, y_r = grid_idx[:, :-1, :].ravel(), grid_idx[:, 1:, :].ravel()
        z_l, z_r = grid_idx[:-1, :, :].ravel(), grid_idx[1:, :, :].ravel()

        # Precompute Peaceman well indices per perforated cell
        r_eq = 0.14 * np.sqrt(self.dx ** 2 + self.dy ** 2)
        peaceman_denom = np.log(r_eq / 0.1) + 1e-3
        well_specs = []
        for w in wells_config:
            wi = int(np.clip(w["i"] - 1, 0, nx - 1))
            wj = int(np.clip(w["j"] - 1, 0, ny - 1))
            k1 = int(np.clip(w.get("k1", 1) - 1, 0, nz - 1))
            k2 = int(np.clip(w.get("k2", nz) - 1, 0, nz - 1))
            ks = np.arange(k1, k2 + 1)
            kh = np.sqrt(kx_safe[ks, wj, wi] * ky_safe[ks, wj, wi])
            wi_factors = conv * 2.0 * np.pi * kh * self.dz / peaceman_denom
            well_specs.append({
                "name": w["name"],
                "type": w["type"],
                "wi": wi,
                "wj": wj,
                "ks": ks,
                "cells": grid_idx[ks, wj, wi],
                "wi_factors": wi_factors,
                "target": w.get("target_rate", 30.0)
            })

        # Effective compressibility storage term (bar^-1) for transient decline
        c_total = 1.5e-4
        p_bhp_prod = 205.0
        p_bhp_inj = 295.0

        results = {
            "time_days": timesteps_days,
            "wells": {w["name"]: {"WOPR": [], "WWPR": [], "WWCT": [], "WBHP": []} for w in wells_config},
            "field": {"FOPR": [], "FWPR": [], "FWCT": [], "FPR": []}
        }

        prev_t = 0.0
        for t_day in timesteps_days:
            dt = max(1.0, t_day - prev_t)
            prev_t = t_day

            krw, kro, total_mob, fw = self._relperm(sw)

            # Face total mobilities (harmonic/arithmetic average)
            mob_x = 0.5 * (total_mob[:, :, :-1] + total_mob[:, :, 1:])
            mob_y = 0.5 * (total_mob[:, :-1, :] + total_mob[:, 1:, :])
            mob_z = 0.5 * (total_mob[:-1, :, :] + total_mob[1:, :, :])

            tx = (geom_tx * mob_x).ravel()
            ty = (geom_ty * mob_y).ravel()
            tz = (geom_tz * mob_z).ravel()

            # Diagonal compressibility accumulation + well Peaceman terms
            accum = (cell_pv.ravel() * c_total) / dt
            diag = accum.copy()
            rhs = accum * p.ravel()

            # Off-diagonal sums into diagonal
            np.add.at(diag, x_l, tx)
            np.add.at(diag, x_r, tx)
            np.add.at(diag, y_l, ty)
            np.add.at(diag, y_r, ty)
            np.add.at(diag, z_l, tz)
            np.add.at(diag, z_r, tz)

            # Implicit Peaceman well terms in pressure system
            for ws in well_specs:
                cells = ws["cells"]
                if ws["type"] == "INJ":
                    # Injectors supply water up to target rate or max BHP
                    mob_inj = np.maximum(krw[ws["ks"], ws["wj"], ws["wi"]] / self.mu_w + 0.25 / self.mu_w, 0.2)
                    wi_mob = ws["wi_factors"] * mob_inj
                    diag[cells] += wi_mob
                    rhs[cells] += wi_mob * p_bhp_inj
                else:
                    mob_prod = total_mob[ws["ks"], ws["wj"], ws["wi"]]
                    wi_mob = ws["wi_factors"] * mob_prod
                    diag[cells] += wi_mob
                    rhs[cells] += wi_mob * p_bhp_prod

            rows = np.concatenate([x_l, x_r, y_l, y_r, z_l, z_r, np.arange(self.n_cells)])
            cols = np.concatenate([x_r, x_l, y_r, y_l, z_r, z_l, np.arange(self.n_cells)])
            vals = np.concatenate([-tx, -tx, -ty, -ty, -tz, -tz, diag])

            A = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_cells, self.n_cells))
            p = spla.spsolve(A, rhs).reshape((nz, ny, nx))

            # Compute well flows from updated pressure field
            q_inj_cells = np.zeros((nz, ny, nx), dtype=np.float64)
            q_prod_cells = np.zeros((nz, ny, nx), dtype=np.float64)
            well_oil_rates = {}
            well_water_rates = {}
            well_bhps = {}

            for ws in well_specs:
                ks, wj, wi = ws["ks"], ws["wj"], ws["wi"]
                p_cell = p[ks, wj, wi]

                if ws["type"] == "INJ":
                    mob_inj = np.maximum(krw[ks, wj, wi] / self.mu_w + 0.25 / self.mu_w, 0.2)
                    q_cells = ws["wi_factors"] * mob_inj * np.maximum(0.0, p_bhp_inj - p_cell)
                    # Cap at target injection rate if exceeded
                    tot = float(np.sum(q_cells))
                    if tot > ws["target"] and tot > 0:
                        q_cells = q_cells * (ws["target"] / tot)
                    q_inj_cells[ks, wj, wi] += q_cells
                    well_water_rates[ws["name"]] = float(np.sum(q_cells))
                    well_oil_rates[ws["name"]] = 0.0
                    well_bhps[ws["name"]] = float(np.mean(p_cell + q_cells / np.maximum(ws["wi_factors"] * mob_inj, 1e-4)))
                else:
                    mob_prod = total_mob[ks, wj, wi]
                    q_cells = ws["wi_factors"] * mob_prod * np.maximum(0.0, p_cell - p_bhp_prod)
                    tot = float(np.sum(q_cells))
                    # Cap only at high surface choke (3.0 * target) so permeability directly controls early/mid rate
                    max_liq = ws["target"] * 3.0
                    if tot > max_liq and tot > 0:
                        q_cells = q_cells * (max_liq / tot)
                    q_prod_cells[ks, wj, wi] += q_cells
                    fw_c = fw[ks, wj, wi]
                    qw = float(np.sum(q_cells * fw_c))
                    qo = float(np.sum(q_cells * (1.0 - fw_c)))
                    well_water_rates[ws["name"]] = qw
                    well_oil_rates[ws["name"]] = qo
                    well_bhps[ws["name"]] = float(np.mean(np.maximum(50.0, p_cell - q_cells / np.maximum(ws["wi_factors"] * mob_prod, 1e-4))))

            # Explicit upwind saturation advection with CFL sub-stepping
            flux_x = (geom_tx * mob_x) * (p[:, :, :-1] - p[:, :, 1:])
            flux_y = (geom_ty * mob_y) * (p[:, :-1, :] - p[:, 1:, :])
            flux_z = (geom_tz * mob_z) * (p[:-1, :, :] - p[1:, :, :])

            # Estimate maximum throughput per pore volume to pick stable substeps
            total_out = np.abs(q_inj_cells) + np.abs(q_prod_cells)
            total_out[:, :, :-1] += np.maximum(flux_x, 0.0)
            total_out[:, :, 1:] += np.maximum(-flux_x, 0.0)
            total_out[:, :-1, :] += np.maximum(flux_y, 0.0)
            total_out[:, 1:, :] += np.maximum(-flux_y, 0.0)
            total_out[:-1, :, :] += np.maximum(flux_z, 0.0)
            total_out[1:, :, :] += np.maximum(-flux_z, 0.0)

            max_cfl = float(np.max(total_out * dt / cell_pv))
            n_sub = int(np.clip(np.ceil(max_cfl / 0.6), 1, 20))
            sub_dt = dt / n_sub

            for _ in range(n_sub):
                _, _, _, fw_sub = self._relperm(sw)
                # Upwind fractional flow on faces
                fw_x = np.where(flux_x >= 0.0, fw_sub[:, :, :-1], fw_sub[:, :, 1:])
                fw_y = np.where(flux_y >= 0.0, fw_sub[:, :-1, :], fw_sub[:, 1:, :])
                fw_z = np.where(flux_z >= 0.0, fw_sub[:-1, :, :], fw_sub[1:, :, :])

                qw_x = flux_x * fw_x
                qw_y = flux_y * fw_y
                qw_z = flux_z * fw_z

                net_water = q_inj_cells - q_prod_cells * fw_sub
                net_water[:, :, :-1] -= qw_x
                net_water[:, :, 1:] += qw_x
                net_water[:, :-1, :] -= qw_y
                net_water[:, 1:, :] += qw_y
                net_water[:-1, :, :] -= qw_z
                net_water[1:, :, :] += qw_z

                sw = np.clip(sw + (sub_dt / cell_pv) * net_water, self.sw_init, 1.0 - self.sor)

            f_oil = sum(well_oil_rates[ws["name"]] for ws in well_specs if ws["type"] == "PROD")
            f_water = sum(well_water_rates[ws["name"]] for ws in well_specs if ws["type"] == "PROD")
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
