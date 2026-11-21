"""
Batch Synthetic Reservoir Data Generator.
Coordinates geostatistics, petrophysical co-simulation, Eclipse deck output,
hydrodynamic simulations, and ML dataset packaging.
"""

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from engine.deck_io import DeckParser
from engine.geostats import GeostatisticalEngine
from engine.petrophysics import PetrophysicsEngine
from engine.proxy_flow import ReservoirProxySimulator
from engine.opm_runner import OpmFlowRunner
from engine.validator import ReservoirValidator


class SyntheticDataGenerator:
    """
    High-level generator orchestrating the creation of synthetic reservoir ensembles.
    """

    def __init__(self, base_model_path: Optional[Union[str, Path]] = None):
        if base_model_path is None:
            primary = Path("data/baseline_deck/HEAD_OF_MODEL_1.DATA")
            fallback = Path("Материалы для работы/HEAD_OF_MODEL_1.DATA")
            self.base_model_path = primary.resolve() if primary.is_file() else fallback.resolve()
        else:
            self.base_model_path = Path(base_model_path).resolve()
        if not self.base_model_path.is_file():
            raise FileNotFoundError(f"Baseline model not found: {self.base_model_path}")
        self.base_model = DeckParser.load_reservoir_model(self.base_model_path)
        self.calibration = PetrophysicsEngine.calibrate_from_model(self.base_model["properties"])

    def generate_ensemble(
        self,
        n_models: int = 5,
        rx: float = 4.0,
        ry: float = 4.0,
        rz: float = 2.0,
        variogram_type: str = "exponential",
        perturbation_fraction: float = 0.25,
        corr_factor: float = 0.90,
        aniso_z: Optional[float] = None,
        run_simulation: bool = True,
        sim_engine: str = "proxy",  # 'proxy' | 'opm'
        base_seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict:
        """
        Generates N synthetic reservoir realizations with dynamic simulation and ML packaging.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(output_dir or f"output/ensemble_{timestamp}").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        dimens = self.base_model["dimens"]  # (Nx, Ny, Nz)
        nz, ny, nx = dimens[2], dimens[1], dimens[0]
        base_poro = self.base_model["properties"].get("PORO")
        if base_poro is None:
            base_poro = np.full((nz, ny, nx), 0.18)

        # Simulator instance
        proxy_sim = ReservoirProxySimulator(dimens=dimens) if sim_engine == "proxy" else None

        ensemble_meta = {
            "created_at": datetime.now().isoformat(),
            "n_models": n_models,
            "dimens": dimens,
            "parameters": {
                "rx": rx, "ry": ry, "rz": rz,
                "variogram_type": variogram_type,
                "perturbation_fraction": perturbation_fraction,
                "corr_factor": corr_factor,
                "aniso_z": aniso_z or self.calibration.get("anisotropy_z", 0.2),
                "sim_engine": sim_engine
            },
            "models": []
        }

        # Data collection for ML numpy export
        cube_poro = []
        cube_permx = []
        cube_permy = []
        cube_permz = []
        cube_ntg = []
        curve_fopr = []
        curve_fwpr = []
        curve_fwct = []
        csv_rows = []

        for i in range(1, n_models + 1):
            seed = base_seed + i * 17
            model_id = f"SYNTH_{i:03d}"
            msg = f"Generating model {i}/{n_models} ({model_id})..."
            if progress_callback:
                progress_callback(i, n_models, msg)

            model_dir = target_dir / model_id
            inc_dir = model_dir / "LIBRARY_OF_MODULS"
            inc_dir.mkdir(parents=True, exist_ok=True)

            # 1. Geostatistical Porosity generation with 3D continuity (Rx, Ry, Rz)
            synth_poro = GeostatisticalEngine.perturb_reservoir_property(
                base_array=base_poro,
                perturbation_fraction=perturbation_fraction,
                rx=rx,
                ry=ry,
                rz=rz,
                variogram_type=variogram_type,
                seed=seed,
                min_val=0.01,
                max_val=0.40
            )

            # 2. Petrophysical Co-Simulation
            synth_props = PetrophysicsEngine.cosimulate_properties(
                synthetic_poro=synth_poro,
                calibration=self.calibration,
                corr_factor=corr_factor,
                aniso_z=aniso_z,
                seed=seed + 7
            )

            # 3. Validation / QC
            qc_report = ReservoirValidator.validate_property_set(synth_props, self.base_model["properties"])

            # 4. Serialize Eclipse Include files (.INC)
            DeckParser.write_property_inc(inc_dir / "EX1_PORO.inc", "PORO", synth_props["PORO"])
            DeckParser.write_property_inc(inc_dir / "EX1_PERMX.inc", "PERMX", synth_props["PERMX"])
            DeckParser.write_property_inc(inc_dir / "EX1_PERMY.inc", "PERMY", synth_props["PERMY"])
            DeckParser.write_property_inc(inc_dir / "EX1_PERMZ.inc", "PERMZ", synth_props["PERMZ"])
            DeckParser.write_property_inc(inc_dir / "EX1_NTG.inc", "NTG", synth_props["NTG"])

            # Copy remaining static modular includes (GRID, PVT, SCHEDULE)
            base_lib = self.base_model_path.parent / "LIBRARY_OF_MODULS"
            if base_lib.is_dir():
                for item in base_lib.iterdir():
                    dst = inc_dir / item.name
                    if not dst.exists():
                        shutil.copy2(item, dst)

            # Write master .DATA file
            master_data_file = model_dir / f"{model_id}.DATA"
            # Read base master, replace title
            base_text = self.base_model_path.read_text(encoding="utf-8", errors="ignore")
            # Ensure correct schedule include reference
            base_text = base_text.replace("SCHEDULE.SCH", "SCHEDULRE.SCH")
            base_text = base_text.replace("EX1 - FIRST TUTORIAL FOR SCHEDULE", f"SYNTHETIC MODEL {model_id} (Seed {seed})")
            master_data_file.write_text(base_text, encoding="utf-8")

            # 5. Hydrodynamic Simulation
            sim_results = None
            if run_simulation:
                if sim_engine == "opm":
                    opm_res = OpmFlowRunner.run_flow_simulation(master_data_file, output_dir=model_dir)
                    if not opm_res["success"]:
                        # Fallback to proxy if OPM Flow is not installed
                        sim_results = proxy_sim.simulate(
                            synth_props["PORO"], synth_props["PERMX"], synth_props["PERMY"], synth_props["PERMZ"]
                        )
                    else:
                        sim_results = {"engine": "opm", "details": opm_res}
                else:
                    sim_results = proxy_sim.simulate(
                        synth_props["PORO"], synth_props["PERMX"], synth_props["PERMY"], synth_props["PERMZ"]
                    )

            # Store in ML ensemble arrays
            cube_poro.append(synth_props["PORO"])
            cube_permx.append(synth_props["PERMX"])
            cube_permy.append(synth_props["PERMY"])
            cube_permz.append(synth_props["PERMZ"])
            cube_ntg.append(synth_props["NTG"])

            if sim_results and "field" in sim_results:
                fopr_vals = sim_results["field"]["FOPR"]
                fwpr_vals = sim_results["field"]["FWPR"]
                fwct_vals = sim_results["field"]["FWCT"]
                curve_fopr.append(fopr_vals)
                curve_fwpr.append(fwpr_vals)
                curve_fwct.append(fwct_vals)

                for t_idx, (t_d, fo, fw, fwct) in enumerate(zip(sim_results["time_days"], fopr_vals, fwpr_vals, fwct_vals)):
                    csv_rows.append({
                        "model_id": model_id,
                        "time_days": t_d,
                        "fopr": fo,
                        "fwpr": fw,
                        "fwct": fwct,
                        "mean_poro": round(float(np.mean(synth_props["PORO"])), 4),
                        "mean_permx": round(float(np.mean(synth_props["PERMX"])), 2)
                    })

            # Record model metadata
            ensemble_meta["models"].append({
                "model_id": model_id,
                "seed": seed,
                "dir": str(model_dir),
                "data_file": str(master_data_file),
                "qc": qc_report,
                "simulation": {
                    "completed": sim_results is not None,
                    "engine": sim_engine
                }
            })

        # 6. Save ML Artifacts (.npz and .csv)
        npz_path = target_dir / "dataset_ml_features_targets.npz"
        np.savez_compressed(
            npz_path,
            poro=np.array(cube_poro, dtype=np.float32),
            permx=np.array(cube_permx, dtype=np.float32),
            permy=np.array(cube_permy, dtype=np.float32),
            permz=np.array(cube_permz, dtype=np.float32),
            ntg=np.array(cube_ntg, dtype=np.float32),
            fopr=np.array(curve_fopr, dtype=np.float32) if curve_fopr else np.array([]),
            fwpr=np.array(curve_fwpr, dtype=np.float32) if curve_fwpr else np.array([]),
            fwct=np.array(curve_fwct, dtype=np.float32) if curve_fwct else np.array([])
        )

        csv_path = target_dir / "summary_production_curves.csv"
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df.to_csv(csv_path, index=False)

        # Save Manifest JSON
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(json.dumps(ensemble_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        result_summary = {
            "success": True,
            "target_dir": str(target_dir),
            "manifest_path": str(manifest_path),
            "npz_dataset": str(npz_path),
            "csv_dataset": str(csv_path) if csv_rows else None,
            "n_generated": n_models,
            "models": ensemble_meta["models"]
        }

        if progress_callback:
            progress_callback(n_models, n_models, f"Done! {n_models} models generated successfully.")

        return result_summary
