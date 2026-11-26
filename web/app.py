"""
FastAPI Backend for OPM-Synth Studio.
Provides REST APIs for 3D/2D reservoir visualization, batch generation,
validation diagnostics, file inspection, and text file export (ZIP / .DATA / .INC).
"""

import asyncio
import io
import json
from pathlib import Path
import threading
from typing import Dict, List, Optional
import zipfile

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pandas as pd

from engine.deck_io import DeckParser
from engine.geostats import GeostatisticalEngine
from engine.opm_runner import OpmFlowRunner
from engine.proxy_flow import ReservoirProxySimulator
from engine.validator import ReservoirValidator
from generator import SyntheticDataGenerator

app = FastAPI(title="OPM-Synth Studio API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
BASE_DIR = Path(__file__).resolve().parent.parent
primary_deck = BASE_DIR / "data" / "baseline_deck" / "HEAD_OF_MODEL_1.DATA"
fallback_deck = BASE_DIR / "Материалы для работы" / "HEAD_OF_MODEL_1.DATA"
BASE_MODEL_PATH = primary_deck if primary_deck.is_file() else fallback_deck

generator_instance = None
try:
    generator_instance = SyntheticDataGenerator(base_model_path=BASE_MODEL_PATH)
except Exception as e:
    print(f"Warning initializing base model: {e}")

generation_status = {
    "is_running": False,
    "current_step": 0,
    "total_steps": 0,
    "message": "Idle",
    "last_result": None
}


class GenerateRequest(BaseModel):
    n_models: int = 5
    rx: float = 4.0
    ry: float = 4.0
    rz: float = 2.5
    variogram_type: str = "exponential"
    perturbation_fraction: float = 0.25
    corr_factor: float = 0.90
    aniso_z: float = 0.20
    run_simulation: bool = True
    sim_engine: str = "proxy"
    base_seed: int = 42


@app.get("/api/system")
def get_system_info():
    """Returns environment status (OPM flow detection, base model info)."""
    opm_status = OpmFlowRunner.detect_flow_installation()
    base_loaded = generator_instance is not None
    dimens = generator_instance.base_model["dimens"] if base_loaded else (0, 0, 0)
    props = list(generator_instance.base_model["properties"].keys()) if base_loaded else []

    return {
        "status": "ready",
        "base_model": {
            "loaded": base_loaded,
            "title": generator_instance.base_model["meta"].get("title", "") if base_loaded else "",
            "dimens": dimens,
            "properties": props
        },
        "opm_flow": opm_status,
        "calibration": generator_instance.calibration if base_loaded else {}
    }


@app.get("/api/base-slice")
def get_base_slice(property_name: str = "PORO", layer_k: int = 1):
    """Returns 2D slice for baseline model layer K (1-indexed)."""
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Base model not initialized")

    prop = property_name.upper()
    arr_3d = generator_instance.base_model["properties"].get(prop)
    if arr_3d is None:
        raise HTTPException(status_code=404, detail=f"Property {prop} not found")

    nz, ny, nx = arr_3d.shape
    k_idx = int(np.clip(layer_k - 1, 0, nz - 1))
    slice_2d = arr_3d[k_idx, :, :]

    return {
        "property": prop,
        "layer_k": k_idx + 1,
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "min": float(np.min(arr_3d)),
        "max": float(np.max(arr_3d)),
        "data": slice_2d.tolist()
    }


@app.get("/api/base-3d")
def get_base_3d(property_name: str = "PORO"):
    """Returns 3D flat grid points (X, Y, Z, Value) for 3D visualization."""
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Base model not initialized")

    prop = property_name.upper()
    arr_3d = generator_instance.base_model["properties"].get(prop)
    if arr_3d is None:
        raise HTTPException(status_code=404, detail=f"Property {prop} not found")

    nz, ny, nx = arr_3d.shape
    z_coords, y_coords, x_coords = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")

    return {
        "property": prop,
        "nx": nx, "ny": ny, "nz": nz,
        "x": x_coords.flatten().tolist(),
        "y": y_coords.flatten().tolist(),
        "z": z_coords.flatten().tolist(),
        "values": [round(float(v), 4) for v in arr_3d.flatten()]
    }


@app.get("/api/model-3d")
def get_model_3d(ensemble_id: str, model_id: str, property_name: str = "PORO"):
    """Returns 3D flat grid points (X, Y, Z, Value) for a generated synthetic model."""
    prop = property_name.upper()
    inc_map = {
        "PORO": "EX1_PORO.inc",
        "PERMX": "EX1_PERMX.inc",
        "PERMY": "EX1_PERMY.inc",
        "PERMZ": "EX1_PERMZ.inc",
        "NTG": "EX1_NTG.inc",
    }
    inc_file = inc_map.get(prop)
    if not inc_file:
        raise HTTPException(status_code=404, detail=f"Property {prop} not supported")

    target_path = BASE_DIR / "output" / ensemble_id / model_id / "LIBRARY_OF_MODULS" / inc_file
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Model file not found: {target_path}")

    nx, ny, nz = generator_instance.base_model["dimens"] if generator_instance else (15, 15, 6)
    _, arr = DeckParser.parse_include_array(target_path, expected_size=nx * ny * nz)
    arr_3d = arr.reshape((nz, ny, nx))
    z_coords, y_coords, x_coords = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")

    return {
        "property": prop,
        "model_id": model_id,
        "nx": nx, "ny": ny, "nz": nz,
        "x": x_coords.flatten().tolist(),
        "y": y_coords.flatten().tolist(),
        "z": z_coords.flatten().tolist(),
        "values": [round(float(v), 4) for v in arr_3d.flatten()]
    }


_baseline_flow_cache: Optional[Dict] = None


def _get_baseline_flow() -> Dict:
    """Computes and caches baseline proxy flow curves once."""
    global _baseline_flow_cache
    if _baseline_flow_cache is not None:
        return _baseline_flow_cache
    if not generator_instance:
        return {"time_days": [], "FOPR": [], "FWCT": []}

    props = generator_instance.base_model["properties"]
    dimens = generator_instance.base_model["dimens"]
    sim = ReservoirProxySimulator(dimens=dimens)
    res = sim.simulate(props["PORO"], props["PERMX"], props["PERMY"], props["PERMZ"])
    _baseline_flow_cache = {
        "time_days": res["time_days"],
        "FOPR": res["field"]["FOPR"],
        "FWCT": res["field"]["FWCT"],
    }
    return _baseline_flow_cache


@app.get("/api/flow-curves")
def get_flow_curves(ensemble_id: str, model_id: str):
    """Returns real production curves (FOPR, FWCT) for baseline, selected model, and ensemble."""
    base = _get_baseline_flow()
    csv_path = BASE_DIR / "output" / ensemble_id / "summary_production_curves.csv"
    if not csv_path.is_file():
        return {
            "has_data": False,
            "time_days": base["time_days"],
            "base": {"FOPR": base["FOPR"], "FWCT": base["FWCT"]},
            "selected": None,
            "ensemble": {}
        }

    df = pd.read_csv(csv_path)
    ensemble_curves = {}
    for mid, grp in df.groupby("model_id"):
        grp_sorted = grp.sort_values("time_days")
        ensemble_curves[str(mid)] = {
            "time_days": grp_sorted["time_days"].tolist(),
            "FOPR": [round(float(v), 2) for v in grp_sorted["fopr"].tolist()],
            "FWCT": [round(float(v), 4) for v in grp_sorted["fwct"].tolist()],
        }

    selected = ensemble_curves.get(model_id)
    if selected is None and ensemble_curves:
        first_key = next(iter(ensemble_curves))
        selected = ensemble_curves[first_key]

    return {
        "has_data": selected is not None,
        "model_id": model_id,
        "time_days": selected["time_days"] if selected else base["time_days"],
        "base": {"FOPR": base["FOPR"], "FWCT": base["FWCT"]},
        "selected": selected,
        "ensemble": ensemble_curves
    }


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/ensembles")
def list_ensembles():
    """Lists generated ensembles in output/ directory."""
    output_dir = BASE_DIR / "output"
    ensembles = []
    if output_dir.is_dir():
        for d in sorted(output_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "manifest.json").is_file():
                try:
                    meta = DeckParser.strip_comments((d / "manifest.json").read_text(encoding="utf-8"))
                    info = json.loads(meta)
                    ensembles.append({
                        "id": d.name,
                        "path": str(d),
                        "created_at": info.get("created_at"),
                        "n_models": info.get("n_models", len(info.get("models", []))),
                        "parameters": info.get("parameters", {})
                    })
                except Exception:
                    pass
    return ensembles


@app.get("/api/ensemble-details")
def get_ensemble_details(ensemble_id: str):
    """Returns details and models for a given ensemble."""
    ens_dir = BASE_DIR / "output" / ensemble_id
    manifest_file = ens_dir / "manifest.json"
    if not manifest_file.is_file():
        raise HTTPException(status_code=404, detail="Ensemble not found")

    info = json.loads(manifest_file.read_text(encoding="utf-8"))
    return info


@app.get("/api/model-slice")
def get_synthetic_model_slice(ensemble_id: str, model_id: str, property_name: str = "PORO", layer_k: int = 1):
    """Returns 2D slice for synthetic model layer K."""
    model_dir = BASE_DIR / "output" / ensemble_id / model_id / "LIBRARY_OF_MODULS"
    inc_name = f"EX1_{property_name.upper()}.inc"
    inc_path = model_dir / inc_name
    if not inc_path.is_file():
        raise HTTPException(status_code=404, detail=f"Include file {inc_name} not found")

    dimens = generator_instance.base_model["dimens"] if generator_instance else (15, 15, 6)
    nx, ny, nz = dimens
    kw, arr = DeckParser.parse_include_array(inc_path, expected_size=nx * ny * nz)
    arr_3d = arr.reshape((nz, ny, nx))

    k_idx = int(np.clip(layer_k - 1, 0, nz - 1))
    slice_2d = arr_3d[k_idx, :, :]

    return {
        "ensemble_id": ensemble_id,
        "model_id": model_id,
        "property": property_name.upper(),
        "layer_k": k_idx + 1,
        "nz": nz, "ny": ny, "nx": nx,
        "min": float(np.min(arr_3d)),
        "max": float(np.max(arr_3d)),
        "data": slice_2d.tolist()
    }


@app.get("/api/qc-diagnostic")
def get_qc_diagnostic(ensemble_id: Optional[str] = None, model_id: Optional[str] = None):
    """
    Returns comparative QC data:
    Variograms along X and Z for baseline vs synthetic model,
    and cross-plot scatter points for Poro vs log(K).
    """
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Base model not loaded")

    base_p = generator_instance.base_model["properties"]["PORO"]
    base_k = generator_instance.base_model["properties"]["PERMX"]

    lags_z, g_z_base = GeostatisticalEngine.compute_experimental_variogram(base_p, "z")
    lags_x, g_x_base = GeostatisticalEngine.compute_experimental_variogram(base_p, "x")

    synth_p = None
    synth_k = None
    synth_kz = None
    g_z_synth = []
    g_x_synth = []
    summary_metrics = None

    if ensemble_id and model_id:
        lib = BASE_DIR / "output" / ensemble_id / model_id / "LIBRARY_OF_MODULS"
        dimens = generator_instance.base_model["dimens"]
        nx, ny, nz = dimens
        if (lib / "EX1_PORO.inc").is_file():
            _, arr_p = DeckParser.parse_include_array(lib / "EX1_PORO.inc", expected_size=nx * ny * nz)
            synth_p = arr_p.reshape((nz, ny, nx))
            _, g_z_synth = GeostatisticalEngine.compute_experimental_variogram(synth_p, "z")
            _, g_x_synth = GeostatisticalEngine.compute_experimental_variogram(synth_p, "x")

        if (lib / "EX1_PERMX.inc").is_file():
            _, arr_k = DeckParser.parse_include_array(lib / "EX1_PERMX.inc", expected_size=nx * ny * nz)
            synth_k = arr_k.reshape((nz, ny, nx))

        if (lib / "EX1_PERMZ.inc").is_file():
            _, arr_kz = DeckParser.parse_include_array(lib / "EX1_PERMZ.inc", expected_size=nx * ny * nz)
            synth_kz = arr_kz.reshape((nz, ny, nx))

        if synth_p is not None and synth_k is not None:
            props_to_check = {"PORO": synth_p, "PERMX": synth_k}
            if synth_kz is not None:
                props_to_check["PERMZ"] = synth_kz
            qc_res = ReservoirValidator.validate_property_set(props_to_check, generator_instance.base_model["properties"])
            m = qc_res.get("metrics", {})
            v = m.get("variogram", {})
            summary_metrics = {
                "score": qc_res.get("score", 100),
                "status": qc_res.get("status", "PASS"),
                "vertical_continuity": v.get("vertical_continuity", "PRESERVED"),
                "z_lag1_ratio": v.get("z_lag1_ratio", 0.0),
                "petro_corr_r": m.get("petro_corr_r", 0.0),
                "anisotropy_kz_kx": m.get("anisotropy_kz_kx", 0.0)
            }

    sample_stride = 4
    base_scatter = {
        "poro": [round(float(v), 4) for v in base_p.flatten()[::sample_stride]],
        "permx": [round(float(v), 2) for v in base_k.flatten()[::sample_stride]]
    }

    synth_scatter = None
    if synth_p is not None and synth_k is not None:
        synth_scatter = {
            "poro": [round(float(v), 4) for v in synth_p.flatten()[::sample_stride]],
            "permx": [round(float(v), 2) for v in synth_k.flatten()[::sample_stride]]
        }

    return {
        "lags_z": lags_z.tolist(),
        "lags_x": lags_x.tolist(),
        "variogram_z_base": [round(float(v), 5) for v in g_z_base],
        "variogram_x_base": [round(float(v), 5) for v in g_x_base],
        "variogram_z_synth": [round(float(v), 5) for v in g_z_synth] if len(g_z_synth) else [],
        "variogram_x_synth": [round(float(v), 5) for v in g_x_synth] if len(g_x_synth) else [],
        "base_scatter": base_scatter,
        "synth_scatter": synth_scatter,
        "summary_metrics": summary_metrics
    }


# ==========================================
# Text Files & Export APIs (The Core Feature)
# ==========================================

@app.get("/api/file-list")
def list_model_files(ensemble_id: str, model_id: str):
    """
    Returns list of all generated text files (.DATA, .INC, .CSV, .JSON)
    for a given synthetic realization.
    """
    ens_dir = BASE_DIR / "output" / ensemble_id
    model_dir = ens_dir / model_id
    if not model_dir.is_dir():
        raise HTTPException(status_code=404, detail="Model folder not found")

    files = []
    # 1. Master .DATA file
    data_file = model_dir / f"{model_id}.DATA"
    if data_file.is_file():
        files.append({
            "name": data_file.name,
            "rel_path": data_file.name,
            "type": "data",
            "size_bytes": data_file.stat().st_size
        })

    # 2. Modular include .INC files
    lib_dir = model_dir / "LIBRARY_OF_MODULS"
    if lib_dir.is_dir():
        for f in sorted(lib_dir.iterdir()):
            if f.is_file():
                ext = f.suffix.lower().replace(".", "")
                files.append({
                    "name": f.name,
                    "rel_path": f"LIBRARY_OF_MODULS/{f.name}",
                    "type": ext,
                    "size_bytes": f.stat().st_size
                })

    # 3. CSV summary curves if available
    csv_file = ens_dir / "summary_production_curves.csv"
    if csv_file.is_file():
        files.append({
            "name": csv_file.name,
            "rel_path": csv_file.name,
            "type": "csv",
            "size_bytes": csv_file.stat().st_size
        })

    # 4. Manifest JSON
    man_file = ens_dir / "manifest.json"
    if man_file.is_file():
        files.append({
            "name": man_file.name,
            "rel_path": man_file.name,
            "type": "json",
            "size_bytes": man_file.stat().st_size
        })

    return files


@app.get("/api/file-content")
def get_file_content(ensemble_id: str, model_id: str, rel_path: str):
    """
    Returns raw text content of a generated text file (.DATA, .INC, .CSV, etc.)
    with line numbers for web-based code viewer.
    """
    ens_dir = BASE_DIR / "output" / ensemble_id
    model_dir = ens_dir / model_id

    # Security check: prevent directory traversal
    norm_rel = Path(rel_path.replace("\\", "/"))
    if ".." in str(norm_rel):
        raise HTTPException(status_code=400, detail="Invalid path")

    # Check model dir first, then ensemble dir
    target_file = (model_dir / norm_rel).resolve()
    if not target_file.is_file():
        target_file = (ens_dir / norm_rel).resolve()
        if not target_file.is_file():
            raise HTTPException(status_code=404, detail="File not found")

    text = target_file.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    return {
        "filename": target_file.name,
        "rel_path": rel_path,
        "line_count": len(lines),
        "size_bytes": target_file.stat().st_size,
        "content": text
    }


@app.get("/api/export-zip")
def export_ensemble_zip(ensemble_id: str, model_id: Optional[str] = None):
    """
    Streams a full .ZIP archive with all generated text files (.DATA, .INC, .CSV, .JSON).
    If model_id is specified, archives only that model; otherwise archives the entire ensemble.
    """
    ens_dir = BASE_DIR / "output" / ensemble_id
    if not ens_dir.is_dir():
        raise HTTPException(status_code=404, detail="Ensemble not found")

    target_path = ens_dir / model_id if model_id else ens_dir
    if not target_path.is_dir():
        raise HTTPException(status_code=404, detail="Target model or ensemble directory not found")
    archive_name = f"{model_id or ensemble_id}.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in target_path.rglob("*"):
            if f.is_file():
                rel_in_zip = f.relative_to(target_path)
                z.write(f, arcname=str(rel_in_zip))

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={archive_name}"}
    )


@app.get("/api/download-single-file")
def download_single_file(ensemble_id: str, model_id: str, rel_path: str):
    """Directly downloads a single text file (.DATA or .INC)."""
    ens_dir = BASE_DIR / "output" / ensemble_id
    model_dir = ens_dir / model_id
    target_file = (model_dir / rel_path).resolve()
    if not target_file.is_file():
        target_file = (ens_dir / rel_path).resolve()
        if not target_file.is_file():
            raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(target_file),
        filename=target_file.name,
        media_type="text/plain"
    )


# ==========================================
# Generation Task
# ==========================================

def _run_batch_task(req: GenerateRequest):
    global generation_status
    generation_status["is_running"] = True
    generation_status["total_steps"] = req.n_models

    def progress(step, total, msg):
        generation_status["current_step"] = step
        generation_status["total_steps"] = total
        generation_status["message"] = msg

    try:
        res = generator_instance.generate_ensemble(
            n_models=req.n_models,
            rx=req.rx,
            ry=req.ry,
            rz=req.rz,
            variogram_type=req.variogram_type,
            perturbation_fraction=req.perturbation_fraction,
            corr_factor=req.corr_factor,
            aniso_z=req.aniso_z,
            run_simulation=req.run_simulation,
            sim_engine=req.sim_engine,
            base_seed=req.base_seed,
            output_dir=BASE_DIR / "output" / f"ensemble_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}",
            progress_callback=progress
        )
        generation_status["last_result"] = res
        generation_status["message"] = "Completed successfully!"
    except Exception as e:
        generation_status["message"] = f"Error: {str(e)}"
    finally:
        generation_status["is_running"] = False


@app.post("/api/generate")
def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Triggers batch generation of synthetic models."""
    global generation_status
    if generation_status["is_running"]:
        raise HTTPException(status_code=409, detail="A generation task is already in progress.")

    background_tasks.add_task(_run_batch_task, req)
    return {"status": "started", "n_models": req.n_models}


@app.get("/api/generation-status")
def get_generation_status():
    """Polls real-time generation progress."""
    return generation_status


# Static files mount
static_dir = Path(__file__).parent / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
