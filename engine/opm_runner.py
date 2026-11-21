"""
OPM Flow Simulator Integration Module.
Detects, invokes, and parses output from the external OPM Flow (flow) binary
via native command line, WSL (Windows Subsystem for Linux), or custom binary path.
"""

from pathlib import Path
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Union


class OpmFlowRunner:
    """
    Manages external execution of the industrial OPM Flow reservoir simulator.
    """

    @classmethod
    def detect_flow_installation(cls, custom_path: Optional[str] = None) -> Dict[str, Union[bool, str]]:
        """
        Detects whether OPM Flow executable 'flow' is available.
        Checks: custom path -> native PATH -> WSL.
        """
        # 1. Custom path check
        if custom_path and Path(custom_path).is_file():
            return {
                "available": True,
                "engine": "custom",
                "path": str(custom_path),
                "description": f"Custom OPM Flow executable at {custom_path}"
            }

        # 2. Native PATH check
        native_path = shutil.which("flow")
        if native_path:
            return {
                "available": True,
                "engine": "native",
                "path": native_path,
                "description": f"Native OPM Flow executable found: {native_path}"
            }

        # 3. WSL check (Windows only)
        wsl_path = shutil.which("wsl")
        if wsl_path:
            try:
                proc = subprocess.run(
                    ["wsl", "which", "flow"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return {
                        "available": True,
                        "engine": "wsl",
                        "path": proc.stdout.strip(),
                        "description": f"OPM Flow detected inside WSL at {proc.stdout.strip()}"
                    }
            except Exception:
                pass

        return {
            "available": False,
            "engine": "none",
            "path": "",
            "description": "OPM Flow is not installed locally. Use built-in Proxy Simulator for instant simulation."
        }

    @classmethod
    def run_flow_simulation(
        cls,
        deck_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        custom_flow_path: Optional[str] = None,
        timeout_seconds: int = 300
    ) -> Dict:
        """
        Executes OPM Flow simulator on the provided .DATA deck file.
        Returns execution dictionary with status, timing, and log output.
        """
        deck = Path(deck_path).resolve()
        if not deck.is_file():
            return {"success": False, "error": f"Deck file not found: {deck}"}

        out = Path(output_dir).resolve() if output_dir else deck.parent
        out.mkdir(parents=True, exist_ok=True)

        env_info = cls.detect_flow_installation(custom_flow_path)
        if not env_info["available"]:
            return {
                "success": False,
                "error": "OPM Flow executable is not available on this machine.",
                "details": env_info["description"]
            }

        engine_type = env_info["engine"]
        cmd = []

        if engine_type == "wsl":
            # Convert Windows path to WSL path
            wsl_deck = cls._to_wsl_path(deck)
            wsl_out = cls._to_wsl_path(out)
            cmd = ["wsl", "flow", f"--output-dir={wsl_out}", wsl_deck]
        else:
            flow_exec = env_info["path"]
            cmd = [flow_exec, f"--output-dir={out}", str(deck)]

        t_start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                cwd=str(deck.parent)
            )
            elapsed = time.time() - t_start

            stem = deck.stem
            # Check standard Eclipse/OPM output files
            unsmry_file = out / f"{stem}.UNSMRY"
            smspec_file = out / f"{stem}.SMSPEC"
            prt_file = out / f"{stem}.PRT"

            success = proc.returncode == 0 or unsmry_file.is_file()

            return {
                "success": success,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 2),
                "output_dir": str(out),
                "files_generated": [str(f.name) for f in out.iterdir() if f.stem == stem],
                "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-1500:] if proc.stderr else ""
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"OPM Flow timed out after {timeout_seconds} seconds.",
                "elapsed_seconds": timeout_seconds
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "elapsed_seconds": time.time() - t_start
            }

    @staticmethod
    def _to_wsl_path(windows_path: Path) -> str:
        """Converts Windows Path C:\\Users\\... to /mnt/c/Users/..."""
        p = windows_path.resolve()
        drive = p.drive[0].lower()
        rel = p.as_posix()[len(p.drive):]
        return f"/mnt/{drive}{rel}"
