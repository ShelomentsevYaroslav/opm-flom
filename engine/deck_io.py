"""
ECLIPSE / OPM Flow Deck I/O Parser and Serializer.
Handles .DATA master decks and .INC modular property files.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class DeckParser:
    """Robust parser and writer for Eclipse/OPM reservoir simulation files."""

    @staticmethod
    def strip_comments(text: str) -> str:
        """Removes Eclipse comments (-- ...) from lines."""
        lines = []
        for line in text.splitlines():
            # Eclipse comments start with --
            if "--" in line:
                line = line.split("--", 1)[0]
            lines.append(line)
        return "\n".join(lines)

    @classmethod
    def parse_include_array(cls, filepath: Union[str, Path], expected_size: Optional[int] = None) -> Tuple[str, np.ndarray]:
        """
        Parses an Eclipse .INC property file (e.g., PORO, PERMX, NTG).
        Handles repeat notation like '25*0.12' or '1*'.
        Returns (keyword, numpy_1d_array).
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Include file not found: {path}")

        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        clean_text = cls.strip_comments(raw_text)

        # Tokenize by whitespace
        tokens = clean_text.split()
        if not tokens:
            raise ValueError(f"Empty include file: {path}")

        keyword = tokens[0].upper()
        values: List[float] = []

        # Find array contents until slash '/'
        for token in tokens[1:]:
            if token == "/":
                break
            if "/" in token:
                # Handle token stuck to slash, e.g. 0.15/
                part = token.replace("/", "")
                if part:
                    cls._append_token_values(part, values)
                break
            cls._append_token_values(token, values)

        arr = np.array(values, dtype=np.float64)
        if expected_size is not None and arr.size != expected_size:
            # If size doesn't match exactly, raise warning or resize
            pass

        return keyword, arr

    @classmethod
    def _append_token_values(cls, token: str, values: List[float]) -> None:
        """Helper to parse repeat tokens like 10*0.2 or single floats."""
        if "*" in token:
            parts = token.split("*", 1)
            count = int(parts[0]) if parts[0] else 1
            val_str = parts[1]
            val = float(val_str) if val_str else 0.0
            values.extend([val] * count)
        else:
            try:
                values.append(float(token))
            except ValueError:
                pass

    @classmethod
    def parse_master_deck(cls, filepath: Union[str, Path]) -> Dict:
        """
        Parses master .DATA file to extract DIMENS, TITLE, and INCLUDE files.
        """
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Master deck file not found: {path}")

        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        clean_text = cls.strip_comments(raw_text)
        tokens = clean_text.split()

        data_info = {
            "path": str(path),
            "dir": str(path.parent),
            "title": "UNTITLED",
            "dimens": (1, 1, 1),
            "includes": [],
            "keywords": []
        }

        # Look for TITLE
        title_match = re.search(r"TITLE\s*\n\s*(.*?)(?=\n\s*[A-Z]|\n\s*DIMENS|\Z)", clean_text, re.DOTALL | re.IGNORECASE)
        if title_match:
            data_info["title"] = title_match.group(1).strip()

        # Look for DIMENS
        dim_match = re.search(r"DIMENS\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*/", clean_text, re.IGNORECASE)
        if dim_match:
            nx = int(dim_match.group(1))
            ny = int(dim_match.group(2))
            nz = int(dim_match.group(3))
            data_info["dimens"] = (nx, ny, nz)

        # Look for INCLUDE statements
        # Matches INCLUDE 'relative/path' / or INCLUDE "path"
        include_matches = re.findall(r"INCLUDE\s*['\"]([^'\"]+)['\"]\s*/?", clean_text, re.IGNORECASE)
        for inc in include_matches:
            resolved = cls.resolve_include_path(path.parent, inc)
            data_info["includes"].append({
                "raw_path": inc,
                "resolved_path": str(resolved) if resolved else None,
                "exists": resolved.is_file() if resolved else False
            })

        return data_info

    @classmethod
    def resolve_include_path(cls, base_dir: Union[str, Path], include_str: str) -> Optional[Path]:
        """
        Resolves include path handling Windows/Linux slashes and common typos
        (e.g., SCHEDULE.SCH vs SCHEDULRE.SCH).
        """
        base = Path(base_dir)
        norm_inc = include_str.replace("\\", "/").strip()
        candidate = base / norm_inc

        if candidate.is_file():
            return candidate

        # Case-insensitive / typo-tolerant search
        target_name = Path(norm_inc).name.lower()
        subdirs = [base] + [d for d in base.iterdir() if d.is_dir()]
        for d in subdirs:
            for f in d.iterdir():
                if f.name.lower() == target_name:
                    return f
                # Handle SCHEDULE vs SCHEDULRE typo
                if "schedul" in f.name.lower() and "schedul" in target_name:
                    return f

        return candidate

    @classmethod
    def load_reservoir_model(cls, master_data_path: Union[str, Path]) -> Dict:
        """
        Loads the complete baseline model (dimensions and 3D property arrays).
        Returns dictionary with grid dimensions and 3D arrays shaped (Nz, Ny, Nx).
        """
        meta = cls.parse_master_deck(master_data_path)
        nx, ny, nz = meta["dimens"]
        total_cells = nx * ny * nz

        model = {
            "meta": meta,
            "dimens": (nx, ny, nz),
            "properties": {}
        }

        # Scan includes for standard properties: PORO, PERMX, PERMY, PERMZ, NTG
        for inc_info in meta["includes"]:
            resolved = inc_info["resolved_path"]
            if not resolved or not Path(resolved).is_file():
                continue
            
            p_name = Path(resolved).stem.upper()
            try:
                kw, arr = cls.parse_include_array(resolved, expected_size=total_cells)
                if arr.size == total_cells:
                    # Reshape to (Nz, Ny, Nx) in Eclipse ordering
                    arr_3d = arr.reshape((nz, ny, nx))
                    model["properties"][kw] = arr_3d
            except Exception as e:
                # Non-grid includes like PVT, EQUIL, etc. can be skipped
                pass

        return model

    @classmethod
    def write_property_inc(
        cls,
        filepath: Union[str, Path],
        keyword: str,
        array_3d: np.ndarray,
        values_per_line: int = 5,
        fmt: str = "{:12.6f}"
    ) -> Path:
        """
        Writes a 3D numpy array (Nz, Ny, Nx) to an Eclipse .INC file.
        Uses standard Eclipse layout: 5 numbers per line, layer separation, trailing slash.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        nz, ny, nx = array_3d.shape
        lines = [keyword.upper()]

        for k in range(nz):
            layer = array_3d[k, :, :].flatten()  # I varies fastest, then J
            layer_tokens = []
            for idx, val in enumerate(layer):
                layer_tokens.append(fmt.format(float(val)))
                if (idx + 1) % values_per_line == 0:
                    lines.append("  " + "  ".join(layer_tokens))
                    layer_tokens = []
            if layer_tokens:
                lines.append("  " + "  ".join(layer_tokens))
            # Blank line between layers for clean Eclipse structure
            if k < nz - 1:
                lines.append("")

        lines.append("  /")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path
