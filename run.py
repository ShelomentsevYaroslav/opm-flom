"""
Entry point for OPM-Synth Studio.
Launches the local web server and opens the interactive dashboard.
Usage:
    python run.py             # Launches Web UI at http://127.0.0.1:8000
    python run.py --cli       # Runs batch generation in CLI mode
"""

import argparse
import os
import socket
import sys
import webbrowser
import uvicorn


def find_free_port(start_port: int = 8000) -> int:
    """Finds an open port starting from start_port."""
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    parser = argparse.ArgumentParser(description="OPM-Synth Studio: Synthetic Reservoir Data Platform")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: auto-detected from 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open default browser")
    parser.add_argument("--cli", action="store_true", help="Run batch generation in CLI mode")
    parser.add_argument("--n-models", type=int, default=5, help="Number of synthetic models for CLI mode")
    args = parser.parse_args()

    if args.cli:
        from generator import SyntheticDataGenerator
        print("=" * 60)
        print("  OPM-Synth Studio: CLI Batch Generation")
        print("=" * 60)
        gen = SyntheticDataGenerator()
        print(f"Generating ensemble with {args.n_models} models...")
        res = gen.generate_ensemble(n_models=args.n_models)
        print("=" * 60)
        print(f"✓ Successfully generated {res['n_generated']} models!")
        print(f"  Target directory: {res['target_dir']}")
        print(f"  ML Dataset (.npz): {res['npz_dataset']}")
        print("=" * 60)
        return

    port = args.port or find_free_port(8000)
    url = f"http://{args.host}:{port}"

    print("=" * 65)
    print("   ⚗️  OPM-Synth Studio — Petroleum Synthetic Data Platform")
    print("=" * 65)
    print(f"  🌐 Сервер запущен: {url}")
    print(f"  📁 Базовая модель: data/baseline_deck/HEAD_OF_MODEL_1.DATA")
    print("  🖥️  Откройте URL в веб-браузере")
    print("  Для остановки сервера нажмите Ctrl + C")
    print("=" * 65)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("web.app:app", host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
