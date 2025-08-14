from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent


def find_python(venv_dir: Path) -> str:
    if os.name == "nt":
        candidate = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    else:
        candidate = (PROJECT_ROOT / ".venv" / "bin" / "python")
    if candidate.exists():
        return str(candidate)
    return shutil.which("python") or sys.executable


def find_npm() -> str:
    exe = shutil.which("npm")
    if exe is None and os.name == "nt":
        # On Windows npm is usually npm.cmd
        exe = shutil.which("npm.cmd")
    if exe is None:
        raise RuntimeError("找不到 npm，請先安裝 Node.js (含 npm)。")
    return exe


def stream_output(prefix: str, proc: subprocess.Popen) -> None:
    try:
        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            try:
                text = line.decode(errors="ignore").rstrip()
            except Exception:
                text = str(line).rstrip()
            print(f"[{prefix}] {text}")
    except Exception:
        pass


def launch(command: List[str], cwd: Path, prefix: str) -> subprocess.Popen:
    env = os.environ.copy()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    t = threading.Thread(target=stream_output, args=(prefix, proc), daemon=True)
    t.start()
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="啟動後端與前端（選用啟動遠端ASR）")
    parser.add_argument("--no-frontend", action="store_true", help="不要啟動前端 Vite 伺服器")
    parser.add_argument("--no-backend", action="store_true", help="不要啟動後端 FastAPI")
    parser.add_argument("--with-remote", action="store_true", help="同時啟動遠端 Whisper 推論伺服器")
    parser.add_argument("--backend-port", default="8000", help="後端埠，預設 8000")
    parser.add_argument("--remote-port", default="8001", help="遠端伺服器埠，預設 8001")
    parser.add_argument("--frontend-port", default="5173", help="前端埠，預設 5173（Vite 自行決定）")
    args = parser.parse_args()

    procs: List[subprocess.Popen] = []

    try:
        # Backend
        if not args.no_backend:
            backend_dir = PROJECT_ROOT / "backend"
            py_backend = find_python(PROJECT_ROOT / ".venv")
            cmd_backend = [
                py_backend,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(args.backend_port),
                "--reload",
            ]
            procs.append(launch(cmd_backend, backend_dir, prefix="backend"))

        # Remote ASR (optional)
        if args.with_remote:
            remote_dir = PROJECT_ROOT / "remote_server"
            py_remote = find_python(PROJECT_ROOT / ".venv")
            cmd_remote = [
                py_remote,
                "-m",
                "uvicorn",
                "remote_inference_server:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(args.remote_port),
            ]
            procs.append(launch(cmd_remote, remote_dir, prefix="remote"))

        # Frontend
        if not args.no_frontend and (PROJECT_ROOT / "frontend" / "package.json").exists():
            npm = find_npm()
            cmd_frontend = [npm, "run", "dev"]
            procs.append(launch(cmd_frontend, PROJECT_ROOT / "frontend", prefix="frontend"))
        elif not args.no_frontend:
            print("[info] 找不到 frontend/package.json，略過前端啟動")

        print("服務啟動中。按 Ctrl+C 停止。")

        # Wait all
        def handle_sigint(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        for proc in procs:
            proc.wait()

        return 0
    except KeyboardInterrupt:
        print("收到中斷，正在關閉子行程...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        return 0
    except Exception as e:
        print(f"[error] {e}")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


