# src/fed/run_fed.py
import argparse
import os
import socket
import subprocess
import sys
import time
from typing import List, Tuple

from src.data.load_dreamer import load_dreamer_mat


def list_subjects(mat: str) -> List[str]:
    subs, _ = load_dreamer_mat(mat)
    return sorted(list(subs.keys()))


def get_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return port


def wait_port(host: str, port: int, timeout: float = 30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _terminate(proc: subprocess.Popen, name: str, grace: float = 3.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < grace:
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()
    except Exception as e:
        print(f"[WARN] Failed to terminate {name}: {e}", file=sys.stderr)


def _terminate_many(procs: List[Tuple[subprocess.Popen, str]]) -> None:
    # try terminate first
    for p, n in procs:
        _terminate(p, n, grace=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default="/home/rajeshkumarjogi/Desktop/eeg-fl-emotion/Dataset/DREAMER.mat")
    ap.add_argument("--target", type=str, default="arousal", choices=["arousal", "valence"])
    ap.add_argument("--server", type=str, default="")  # if empty, auto-pick free port
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--fraction_fit", type=float, default=0.5)
    ap.add_argument("--fraction_evaluate", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="results/fed")  # where server saves artifacts
    ap.add_argument("--stagger", type=float, default=0.3, help="Seconds between client launches")
    args = ap.parse_args()

    # Ensure output folder
    os.makedirs(args.out_dir, exist_ok=True)

    # Pick subjects for this run
    subs = list_subjects(args.mat)
    if len(subs) < args.clients:
        print(f"[WARN] Requested {args.clients} clients, but only {len(subs)} subjects exist. Using {len(subs)}.")
    train_ids = subs[: args.clients]

    # Server address
    if args.server:
        host, port_str = args.server.split(":")
        port = int(port_str)
    else:
        host, port = "127.0.0.1", get_free_port()
    server_addr = f"{host}:{port}"

    # Build server command
    srv_cmd = [
        "python", "-m", "src.fed.server",
        "--server", server_addr,
        "--rounds", str(args.rounds),
        "--clients", str(args.clients),
        "--target", args.target,
        "--mat", args.mat,
        "--fraction_fit", str(args.fraction_fit),
        "--fraction_evaluate", str(args.fraction_evaluate),
        "--out_dir", args.out_dir,
    ]
    print("[RUN] starting server:", " ".join(srv_cmd))
    srv = subprocess.Popen(srv_cmd)

    try:
        # Wait for port to open
        print(f"[RUN] waiting for server on {server_addr} ...")
        if not wait_port(host, port, timeout=60):
            _terminate(srv, "server")
            raise RuntimeError("Server did not open the port in time.")

        # Launch clients
        procs: List[Tuple[subprocess.Popen, str]] = []
        for sid in train_ids:
            cli_cmd = [
                "python", "-m", "src.fed.client",
                "--server", server_addr,
                "--subject", sid,
                "--target", args.target,
                "--mat", args.mat,
                "--epochs", str(args.epochs),
                "--retries", "30",
                "--retry_sleep", "3.0",
            ]
            if args.max_windows:
                cli_cmd += ["--max_windows", str(args.max_windows)]
            p = subprocess.Popen(cli_cmd)
            procs.append((p, f"client[{sid}]"))
            time.sleep(args.stagger)

        # Wait for all to finish
        rc = 0
        for p, n in procs:
            r = p.wait()
            if r != 0:
                print(f"[WARN] {n} exited with code {r}")
                rc = rc or r

        # Wait for server
        rc_srv = srv.wait()
        if rc_srv != 0:
            print(f"[WARN] server exited with code {rc_srv}")
            rc = rc or rc_srv

        print("[RUN] FL finished. Output:", args.out_dir)
        sys.exit(rc)

    except KeyboardInterrupt:
        print("\n[RUN] Caught Ctrl+C — shutting down gracefully...")
        # stop clients then server
        # (they’ll also stop themselves on disconnect)
        _terminate_many(procs if "procs" in locals() else [])
        _terminate(srv, "server")
        print("[RUN] Clean exit after interrupt.")
        sys.exit(130)  # 128 + SIGINT

    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        # try to clean up
        _terminate_many(procs if "procs" in locals() else [])
        _terminate(srv, "server")
        sys.exit(1)


if __name__ == "__main__":
    main()
