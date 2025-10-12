import argparse
import ast
import json
import sys
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic routes via Synthegy server")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to uspto_190_targets.txt (tuple lines; first entry is SMILES)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to write JSON outputs (one combined + per-route files)",
    )
    parser.add_argument(
        "--url",
        default="http://0.0.0.0:3465/synthesize",
        help="Synthegy server synthesize URL",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Request timeout in seconds per target",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting line index for naming (1-based)",
    )
    return parser.parse_args()


def parse_first_smiles(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        t = ast.literal_eval(line)
        if isinstance(t, tuple) and len(t) >= 1:
            return t[0]
    except Exception:
        return None
    return None


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2))


def main() -> int:
    args = parse_args()
    targets_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    success = 0
    failed = 0

    with targets_path.open("r") as fh:
        for idx0, line in enumerate(fh, start=0):
            idx = args.start_index + idx0
            smi = parse_first_smiles(line)
            if not smi:
                continue

            payload = {"smiles": smi}
            combined_path = outdir / f"target_{idx:03d}.json"
            error_path = outdir / f"target_{idx:03d}_error.txt"

            try:
                resp = session.post(args.url, json=payload, timeout=args.timeout)
                if resp.status_code != 200:
                    failed += 1
                    error_path.write_text(f"HTTP {resp.status_code}: {resp.text}\n")
                    continue

                data = resp.json()
                write_json(combined_path, data)

                # routes = data.get("routes") or []
                # for ridx, route in enumerate(routes, start=1):
                #     write_json(outdir / f"target_{idx:03d}_routes_{ridx:02d}.json", route)

                # results = data.get("result") or []
                # for ridx, res in enumerate(results, start=1):
                #     write_json(outdir / f"target_{idx:03d}_result_{ridx:02d}.json", res)

                success += 1
            except Exception as e:
                failed += 1
                error_path.write_text(str(e))

    summary_path = outdir / "summary.txt"
    summary_path.write_text(f"success={success} failed={failed} outdir={outdir}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


