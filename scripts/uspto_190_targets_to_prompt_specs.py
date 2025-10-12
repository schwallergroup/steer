import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PROMPT = (
    "Highly feasible synthesis with high overall yields, consider potential side reactions and byproducts. Also ensure no unnecesary reactions are performed."
)


def parse_args() -> argparse.Namespace:
    """
    Convert a USPTO-190 targets file into a prompt_specs-style JSON array.

    Each line in the input file is expected to be a Python tuple literal where the
    first element is the target SMILES string. Example line:
        ('SMILES_TARGET', 'SMILES_SOMETHING_ELSE')

    Example usage:
        python steer/scripts/uspto_190_targets_to_prompt_specs.py \
            --input /home/andres/Documents/steer/uspto_190_targets.txt \
            --output /home/andres/Documents/steer/data/feasibility/prompt_specs.uspto190.json

    You may customize the prompt and evaluation config using flags.
    """
    parser = argparse.ArgumentParser(
        description="Generate prompt_specs-style target definitions from USPTO-190 targets"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to uspto_190_targets.txt (tuple-per-line; first entry is SMILES)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write combined JSON array of target definitions",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to include for each target",
    )
    parser.add_argument(
        "--eval-type",
        choices=["RingBreakDepth", "SpecificBondBreak", "MultiRxnCond"],
        default="RingBreakDepth",
        help="Evaluation type to set on each target definition",
    )
    parser.add_argument(
        "--depth-type",
        choices=["diff", "bool"],
        default="diff",
        help="Type for eval_config.target_depth",
    )
    parser.add_argument(
        "--depth-value",
        type=int,
        default=1,
        help="Value for eval_config.target_depth.value (e.g., 1 for diff, -1 for bool)",
    )
    parser.add_argument(
        "--id-scheme",
        choices=["md5", "index"],
        default="md5",
        help="How to generate the id field (md5 of SMILES or line index)",
    )
    parser.add_argument(
        "--id-prefix",
        default="uspto190-",
        help="Prefix to use when --id-scheme=index",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="If provided, deduplicate identical SMILES while preserving first occurrence",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of entries to include",
    )
    return parser.parse_args()


def parse_first_smiles(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    try:
        t = ast.literal_eval(line)
        if isinstance(t, tuple) and len(t) >= 1 and isinstance(t[0], str):
            return t[0]
    except Exception:
        return None
    return None


def iter_smiles_from_file(path: Path) -> Iterable[str]:
    with path.open("r") as fh:
        for line in fh:
            smi = parse_first_smiles(line)
            if smi:
                yield smi


def make_id(smiles: str, scheme: str, index: int, prefix: str) -> str:
    if scheme == "index":
        return f"{prefix}{index:03d}"
    # default: md5 of SMILES
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


def build_entry(
    smiles: str,
    prompt: str,
    eval_type: str,
    depth_type: str,
    depth_value: int,
    id_value: str,
) -> Dict[str, Any]:
    return {
        "smiles": smiles,
        "prompt": prompt,
        "id": id_value,
        "eval_type": eval_type,
        "eval_config": {
            "target_depth": {
                "type": depth_type,
                "value": depth_value,
            }
        },
    }


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    seen = set()
    entries: List[Dict[str, Any]] = []

    for idx, smiles in enumerate(iter_smiles_from_file(in_path), start=1):
        if args.dedupe:
            if smiles in seen:
                continue
            seen.add(smiles)

        if args.limit is not None and len(entries) >= args.limit:
            break

        id_value = make_id(
            smiles=smiles, scheme=args.id_scheme, index=idx, prefix=args.id_prefix
        )
        id_value = f"target_{idx:03d}"
        entry = build_entry(
            smiles=smiles,
            prompt=args.prompt,
            eval_type=args.eval_type,
            depth_type=args.depth_type,
            depth_value=args.depth_value,
            id_value=id_value,
        )
        entries.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


