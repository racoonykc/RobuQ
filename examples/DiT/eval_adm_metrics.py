"""Run ADM/guided-diffusion TensorFlow metrics on DiT sample npz files.

This wrapper intentionally delegates metric math to OpenAI's ADM evaluator so
FID, sFID, Inception Score, Precision, and Recall follow that implementation.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


IMAGENET256_REF_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
    "ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
)


METRIC_PATTERNS = {
    "inception_score": re.compile(r"Inception Score:\s*([-+0-9.eE]+)"),
    "fid": re.compile(r"FID:\s*([-+0-9.eE]+)"),
    "sfid": re.compile(r"sFID:\s*([-+0-9.eE]+)"),
    "precision": re.compile(r"Precision:\s*([-+0-9.eE]+)"),
    "recall": re.compile(r"Recall:\s*([-+0-9.eE]+)"),
}


def require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def validate_sample_npz(path, expected_num_samples=None):
    path = require_file(path, "sample npz")
    with np.load(path) as data:
        if "arr_0" not in data:
            raise ValueError(f"{path} does not contain arr_0.")
        arr = data["arr_0"]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"{path} arr_0 must be NHWC RGB, got shape {arr.shape}.")
        if arr.dtype != np.uint8:
            raise ValueError(f"{path} arr_0 must be uint8 in [0, 255], got {arr.dtype}.")
        if expected_num_samples is not None and arr.shape[0] != expected_num_samples:
            raise ValueError(
                f"{path} contains {arr.shape[0]} samples, expected {expected_num_samples}."
            )
        return tuple(arr.shape)


def validate_reference_npz(path):
    path = require_file(path, "ADM reference npz")
    with np.load(path) as data:
        keys = set(data.keys())
    if not ({"mu", "sigma", "mu_s", "sigma_s"} <= keys or "arr_0" in keys):
        raise ValueError(
            f"{path} is not an ADM reference batch. Expected arr_0 or "
            "precomputed mu/sigma/mu_s/sigma_s."
        )
    return sorted(keys)


def resolve_evaluator(args):
    if args.evaluator:
        return require_file(args.evaluator, "ADM evaluator")
    candidates = [
        Path(args.guided_diffusion_root) / "evaluations" / "evaluator.py"
        if args.guided_diffusion_root
        else None,
        Path.cwd() / "guided-diffusion" / "evaluations" / "evaluator.py",
        Path.cwd() / "evaluations" / "evaluator.py",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find ADM evaluator.py. Provide --evaluator or "
        "--guided-diffusion-root. Expected OpenAI guided-diffusion/evaluations/evaluator.py."
    )


def check_runtime_deps():
    missing = []
    for module in ("tensorflow", "requests", "scipy", "tqdm"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "ADM evaluator dependencies are missing in this Python environment: "
            + ", ".join(missing)
            + ". Install the guided-diffusion evaluation environment before running ADM metrics."
        )


def parse_metrics(stdout):
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(stdout)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


class chdir:
    def __init__(self, path):
        self.path = path
        self.prev = None

    def __enter__(self):
        if self.path is None:
            return
        self.prev = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc, tb):
        if self.prev is not None:
            os.chdir(self.prev)


def load_evaluator_module(evaluator):
    spec = importlib.util.spec_from_file_location("adm_evaluator", evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_adm(args):
    args.sample_npz = Path(args.sample_npz).resolve()
    args.ref_npz = Path(args.ref_npz).resolve()
    if args.evaluator:
        args.evaluator = Path(args.evaluator).resolve()
    if args.guided_diffusion_root:
        args.guided_diffusion_root = Path(args.guided_diffusion_root).resolve()
    if args.inception_graph:
        args.inception_graph = Path(args.inception_graph).resolve()
    sample_shape = validate_sample_npz(args.sample_npz, args.num_samples)
    ref_keys = validate_reference_npz(args.ref_npz)
    evaluator = resolve_evaluator(args)
    check_runtime_deps()
    if args.inception_graph:
        require_file(args.inception_graph, "ADM Inception graph")
    if args.validate_only:
        result = {
            "sample_npz": str(args.sample_npz),
            "sample_shape": sample_shape,
            "ref_npz": str(args.ref_npz),
            "ref_keys": ref_keys,
            "evaluator": str(evaluator),
            "inception_graph": str(args.inception_graph) if args.inception_graph else None,
            "status": "ok",
        }
        print(json.dumps(result, indent=2))
        return result

    env = os.environ.copy()
    env["PYTHONPATH"] = str(evaluator.parent) + os.pathsep + env.get("PYTHONPATH", "")

    run_cwd = None
    tmp_dir = None
    if args.inception_graph:
        graph = require_file(args.inception_graph, "ADM Inception graph")
        run_cwd = graph.parent
        if graph.name != "classify_image_graph_def.pb":
            tmp_dir = tempfile.TemporaryDirectory()
            run_cwd = Path(tmp_dir.name)
            shutil.copy2(graph, run_cwd / "classify_image_graph_def.pb")

    if args.skip_precision_recall:
        module = load_evaluator_module(evaluator)
        config = module.tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with chdir(run_cwd):
            adm_eval = module.Evaluator(module.tf.Session(config=config))
            print("warming up TensorFlow...")
            adm_eval.warmup()
            print("computing/reading reference batch statistics...")
            ref_stats, ref_stats_spatial = adm_eval.read_statistics(str(args.ref_npz), None)
            print("computing sample batch activations...")
            sample_acts = adm_eval.read_activations(str(args.sample_npz))
            print("computing sample batch statistics...")
            sample_stats, sample_stats_spatial = adm_eval.read_statistics(str(args.sample_npz), sample_acts)
            metrics = {
                "inception_score": adm_eval.compute_inception_score(sample_acts[0]),
                "fid": sample_stats.frechet_distance(ref_stats),
                "sfid": sample_stats_spatial.frechet_distance(ref_stats_spatial),
            }
        if tmp_dir is not None:
            tmp_dir.cleanup()
        result = {
            "sample_npz": str(args.sample_npz),
            "sample_shape": sample_shape,
            "ref_npz": str(args.ref_npz),
            "ref_keys": ref_keys,
            "evaluator": str(evaluator),
            "metrics": metrics,
            "skipped_metrics": ["precision", "recall"],
        }
        print(json.dumps(result, indent=2))
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved ADM metrics JSON to {out}")
        return result

    cmd = [sys.executable, str(evaluator), str(args.ref_npz), str(args.sample_npz)]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env, cwd=run_cwd)
    if tmp_dir is not None:
        tmp_dir.cleanup()
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"ADM evaluator failed with exit code {proc.returncode}.")

    result = {
        "sample_npz": str(args.sample_npz),
        "sample_shape": sample_shape,
        "ref_npz": str(args.ref_npz),
        "ref_keys": ref_keys,
        "evaluator": str(evaluator),
        "metrics": parse_metrics(proc.stdout),
        "raw_stdout": proc.stdout,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved ADM metrics JSON to {out}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-npz", required=True, type=Path)
    parser.add_argument("--ref-npz", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--evaluator", type=Path, default=None)
    parser.add_argument("--guided-diffusion-root", type=Path, default=None)
    parser.add_argument("--inception-graph", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--skip-precision-recall",
        action="store_true",
        help="Compute ADM FID/sFID/Inception Score only; skip expensive Precision/Recall.",
    )
    parser.add_argument(
        "--print-imagenet256-ref-url",
        action="store_true",
        help="Print the official ADM ImageNet-256 reference batch URL and exit.",
    )
    args = parser.parse_args()
    if args.print_imagenet256_ref_url:
        print(IMAGENET256_REF_URL)
        return
    run_adm(args)


if __name__ == "__main__":
    main()
