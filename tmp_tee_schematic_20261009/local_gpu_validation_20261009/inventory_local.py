"""Read-only inventory of installed runtimes and actual 0713summary checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SUMMARY = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary")
CORE = SUMMARY.with_name("0713core")


def inspect_checkpoint(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = directory / "tensor_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tensors = {}
    for name in ("a_raw", "b_raw", "a", "b"):
        if name in checkpoint:
            tensor = checkpoint[name]
            tensors[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "is_complex": tensor.is_complex(), "finite": bool(torch.isfinite(tensor).all())}
    if not tensors and "params" in checkpoint:
        for number, tensor in enumerate(checkpoint["params"]):
            tensors[f"params[{number}]"] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                           "is_complex": tensor.is_complex(), "finite": bool(torch.isfinite(tensor).all())}
    original_filename = f"sweep_D{manifest['D']}_chi{manifest['chi']}_best.pt"
    original_path = CORE / manifest["source_job"] / original_filename
    def json_safe(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else str(value)
        return value
    return {
        "D": manifest["D"], "J2": manifest["J2"], "optimized_chi": manifest["chi"],
        "requested_test_chi_float": 1.25 * manifest["D"] ** 2,
        "ceil_test_chi": math.ceil(1.25 * manifest["D"] ** 2),
        "checkpoint_path": str(checkpoint_path), "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "source_job": manifest["source_job"], "original_filename": original_filename,
        "original_path": str(original_path), "original_exists": original_path.is_file(),
        "manifest": manifest, "checkpoint_keys": list(checkpoint), "tensor_metadata": tensors,
        "checkpoint_metadata": {key: json_safe(checkpoint.get(key))
                                for key in ("D_bond", "chi", "loss", "energy", "step", "timestamp")},
    }


def main():
    preferred = [inspect_checkpoint(SUMMARY / "J2_0p26" / "2tensor_twoC3" / f"D_{d}")
                 for d in range(2, 7)]
    target_d8 = []
    for directory in sorted(SUMMARY.glob("J2_*/2tensor_twoC3/D_8")):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if 0.24 - 1e-12 <= manifest["J2"] <= 0.275 + 1e-12:
            target_d8.append(inspect_checkpoint(directory))
    disk = {drive: dict(zip(("total_bytes", "used_bytes", "free_bytes"), shutil.disk_usage(drive)))
            for drive in ("C:/", "D:/")}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    inventory = {
        "python": sys.executable, "python_version": sys.version, "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda, "torch_cuda_available": torch.cuda.is_available(),
        "nvidia_smi": gpu.stdout.strip(), "disk": disk, "summary_root": str(SUMMARY),
        "preferred_test_checkpoints": preferred, "D8_target_checkpoints": target_d8,
        "D8_tensor_count": len(target_d8), "eventual_three_pair_job_count": 3 * len(target_d8),
        "mutations": "Only this dedicated inventory JSON/Markdown output; no environment, data or cluster files changed.",
    }
    (HERE / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Local GPU and true checkpoint inventory", "",
             f"GPU: `{gpu.stdout.strip()}`.",
             f"Default Python: `{sys.executable}`; torch `{torch.__version__}`; CUDA available: `{torch.cuda.is_available()}`.",
             "Python 3.10 exists at D:/Programs/Python310/python.exe but has no torch. No conda or repository venv was found.",
             "", "## Actual local-test tensors", "",
             "All five are selected 0713summary two-C3 checkpoints at J2=0.26, with finite real float64 a_raw/b_raw tensors.",
             "Fractional 1.25 D² for odd D requires integer convention; the table records ceiling, without imposing it on production code.",
             "", "| D | 1.25 D² | ceiling χ | optimized χ | original file |", "|---:|---:|---:|---:|---|",]
    for item in preferred:
        lines.append(f"| {item['D']} | {item['requested_test_chi_float']} | {item['ceil_test_chi']} | {item['optimized_chi']} | `{item['source_job']}/{item['original_filename']}` |")
    lines += ["", "## Eventual D8 target inventory", "",
              f"{len(target_d8)} actual tensors in inclusive J2=[0.24,0.275], hence {3*len(target_d8)} eventual jobs if and only if the test gate passes.", ""]
    for item in target_d8:
        lines.append(f"- J2={item['J2']}: `{item['checkpoint_path']}`; optimized χ={item['optimized_chi']}.")
    lines += ["", "## Existing environment and disk", "",
              "The GPU and driver are present, but default PyTorch is CPU-only. A CUDA-capable isolated environment is needed for actual GPU tests; no package was installed during this inventory.",
              "Driver 551.23 advertises CUDA 12.4. An official PyTorch 2.6 cu124 build is the matching candidate; verify its import/device probe before running tests.",
              "An isolated env can reuse CPU-installed scientific packages via system-site-packages, with torch overridden inside that env, to avoid mutating the existing global environment.", ""]
    for drive, values in disk.items():
        lines.append(f"- {drive}: free {values['free_bytes']/2**30:.2f} GiB, total {values['total_bytes']/2**30:.2f} GiB.")
    lines += ["", "No cluster job was created. No checkpoint, source manifest or global environment was modified.", ""]
    (HERE / "inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"preferred": [{key: item[key] for key in ("D", "J2", "optimized_chi", "ceil_test_chi", "checkpoint_path", "original_filename", "tensor_metadata")} for item in preferred],
                      "target_D8_J2": [item["J2"] for item in target_d8], "target_D8_count": len(target_d8),
                      "nvidia_smi": inventory["nvidia_smi"], "disk": disk}, indent=2))


if __name__ == "__main__":
    main()
