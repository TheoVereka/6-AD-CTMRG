"""Render the recorded local GPU evidence without running any numerics.

The running driver and production sources are never edited. Incomplete or
failed evidence is displayed explicitly and cannot produce a successful gate.
Each different snapshot is preserved alongside the current readable report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as error:
        return None, f"{path.name}: {type(error).__name__}: {error}"


def finite_number(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def scientific(value):
    return f"{value:.3e}" if finite_number(value) else "—"


def decimal(value, digits=2):
    return f"{value:.{digits}f}" if finite_number(value) else "—"


def maximum(values):
    valid = [value for value in values if finite_number(value)]
    return max(valid) if valid else None


def selected_stage(pair):
    valid = [stage for stage in pair.get("stages", []) if stage.get("eigensolvers_converged")
             and "S2_with_full_T1" in stage]
    return valid[-1] if valid else {}


def solver_records(pair, all_stages=False):
    stages = pair.get("stages", []) if all_stages else [selected_stage(pair)]
    records = []
    for stage in stages:
        if stage.get("T1_iterative"):
            records.append(stage["T1_iterative"])
        records.extend(stage.get("T2_sectors", []))
    records.extend(pair.get("independent_seed", {}).get("T2_sectors", []))
    return records


def case_summary(folder, d, summary_case=None):
    chi = math.ceil(1.25 * d**2)
    case_dir = folder / f"D{d}_chi{chi}"
    case, error = read_json(case_dir / "case.json")
    if case is None:
        case = summary_case or {"D": d, "chi": chi, "status": "not_started", "passed": False}
    pairs, errors = [], []
    expected_pairs = (1, 2, 3) if d == 2 else (1,)
    for number in expected_pairs:
        path = case_dir / f"pair{number}_validation.json"
        if not path.is_file():
            continue
        pair, pair_error = read_json(path)
        if pair is not None:
            pairs.append(pair)
        if pair_error:
            errors.append(pair_error)
    if error and case.get("status") != "not_started":
        errors.append(error)
    stages = [selected_stage(pair) for pair in pairs]
    solvers = [solver for pair in pairs for solver in solver_records(pair)]
    all_solvers = [solver for pair in pairs for solver in solver_records(pair, all_stages=True)]
    return {"D": d, "chi": chi, "status": case.get("status", "unknown"), "passed": case.get("passed", False),
            "seconds": case.get("seconds"), "peak_allocated_GiB": case.get("GPU_peak_allocated_GiB"),
            "peak_reserved_GiB": case.get("GPU_peak_reserved_GiB"),
            "pair_numbers": [pair["pair"] for pair in pairs],
            "mode_counts": [stage.get("requested_modes_per_sector") for stage in stages],
            "mode_change": maximum(stage.get("mode_increase_S2_max_absolute_change") for stage in stages),
            "seed_change": maximum(pair.get("independent_seed", {}).get("max_absolute_S2_change") for pair in pairs),
            "dense_error": maximum(stage.get("dense_reference_S2_max_absolute_error") for stage in stages),
            "T1_entropy_error": maximum(stage.get("T1_iterative_entropy_absolute_error") for stage in stages),
            "operator_error": maximum(check.get("matvec_relative_error") for pair in pairs
                                        for check in pair.get("packed_operator_checks", [])),
            "max_residual": maximum(value for solver in solvers for value in solver.get("relative_residuals", [])),
            "max_orthogonality_error": maximum(solver.get("orthogonality_error") for solver in all_solvers),
            "matvec_count": sum(solver.get("matvec_count", 0) for solver in all_solvers),
            "restart_count": sum(solver.get("restarts", 0) for solver in all_solvers),
            "rank_deflations": sum(solver.get("rank_deflations", 0) for solver in all_solvers),
            "random_completions": sum(solver.get("random_completions", 0) for solver in all_solvers),
            "gram_repairs": sum(solver.get("gram_repairs", 0) for solver in all_solvers),
            "ctm_steps": [item.get("steps") for item in case.get("ctm", {}).get("environments", [])],
            "ctm_retries": [item.get("retry_count", 0) for item in case.get("ctm", {}).get("environments", [])],
            "errors": errors,
            "criteria": [{"pair": pair["pair"], **pair.get("criteria", {})} for pair in pairs],
            "case_file": str(case_dir / "case.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=HERE)
    parser.add_argument("--final-cli", type=Path, default=None,
                        help="Optional final production CLI validation JSON; auto-detect the standard path when present.")
    args = parser.parse_args(argv)
    folder = args.folder.resolve()
    summary, summary_error = read_json(folder / "validation_summary.json")
    summary = summary or {}
    requested = summary.get("requested_D", [2, 3, 4, 5, 6])
    existing = {item["D"]: item for item in summary.get("cases", [])}
    cases = [case_summary(folder, d, existing.get(d)) for d in requested]
    full_gate = bool(summary.get("full_D2_to_D6_gate_passed", False)
                     and set(requested) == {2, 3, 4, 5, 6}
                     and len(summary.get("cases", [])) == 5
                     and all(item["passed"] for item in cases)
                     and not summary_error)
    runtime = summary.get("runtime", {})
    cli_path = args.final_cli.resolve() if args.final_cli else folder / "production_final" / "final_cli_validation.json"
    cli, cli_error = read_json(cli_path) if cli_path.is_file() else (None, None)
    cli_passed = bool(cli and cli.get("passed", False) and not cli_error)
    snapshot = {"generated_UTC": datetime.now(timezone.utc).isoformat(), "full_gate_passed": full_gate,
                "summary_read_error": summary_error, "runtime": runtime, "cases": cases,
                "total_seconds": summary.get("total_seconds"),
                "source_hashes": runtime.get("source_files_sha256", {}),
                "final_cli_validation_path": str(cli_path) if cli_path.is_file() else None,
                "final_cli_validation": cli, "final_cli_passed": cli_passed,
                "final_cli_read_error": cli_error}
    # Evidence snapshots are content-addressed; reruns preserve every distinct
    # partial/failed/successful state without modifying the driver evidence.
    content = json.dumps({key: value for key, value in snapshot.items() if key != "generated_UTC"},
                         ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()[:12]
    snapshot_dir = folder / "report_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    prefix = "passed" if full_gate else "not_passed"
    snapshot_path = snapshot_dir / f"{prefix}_{digest}.json"
    if not snapshot_path.is_file():
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "全部五个 D 的本机 GPU 测试通过；条件 gate=True。" if full_gate else "完整测试 gate 尚未通过；不得据此开始条件后续工作。"
    lines = ["# 本机 GPU 的 Krylov 与大 L 熵验证", "", f"**{status}**", "",
             f"记录时间（UTC）：{snapshot['generated_UTC']}。", "",
             f"GPU：{runtime.get('GPU_name', '未记录')}；PyTorch `{runtime.get('torch_version', '未记录')}`，CUDA `{runtime.get('torch_cuda_version', '未记录')}`。",
             "实际输入全部来自 `0713summary/J2_0p26/2tensor_twoC3`，没有用随机 tensor 替代物理输入。χ 取 `ceil(1.25 D²)`。", "",
             "| D | χ | 状态 | 总耗时/s | 峰值显存/GiB | 增加模数的最大 ΔS₂ | 独立 seed 最大 ΔS₂ | D2 完整 dense 最大误差 |",
             "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        lines.append(f"| {case['D']} | {case['chi']} | {case['status']} | {decimal(case['seconds'], 1)} | {decimal(case['peak_allocated_GiB'])} | {scientific(case['mode_change'])} | {scientific(case['seed_change'])} | {scientific(case['dense_error'])} |")
    lines += ["", "显存列是 PyTorch peak allocated；下表另记录 peak reserved。总耗时包含两次 CTMRG、算符检查、dense 对照及全部求谱轮次。", "",
              "对每个 D，normal/swapped CTMRG 各算一次并保存三个 pair；D2 测全部三个 pair，D3–6 测 pair1。全部比较覆盖 451 个偶数长度 L=100,102,…,1000，阈值为绝对熵误差 `1e-4`。",
              "64 维最大 basis、block4、float64；先比较每扇区 8→16 模，必要时 32 模，再以独立 seed 重新求谱。T1 使用完整 dense 谱作对照；D2χ5 的 T2 用独立 NumPy 收缩显式构造 625×625 矩阵并求完整谱。另检查 packed 坐标往返、范数、replica parity 和 matvec。",
              "D2 有独立完整谱误差；较大 D 的误差证据是增加模数及独立 seed 的稳定性，不把它表述为未知漏谱的严格数学上界。", "",
              "## 求谱与资源记录", "",
              "| D | 实测 pair | 最终每扇区模数 | T1 导致的最大 ΔS₂ | 最大 Ritz 残差 | 最大 ‖QᵀQ−I‖ | 总 matvec | 总 restart | peak reserved/GiB | normal/swap CTM 步数 |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for case in cases:
        pairs = ",".join(map(str, case["pair_numbers"])) or "—"
        modes = ",".join(map(str, case["mode_counts"])) or "—"
        steps = "/".join(map(str, case["ctm_steps"])) or "—"
        lines.append(f"| {case['D']} | {pairs} | {modes} | {scientific(case['T1_entropy_error'])} | {scientific(case['max_residual'])} | {scientific(case['max_orthogonality_error'])} | {case['matvec_count']} | {case['restart_count']} | {decimal(case['peak_reserved_GiB'])} | {steps} |")
    lines += ["", "Ritz 残差按主谱尺度归一化；该列取最终轮与独立 seed 复算的最大值。正交误差和 matvec/restart 统计涵盖所有已记录轮次。随机向量仅用于算符测试与迭代起点，输入 edge tensor 均来自真实 CTMRG。", "",
              "近零候选由秩判断丢弃，必要时补独立方向；这些是可恢复的内部操作，不会被直接当成整个算例失败。CTMRG 达步数上限会有限重试；最终仍不满足要求或 OOM 才记录 notpassed。", ""]
    if cli is not None:
        try:
            cli_link = cli_path.relative_to(folder).as_posix()
        except ValueError:
            cli_link = str(cli_path)
        lines += ["## 最终生产 CLI 回归", "",
                  f"最终入口对照状态：`passed={str(cli_passed).lower()}`；[原始 CLI 验证记录]({cli_link})。该状态与上面的五个 D 完整 GPU gate 分别记录。", "",
                  "| D | χ | device | 实际执行模数 | 耗时/s | 最大绝对 S₂ 差 | 对照参考 | 状态 |",
                  "|---:|---:|---|---|---:|---:|---|---|"]
        cli_notes = []
        for case in cli.get("cases", []):
            case_runtime = case.get("runtime", {})
            modes = ",".join(map(str, case.get("executed_modes", []))) or "—"
            lines.append(f"| {case.get('D', '—')} | {case.get('chi', '—')} | {case_runtime.get('device', '—')} | {modes} | {decimal(case_runtime.get('seconds'))} | {scientific(case.get('max_absolute_S2_error'))} | {case.get('reference', '未记录')} | {case.get('status', '未记录')} |")
            for note_key in ("comparison_note", "precision_note", "cross_device_note", "note"):
                if case.get(note_key):
                    cli_notes += ["", f"D={case.get('D')}：{case[note_key]}", ""]
            cross_device_difference = case.get("CPU_vs_GPU_CTM_S2_difference")
            if finite_number(cross_device_difference):
                cli_notes += ["", f"D={case.get('D')}：CPU/GPU 各自重算 CTMRG 后的最大 S₂ 差为 "
                          f"`{scientific(cross_device_difference)}`；上表的同一份 edge 求谱误差为 "
                          f"`{scientific(case.get('max_absolute_S2_error'))}`。两种差异分别记录。", ""]
        lines.extend(cli_notes)
        lines += ["", "完整五个 D 的 GPU 测试之后，主代码只增加稳定后提前结束的流程、source hash 与 runtime/VRAM 日志；求谱器、T₁/T₂ 收缩和 packed 坐标未改变。最终 GPU D4 入口另检查 batch2。CPU 与 GPU 各自重新运行 CTMRG 的对照差异，不等同于固定同一份 edge 时的求谱误差；应按原始记录的参考对象区分。", "",
                  "D8 尚未在本机做同等规模验证。", ""]
    elif cli_error:
        lines += ["## 最终生产 CLI 回归", "", f"最终 CLI 记录尚未可读：`{cli_error}`；不能宣称该项通过。", ""]
    if summary_error:
        lines += [f"读取进行中的 summary 时发生：`{summary_error}`。这是未完成快照，不是通过结果。", ""]
    for case in cases:
        if case["errors"]:
            lines += [f"D={case['D']} 的未完成/读取诊断：", ""]
            lines += [f"- `{error}`" for error in case["errors"]]
            lines.append("")
    lines += ["## 原始证据", "", "- [总状态与运行参数](validation_summary.json)",
              f"- [本报告的不可覆盖快照](report_snapshots/{snapshot_path.name})"]
    for case in cases:
        relative = Path(case["case_file"]).relative_to(folder).as_posix()
        lines.append(f"- [D={case['D']} 详细记录]({relative})")
    lines += ["", "验证记录保留了实际 checkpoint 的 SHA256、manifest 来源与本次 driver/生产脚本 SHA256。此报告生成器只读原始结果，不运行 GPU、不创建 cluster 文件、不删除失败或中间证据。", ""]
    report = "\n".join(lines)
    (folder / "本机GPU验证结果.md").write_text(report, encoding="utf-8")
    markdown_snapshot = snapshot_path.with_suffix(".md")
    if not markdown_snapshot.is_file():
        markdown_snapshot.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(folder / "本机GPU验证结果.md"), "snapshot": str(snapshot_path),
                      "full_gate_passed": full_gate,
                      "final_cli_passed": cli_passed,
                      "cases": [{key: case[key] for key in ("D", "chi", "status", "seconds", "mode_change", "seed_change", "dense_error")}
                                for case in cases]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
