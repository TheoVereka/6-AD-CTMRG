# 本机 GPU 的 Krylov 与大 L 熵验证

**全部五个 D 的本机 GPU 测试通过；条件 gate=True。**

记录时间（UTC）：2026-10-09T17:09:42.967260+00:00。

GPU：NVIDIA GeForce RTX 4060 Laptop GPU；PyTorch `2.6.0+cu124`，CUDA `12.4`。
实际输入全部来自 `0713summary/J2_0p26/2tensor_twoC3`，没有用随机 tensor 替代物理输入。χ 取 `ceil(1.25 D²)`。

| D | χ | 状态 | 总耗时/s | 峰值显存/GiB | 增加模数的最大 ΔS₂ | 独立 seed 最大 ΔS₂ | D2 完整 dense 最大误差 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | 5 | passed | 5.7 | 0.01 | 0.000e+00 | 9.095e-13 | 0.000e+00 |
| 3 | 12 | passed | 4.2 | 0.03 | 0.000e+00 | 1.819e-12 | — |
| 4 | 20 | passed | 17.6 | 0.19 | 9.095e-13 | 2.728e-12 | — |
| 5 | 32 | passed | 186.6 | 0.65 | 1.819e-12 | 1.819e-12 | — |
| 6 | 45 | passed | 1517.2 | 2.49 | 3.638e-12 | 1.819e-12 | — |

显存列是 PyTorch peak allocated；下表另记录 peak reserved。总耗时包含两次 CTMRG、算符检查、dense 对照及全部求谱轮次。

对每个 D，normal/swapped CTMRG 各算一次并保存三个 pair；D2 测全部三个 pair，D3–6 测 pair1。全部比较覆盖 451 个偶数长度 L=100,102,…,1000，阈值为绝对熵误差 `1e-4`。
64 维最大 basis、block4、float64；先比较每扇区 8→16 模，必要时 32 模，再以独立 seed 重新求谱。T1 使用完整 dense 谱作对照；D2χ5 的 T2 用独立 NumPy 收缩显式构造 625×625 矩阵并求完整谱。另检查 packed 坐标往返、范数、replica parity 和 matvec。
D2 有独立完整谱误差；较大 D 的误差证据是增加模数及独立 seed 的稳定性，不把它表述为未知漏谱的严格数学上界。

## 求谱与资源记录

| D | 实测 pair | 最终每扇区模数 | T1 导致的最大 ΔS₂ | 最大 Ritz 残差 | 最大 ‖QᵀQ−I‖ | 总 matvec | 总 restart | peak reserved/GiB | normal/swap CTM 步数 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 1,2,3 | 16,16,16 | 1.819e-12 | 6.171e-10 | 9.369e-15 | 855 | 0 | 0.02 | 6/6 |
| 3 | 1 | 16 | 0.000e+00 | 5.726e-10 | 2.380e-14 | 689 | 9 | 0.04 | 14/13 |
| 4 | 1 | 16 | 0.000e+00 | 9.261e-10 | 2.731e-14 | 727 | 10 | 0.26 | 13/13 |
| 5 | 1 | 16 | 5.457e-12 | 9.018e-10 | 3.020e-14 | 740 | 10 | 0.79 | 17/12 |
| 6 | 1 | 16 | 0.000e+00 | 8.885e-10 | 4.975e-14 | 746 | 11 | 2.63 | 14/14 |

Ritz 残差按主谱尺度归一化；该列取最终轮与独立 seed 复算的最大值。正交误差和 matvec/restart 统计涵盖所有已记录轮次。随机向量仅用于算符测试与迭代起点，输入 edge tensor 均来自真实 CTMRG。

近零候选由秩判断丢弃，必要时补独立方向；这些是可恢复的内部操作，不会被直接当成整个算例失败。CTMRG 达步数上限会有限重试；最终仍不满足要求或 OOM 才记录 notpassed。

## 最终生产 CLI 回归

最终入口对照状态：`passed=true`；[原始 CLI 验证记录](production_final/final_cli_validation.json)。该状态与上面的五个 D 完整 GPU gate 分别记录。

| D | χ | device | 实际执行模数 | 耗时/s | 最大绝对 S₂ 差 | 对照参考 | 状态 |
|---:|---:|---|---|---:|---:|---|---|
| 2 | 5 | cpu | 8,16 | 1.27 | 9.095e-13 | independent_full_dense_T1_T2_same_CPU_edges | spectrally_stable_estimate |
| 4 | 20 | cuda:0 | 8,16 | 10.32 | 0.000e+00 | actual_GPU_driver_full_T1 | spectrally_stable_estimate |

D=2：CPU/GPU 各自重算 CTMRG 后的最大 S₂ 差为 `9.044e-08`；上表的同一份 edge 求谱误差为 `9.095e-13`。两种差异分别记录。


完整五个 D 的 GPU 测试之后，主代码只增加稳定后提前结束的流程、source hash 与 runtime/VRAM 日志；求谱器、T₁/T₂ 收缩和 packed 坐标未改变。最终 GPU D4 入口另检查 batch2。CPU 与 GPU 各自重新运行 CTMRG 的对照差异，不等同于固定同一份 edge 时的求谱误差；应按原始记录的参考对象区分。

D8 尚未在本机做同等规模验证。

## 原始证据

- [总状态与运行参数](validation_summary.json)
- [本报告的不可覆盖快照](report_snapshots/passed_5ecd9f5a4972.json)
- [D=2 详细记录](D2_chi5/case.json)
- [D=3 详细记录](D3_chi12/case.json)
- [D=4 详细记录](D4_chi20/case.json)
- [D=5 详细记录](D5_chi32/case.json)
- [D=6 详细记录](D6_chi45/case.json)

验证记录保留了实际 checkpoint 的 SHA256、manifest 来源与本次 driver/生产脚本 SHA256。此报告生成器只读原始结果，不运行 GPU、不创建 cluster 文件、不删除失败或中间证据。
