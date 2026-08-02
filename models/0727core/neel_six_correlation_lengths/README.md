# Néel six-correlation-length cluster bundle

这个目录是完整、可重复上传的 cluster bundle。正常流程只有三条命令：本地收集、
cluster 提交、本地画图。

## 1. 本地收集 checkpoints

在本目录运行：

```powershell
python .\collect_checkpoints.py
```

脚本动态收集：

- `data/D345678910` 中所有 `neel_symmetrized__J2_*` 的可用 `D=3..10`；
- `data/0713summary/J2_*/` 中所有名字以 `neel` 开头的 ansatz（包括
  `neel_free_param` 和 `neel_symmetrized`）。

每个 `(J2,D)` 只保留一个 checkpoint。`0713summary` 与 `D345678910` 冲突时
总是以前者为准；summary 自身有多个 Néel ansatz 时选择记录 energy 最低的一个。
所有来源、hash、覆盖关系和无法匹配的旧 observable 都记录在自动生成的
`checkpoint_manifest.json` 中。每次收集都会完整重建 `checkpoints/`，不会遗留已经
从 summary 删除的旧 case。

## 2. 上传并一次性 sbatch

从仓库根目录上传整个文件夹：

```powershell
scp -r models\0727core\neel_six_correlation_lengths chye@CLUSTER:/scratch/chye/
```

在 cluster 上运行：

```bash
cd /scratch/chye/neel_six_correlation_lengths
bash submit_cluster.sh --dry-run
bash submit_cluster.sh
```

第二条命令会为每个 checkpoint 单独 `sbatch`。有效结果写入 `results/`，Slurm
输出写入 `logs/`。结果 JSON 保存输入 checkpoint 的 SHA-256；因此这个文件夹无论
上传、运行过多少次，都只跳过仍与当前 checkpoint 完全对应的已完成结果。输入被
summary 覆盖、旧结果不完整或 CTMRG 未收敛时会自动重新提交。
同一 `(J2,D)` 的重复提交带 Slurm `singleton` dependency，不会同时重复计算；前一
个 job 失败时后一个仍会接着尝试，前一个成功时后一个会在计算入口直接跳过。

## 3. 下载并画全部 figures

可以下载整个 bundle，也可以只把 `results/` 合并回本目录：

```powershell
scp -r chye@CLUSTER:/scratch/chye/neel_six_correlation_lengths/results models\0727core\neel_six_correlation_lengths\
python models\0727core\neel_six_correlation_lengths\plot_six_inverse_xi.py
```

图片统一写到 `figures/`（PDF 和 PNG 各一份）：

- `all_J2_generalized_inverse_xi.*`：所有 J2 的 generalized `1/xi`；
- `all_J2_ordinary_inverse_xi.*`：所有 J2 的 ordinary `1/xi`；
- `per_J2/J2_*_inverse_xi.*`：每个 J2 一张、同时包含 generalized 和 ordinary。

每个 `(J2,D)` 的三个几何方向 `1/xi` 先排序，中心点取中位数，errorbar 的下/上端
分别取最小值/最大值。overview 中不同 J2 使用不同且在两张图间一致的颜色。所有
per-J2 图使用由全体 J2、两种 eigenproblem 共同计算出的相同 `xlim` 和 `ylim`。
