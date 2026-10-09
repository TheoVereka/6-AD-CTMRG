# D8 two-C3 圆柱 S₂：Izar 的 30 个独立 job

J₂ 按以下优先顺序，每个 tensor 的三个 env pair 分别一个 job：

```text
0.26, 0.25, 0.265, 0.32, 0.20, 0.27, 0.245, 0.275, 0.24, 0.28
```

已复制并核验全部10份真实D8 checkpoint。其中9份来自 `0713summary`；**J₂=0.20 不在0713summary中**，使用 `data/0507core/2tensor_twoC3__J2_0p2_20260617_075601/sweep_D8_chi104_best.pt`。其 hyperparams 明确 J₁=1、J₂=0.2、ansatz=`2tensor_twoC3`，无pinning field。`seed_manifest.csv` 按优先顺序记录逐字节SHA256、来源和原始χ。全部 `a_raw/b_raw` 都是有限实数float64，shape `(8,8,8,2)`；运行时重新构造χ=80的CTM。

用户已提供当前job `3204848|i7p02m12|normal|3-00:00:00`，并明确指定前五个J₂三天、后五个七天。最终30份header已生成，`qos_allocation.json` 为 `ready=true`，分配固定如下：

| J₂（按提交优先顺序） | QOS | walltime | job数 |
|---|---|---|---:|
| 0.26, 0.25, 0.265, 0.32, 0.20 | normal | `71:59:50` | 15 |
| 0.27, 0.245, 0.275, 0.24, 0.28 | long | `167:59:50` | 15 |

每个QOS最多16个运行job；按用户提供的当前状态，normal为已有1+新增15=16，long为已有0+新增15=15。这里的状态来自用户提供的记录；本机没有成功通过SSH查询Izar。

上传整个目录后，在Izar查看计划和提交：

```bash
cd /你的路径/Renyi2Twoc3Izar
bash submit_all.sh --dry-run
bash submit_all.sh
```

也可在Izar提交前重新实测当时QOS容量：

```bash
bash submit_all.sh --refresh-qos --dry-run
bash submit_all.sh --refresh-qos
```

刷新使用 `squeue --user chye --states RUNNING --format '%i|%j|%q'`，只更新容量记录，**始终保留前五normal、后五long的15+15分配及用户顺序**。`qos_allocation.json` 保存查询时间、非敏感队列快照、各QOS数量及需要等待槽的最小job数。若已有更多job，会明确记录QOS槽不足；GPU资源与Slurm优先级也可能导致等待。若账号不同，可执行 `python3 prepare_jobs.py --auto --user 你的账号`。

`jobs/*.run` 各自带最终QOS和walltime：normal=`71:59:50`，long=`167:59:50`。共用实现 `run_pair.sh` 没有SBATCH header，仅负责一个J₂、一个pair的计算。提交器按 `job_manifest.csv` 的优先顺序逐条提交，先检查全部30份header再执行第一条sbatch。

资源沿用现有Izar案例：1张V100 32GB GPU、1个CPU core、40G host RAM，排除i39。加载 `gcc/11.3.0`、`cuda/11.8.0`，激活 `/home/chye/venvs/6adctmrg_Izar`；账户路径变化时修改 `run_pair.sh` 中的这一行。

本机D=2–6、χ=ceil(1.25D²) GPU验证已通过；D2验证三个pair并做dense全谱比较，D3–6验证pair1并做增模和独立起点复核。没有上传或提交任何job。

固定参数：`D=8, χ=80, float64, basis=64, block=4, batch=2, eig_tol=1e-9, entropy_tol=1e-4`。两个 replica 扇区都计算，使用保持内积的半存储坐标。`batch=2` 只切小输出腿工作集，不截断 tensor 或改变算法。

CTM 从正常 `(a,b)` 与交换 `(b,a)` 两个排列构造，使用 `full_svd`、`both` 收敛模式、`tol=1e-7`、energy threshold=`2e-8`；初始最多200步，达到步数上限时自动加倍重试2次。pair 映射如下：

| pair | 正常 CTM env | 交换 CTM env |
|---|---:|---:|
| 1 | 1 | 1 |
| 2 | 2 | 3 |
| 3 | 3 | 2 |

长度 `L=100,102,...,1000` 数切口上的 D-bond；transfer 的幂为 `L/2`，几何周长为 `3L/2` 个蜂窝边长。T₁、T₂ 的主谱依次请求每个扇区 `8→16→32` 个模；至少达到16模，且扩展前后全部长度的 S₂ 变化不超过 `1e-4`，才停止扩展。正常求谱过程中出现近零新方向会自动作秩判断、去除或补独立方向；达到计算限制时保存诊断，不把失败结果当成功。

输出互不覆盖：

```text
Results/J2_0p24/D_8/chi_80/pair_1/
    edges.npz     四个edge及CTM元数据，后续求谱可直接复用
    result.json   谱、残差、mode稳定性、状态、source hash、运行时间、GPU显存峰值
    result.csv    L,S2
slurm_logs/       每个job各自的stdout/error
```

可重新使用缓存 edge 进行求谱，保持 `--pair` 与缓存一致：

```bash
python -u code/renyi2_twoc3.py \
  --edge-file Results/J2_0p24/D_8/chi_80/pair_1/edges.npz \
  --pair 1 --chi 80 --device cuda \
  --subspace 64 --block-size 4 --batch 2 \
  --modes 8 16 32 --eig-tol 1e-9 --entropy-tol 1e-4 \
  --L-min 100 --L-max 1000 --L-step 2 --threads 1 \
  --output Results/J2_0p24/D_8/chi_80/pair_1/recomputed.json
```

`spectrally_stable_estimate` 表示已通过实际谱扩展稳定性判据；D8尚未在本机执行，不能提前保证某个 cluster job 的收敛时间或未知谱尾的严格上界。本机 D2 全谱对照、D3–6 的 mode/独立 seed/T₁全谱比较证据摘要保存在 `local_validation_reference.json`，完整原始证据保留在仓库专用验证文件夹。

代码依赖闭包只有 `code/` 下四份文件；不依赖原仓库位置或绘图文件。Izar现有venv中的 `torch/numpy/scipy/opt_einsum` 即为所需第三方包。代码哈希在 `code_manifest.sha256`。
