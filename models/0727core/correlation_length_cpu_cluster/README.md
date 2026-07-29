# CPU-cluster correlation-length bundle

这个目录名就是集群上的默认 `somefoldername`：
`correlation_length_cpu_cluster`。

## 1. Windows 本地收集 checkpoint

在仓库根目录运行：

```powershell
D:\Programs\Python312\python.exe models\0727core\correlation_length_cpu_cluster\collect_checkpoints.py
```

它扫描 `0713summary/J2_*/2tensor_twoC3/D_*/tensor_best.pt`，复制到本目录的
`checkpoints/` 并重命名为：

```text
tensor_best__J2_0p24__D_7.pt
```

同时生成 JSON 和 TSV 两份 manifest。重复运行时，相同 SHA-256 的文件跳过；
内容不同的同名文件默认拒绝覆盖，确认后可加 `--overwrite`。
manifest 还记录 `correlation_length.py` 与 `core_C3.py` 的 SHA-256；每个集群
worker 开始前都会核对 `/home` 中实际执行的源码，防止 `/home`、`/scratch`
版本混用。

先只检查、不复制：

```powershell
D:\Programs\Python312\python.exe models\0727core\correlation_length_cpu_cluster\collect_checkpoints.py --dry-run
```

完成后，把整个目录同步到以下两个位置，并保证开跑前内容相同：

```text
/home/chye/correlation_length_cpu_cluster/
/scratch/chye/correlation_length_cpu_cluster/
```

`/home` 副本提供 Python 源码，`/scratch` 副本提供 checkpoint、manifest、Slurm
脚本和计算输出。

## 2. 在 CPU cluster 提交

从 `/scratch/chye/correlation_length_cpu_cluster` 运行：

```bash
bash submit_correlation_lengths.sh --dry-run
bash submit_correlation_lengths.sh
```

自动提交 manifest 中全部现有 `D>=7`：

```bash
bash submit_correlation_lengths.sh --min-D 7 --dry-run
bash submit_correlation_lengths.sh --min-D 7
```

提交脚本会把当前 bundle 目录名作为第三个参数显式传给
`correlation_length_job.run`。job 内由此固定访问
`/home/chye/<bundle>/run_one_correlation_length.py` 和
`/scratch/chye/<bundle>/`；不要用 Slurm 临时脚本所在目录推导 bundle 名。

默认选择 manifest 中的所有 J2 与 `D=7,8,9,10` 的笛卡尔积。不存在的组合只报告
`MISSING`，不会提交。每个实际存在的 `.pt` 单独执行一次 `sbatch`。已经存在且通过
内容验证的结果默认跳过。

指定集合的例子：

```bash
bash submit_correlation_lengths.sh --J2 0.24,0.26,0.30 --D 7,9
```

输出位于：

```text
/scratch/chye/correlation_length_cpu_cluster/results/
correlation_length__J2_0p24__D_7.json
```

`.run` 使用 academic 队列、16 CPU、100 GB 内存和
`/home/chye/venvs/6adctmrg`。没有向 `correlation_length.py` 传入固定 seed；
当前 solver 会在每个 job 中从系统熵生成新的 32-bit seed，并把实际 seed 和
`seed_was_user_specified=false` 写入 JSON。CTMRG/rSVD 与 ARPACK 起始向量在同一
job 中使用这个已记录的 seed。

solver 的统一 bond-dimension 策略是：拒绝 D=2 correlation length；D=3、4
强制使用 full-SVD CTMRG truncation；D>=5 保留调用者选择的 SVD 模式。本 bundle
默认提交 D=7–10，因此其生产任务仍使用原有 augmented rSVD 默认值。

## 3. scp 回本地并归位

把整个 scratch bundle 递归复制到本地 data 目录：

```powershell
scp -r chye@CLUSTER:/scratch/chye/correlation_length_cpu_cluster D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\
```

这会形成
`D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\correlation_length_cpu_cluster`。
无参数运行 importer 即可从其中的 `results/` 读取结果、使用其中的 manifest 验证，
然后归位：

```powershell
D:\Programs\Python312\python.exe models\0727core\correlation_length_cpu_cluster\import_completed_results.py
```

脚本以 manifest、文件名、J2、D、两枚特征值和 checkpoint SHA-256 多重校验，然后
写入对应的：

```text
0713summary/J2_.../2tensor_twoC3/D_.../correlation_length.json
```

写入成功后才删除下载目录中的源 JSON，因此默认是“移动”。目标已经存在时默认拒绝；
明确需要替换时加 `--overwrite`。要保留下载副本则加 `--keep-source`。
