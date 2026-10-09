# Krylov修复与回归（2026-10-09）

生产文件：`src_code/scripts/renyi2_spectral.py`。`solve_block`调用接口保持不变，默认容量改为64；结果新增`rank_deflations`、`random_completions`、`gram_repairs`计数。全部basis和matvec仍为real float64，允许非Hermitian算子和复共轭本征值。

## 具体修复

1. **近breakdown不再归一化成假方向。** 候选先投影回目标交换扇区，再两轮减去旧基。thin QR后只对小块R做SVD，按输入块尺度的`4096*eps`判秩。归一化后再次扇区投影、两轮外部正交和小块判秩。判据没有与算子无关的绝对`1e-14`门槛。
2. **秩下降正常恢复。** 自动补充独立随机扇区方向并记录计数。有限投影空间完全耗尽时，检查已有Ritz残差后返回结果；不因这个正常数值事件抛异常。
3. **弱根停止尺度正确。** 残差分母至少为`2*leading_modulus*||x||`；不会要求一个对大L的迹没有贡献的小根达到相对其自身的苛刻误差。
4. **不再把全部近零根当成一个巨大简并簇。** 簇边界比较以边界根自身的模为尺度；仍保留等模簇和复共轭对。任意数量的严格零根不需要全部保留。
5. **重启检查Gram。** 实Schur保留子空间后检查`QᵀQ`；必要时用小Gram的逆平方根同步变换Q及AQ，并重算投影矩阵。没有第二份完整basis。最终若Gram仍异常，明确返回未收敛。

无贡献弱根、正常秩下降和少量舍入误差均不报错。非法维数、非有限matvec等真正无法使用的输入仍会明确报错。达到迭代预算会返回失败状态，不能冒充成功。

## 已执行结果

### 原始故障输入与全谱

`solver_before_after.py`读取保留的旧求谱器副本，不修改历史证据。原80维basis、8modes、同一份tiny pair1：旧求谱器两个扇区均失败，Gram误差分别65.34和276.48；修复后分别16和20次matvec收敛，Gram约`2e-15`。

`solver_breakdown_regression.py`对三个tiny checkpoint pair分别使用8、16、32modes，64维basis，与独立显式T₁/T₂全谱比较**全部451个偶数L=100…1000**。最大熵绝对误差为`2.7284841053187847e-12`。pair3的32modes实际执行过一次restart，最终Gram误差`1.49e-14`。同时验证算子整体尺度`1e-100, 1, 1e100`不影响判秩结果。

### 重启、简并、共轭与投影空间

`run_solver_regressions.py`执行原有production检查并将结果另存于本目录：normal矩阵、5重精确简并、复共轭对、非normal矩阵、正交投影、预算失败与稳定谱求幂全部通过。前四个主测试分别执行9、9、4和11次restart，最大本征值匹配误差`1.91e-14`。

`solver_multiplicity_probe.py`进一步检查7重主根而起始block只有4列的情形：通过自动补方向恢复全部7重数；小block需要1413次matvec，block8只需205次。投影空间维数3和35完全耗尽时均正常收敛，Gram不超过`3e-15`。

以上是求谱器及小尺寸独立dense回归。实际D=2…6的本机GPU结果由同目录的GPU driver及报告另行记录；这些结果通过前，不能用本文替代用户要求的整体验证门槛。

## 重跑命令

```powershell
python -B .\tmp_tee_schematic_20261009\local_gpu_validation_20261009\solver_before_after.py
python -B .\tmp_tee_schematic_20261009\local_gpu_validation_20261009\run_solver_regressions.py
python -B .\tmp_tee_schematic_20261009\local_gpu_validation_20261009\solver_breakdown_regression.py
python -B .\tmp_tee_schematic_20261009\local_gpu_validation_20261009\solver_multiplicity_probe.py
```

依然保留一个明确的数学边界：小残差证明当前Ritz向量的向后误差小，不能单独严格证明未知省略谱、全部简并重数或高度非normal问题的前向本征值误差。大L的S₂精度要以dense参考或增加模数/独立种子后的实际熵稳定性继续验证。
