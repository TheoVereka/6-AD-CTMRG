# 新四张量方案：indexing、原始 PEPS 小圆柱核对、内存

本次完整阅读了 [我们的数值背景.txt](我们的数值背景.txt)。该文件是一份待核验的算法说明，不把其中的运行建议或时间估计当作本次用户指令。以下独立检查项目当前源码，未修改生产代码。

## 1. 先固定三个不能混用的长度与维数

- 原始 PEPS `a,b` 的轴是 `(alpha,beta,gamma,spin)`，物理轴最后；虚腿维数是原始 bond dimension。
- 文件中四张量 `A,B,C,D` 的两条 `8` 维腿，是打开双层切口后的两条原始虚腿；`8` 对应原始 bond dimension 为 8，不能泛化成对所有输入都固定 8。另两条腿是 CTMRG 的 `chi`。
- `L` 是边界上含两个位置的重复胞数；`S_2(2L)` 中 `2L` 数切口上的原始虚键。每重复胞包含左侧 `d,a` 和右侧 `c,b` 两对位置。本图六边形边长为 1 时，两胞内切口相交点间隔均为 1.5，沿 y 的几何周长为 `3L`。把切口键数直接称为以六边形边长计的几何周长会差一个系数。

新背景第 13—35 行把 `S_2(2L)` 改写为另一套记号 `S_2(L)`，幂也改成 `L/2`；代数上一致，但本项目建议保留用户原来的 `S_2(2L)`，幂写 `L`。

源码依据：[core_C3.py](../src_code/scripts/core_C3.py:1294) 的 `twoc3_abcdef_from_ab` 给出循环腿排列；[core.py](../src_code/scripts/core.py:1727) 的双层构造把 `(ket,bra)` 依次融合成 `D_squared`；[correlation_length.py](../src_code/scripts/correlation_length.py:646) 检查实际 edge shape 为 `(chi,chi,D_squared)`。

## 2. 四张量不是左链上的四个相邻站点

新背景中的 `AB` 把两个张量的原始虚腿相连。因此：

```text
同一行：        左边 A ---- 两条 D 腿 ---- B 右边
下一行：        左边 D ---- 两条 D 腿 ---- C 右边

沿 y 重复：     (A,D),(A,D),...    和    (B,C),(B,C),...
```

`AB` 是第一行左右两侧的相连网络，`DC` 是下一行的相连网络。它们不是左边两个相邻方形各自相乘。左、右各自的 chi 腿才沿 y 连接。

**图中的 cut 是交替 gamma/beta 的 armchair cut。直接拿 `correlation_length.obtain_4Ts` 的输出按颜色放进去不正确。** 该函数特意额外执行 `env1 -> env2`，返回 `T1D,T2C`；这些是另一个所需取向的 edge。

当前源码的 CTMRG 返回顺序为：

```text
env1: C21CD, T1F, T2A
env2: C21EB, T1D, T2C
env3: C21AF, T1B, T2E
```

本图左侧 `d(gamma),a(beta)` 应取 env1 的 `T1F,T2A`，其中 `T1F` 通过 C3 是 `T3D` 的代表。右边使用交换原始 a/b 后 env1 的同一对，并反转沿边界的 chi 传播方向。此处 **右侧反向** 很重要；小环长度为 2 时容易看不出这个错误，长度为 4 就能明显检出。

对应四张量的一种明确排列如下；`ab` 和 `ba` 指两次独立 CTMRG，`reshape` 后的前两轴是 chi，后两轴是切口 ket/bra：

```python
A = ab_T1F.reshape(chi, chi, bond_D, bond_D).permute(2, 3, 1, 0)
D = ab_T2A.reshape(chi, chi, bond_D, bond_D).permute(2, 3, 0, 1)
B = ba_T1F.reshape(chi, chi, bond_D, bond_D).permute(3, 2, 0, 1)
C = ba_T2A.reshape(chi, chi, bond_D, bond_D).permute(3, 2, 1, 0)
```

右侧交换两条切口腿，把原生 ket/bra Gram 的普通逐腿收缩写成矩阵乘法的 trace；不是额外复共轭。右侧 chi 轴与左侧采用相反方向，以匹配图中长短蓝柱的互补排列。

源码依据：[core_C3.py](../src_code/scripts/core_C3.py:2150) 的返回顺序；[core_C3.py](../src_code/scripts/core_C3.py:2058) 的 `T1D` 留下 alpha、`T2C` 留下 beta；[correlation_length.py](../src_code/scripts/correlation_length.py:332) 明确额外做一次更新，故其 edge 提取路径应针对本 cut 改写，而非复用其最终四 T。

## 3. 四张量闭合网络的恒等式成立

把左边周期 MPO 称作 `left`，右边已交换切口腿后的周期 MPO 称作 `right`。均有 `2L` 个 D 维局部位置。只要它们表示对应半空间的 Gram，物理约化密度矩阵非零谱由 `sqrt(left) @ right @ sqrt(left)` 给出。因此

\[
S_2(2L)=-\ln\frac{\operatorname{Tr}[(\mathrm{left}\,\mathrm{right})^2]}
{\operatorname{Tr}(\mathrm{left}\,\mathrm{right})^2}.
\]

新背景的 `T1` 和 `ABAB -> DCDC` 的 `T2` 给出：

\[
\operatorname{Tr}(\mathrm{left}\,\mathrm{right})=\operatorname{Tr}(T_1^L),\qquad
\operatorname{Tr}[(\mathrm{left}\,\mathrm{right})^2]=\operatorname{Tr}(T_2^L).
\]

所以其分母、分子和 `-log(Z2)+2log(Z1)` 是正确的 replica 收缩。`T1` 的状态空间是 `chi^2`，`T2` 是 `chi^4`；完全不需要存下 `D^(2L) × D^(2L)` 的密度矩阵。

独立用任意复数四张量、`bond_D=chi=2`，分别显式构造长度 2/4/6 的 MPO 矩阵与 `T1,T2` 全矩阵，比较两种 trace：最大相对差 `2.6e-15`。这证明指标收缩相等，不把任意四个随机张量当作物理正定 Gram。

## 4. 与原始 PEPS、同一 cut 的数值核对

[code_boundary_check.py](algorithm_checks/code_boundary_check.py) 独立构造原始 honeycomb 的周期圆柱通道：固定 `2L` 行、每列沿蓝色 alpha 键配对，横向依次收缩 6 列以恢复完整六色周期。从原始单层张量直接求左右 CP 通道固定点，再算精确边界 `S2`。没有调用旧代码中方向不同的圆柱替代本图。

测试 family 都只施加 two-C3 的循环关系，不施加 `a=b` 或镜面对称。两种 family 均固定 NumPy seed `20261009`：一组正且腿不对称的张量，一组有正负号的普通实随机张量。

| 输入与 cut | chi | 周期 CTM 四张量 S2 | 同 cut 原始 PEPS 圆柱 S2 | 差 |
|---|---:|---:|---:|---:|
| 正数 generic，2L=4 | 2 | 0.0013800840 | 0.0013775649 | 2.52e-6 |
| 正数 generic，2L=4 | 4 | 0.00137756493 | 0.00137756486 | 7.22e-11 |
| 正数 generic，2L=4 | 8 | 0.00137756493 | 0.00137756486 | 7.04e-11 |
| 含符号 generic，2L=4 | 2 | 2.652181694 | 2.554351352 | 9.78e-2 |
| 含符号 generic，2L=4 | 4 | 2.560506312 | 2.554351352 | 6.15e-3 |

正数 family 的 chi=4/8，左右周期 MPO 在原始圆柱通道下的本征残差约为 `5e-9 / 2.5e-8`，支持上述 env1、左右反向的排列。错误取 env2，在同一正数例子的 `2L=4,chi=8` 给出约 `0.00092278`，左右残差约为 `0.20 / 0.33`；这个差异不是 replica trace 公式的错误，而是把不同取向的 edge 塞进了本 cut。

另一个明确反例是省掉右侧 chi 方向反转：`2L=2` 的短环因 trace 恒等式几乎检不出问题，`2L=4,chi=4` 则给出 `S2≈0.001385212`，误差约 `7.65e-6`，远大于正确排列的 `7.22e-11`。所以验证至少应包括 4 个 cut legs，不能只测最短 2-leg 环。

含符号 family 的 chi=8 跑至 600 步上限仍未满足设定的 CTM 谱阈值，结果不能用于宣称收敛。完整状态、残差和数值在 [正数结果](algorithm_checks/code_boundary_check.json)、[含符号结果](algorithm_checks/code_boundary_check_signed.json)。原始圆柱固定点的残差小于 `7e-15`。`D=1` 的原始积态通道另核对得到 `S2=0`。

这支持新路线的正确网络与 code mapping，也直接说明 **CTMRG 迭代收敛不等于边界熵已对 chi 收敛**。含符号例子 chi=4 的 CTM 谱已停止，但仍与真实圆柱差 `0.00615`。本检查没有使用实际 J2=0.24—0.27 的高 D 优化张量，不能替这些点宣布熵精度。

## 5. 主谱省掉长度指数存储，但固定保留几个谱不是精确算法

对给定有限矩阵，即便有 Jordan 块也有 `Tr(T^L)=sum_j lambda_j^L`，重数须取代数重数。全谱代入是四张量闭合网络的精确结果；只取主谱是另外的谱截断。

必要检查包括：覆盖所有外周近简并模及其重数；包含两个 replica 交换扇区；复杂特征值的相位按整数幂保留；检查最短使用长度对增加保留谱是否稳定。把保留数 8 增到 16/32 是有用诊断，不能单凭稳定就当作严格尾谱误差界。

新背景建议起始 block=4。一个起始 block 对精确简并子空间最多提供 4 个方向；若主导重数大于 4，只增加请求特征值数量不能保证找全重数。因此 block 大小、独立起始或显式扇区分解要与重数检查联动。

固定 chi 后，最终只剩有限个主导特征值，`log trace` 必然进入有限谱的线性加常数形式，或包含外周相位造成的周期项。这不能单独证明实际 gapless QSL 没有 `log L` 等修正。物理 scaling 所用长度区间必须对 chi 以及原始 D 的增加稳定；单纯把 L 扫到 1000 不扩大该稳定窗口。

## 6. 450 GB 的实际数量级

以下是 complex128、十进制 GB，未把 450 GB 偷换成 450 GiB。

- 一条完整 `T2` 向量：`16*chi^4` bytes。
- `m` 条完整 Krylov 向量：`16*m*chi^4` bytes。
- 未分批 work：`16*bond_D^2*chi^4` bytes。
- 首条输出 chi 腿每批 `b` 个值时，work：`16*bond_D^2*b*chi^3` bytes；上下两层可复用。
- 一次未剪枝 matvec 的复乘加数：`4*bond_D^2*(bond_D+1)*chi^5`；`bond_D=8` 时是 `2304*chi^5`，与新文件一致。

下表取原始 bond dimension=8、batch=8，以 `m+3` 条向量加两个 work 缓冲作粗略保守主数组预算；真实实现仍需留给 BLAS、重启变换、系统和 CTM 缓存空间。

| chi | 完整 basis 条数 m | basis GB | 主数组粗估 GB |
|---:|---:|---:|---:|
| 64 | 80 | 21.47 | 26.58 |
| 96 | 80 | 108.72 | 127.29 |
| 128 | 80 | 343.60 | 390.84 |
| 144 | 80 | 550.38 | 619.94，超限 |
| 160 | 32 | 335.54 | 434.11，余量过紧；应缩 batch 或压缩扇区 |

chi=128、80 条完整向量在 450 GB 下有明确可行的内存数量级，但一次完整 matvec 已约 `633 TFLOP`。该数字是算法工作量，不是墙钟时间预测；未在指定 cluster 实测，不接受新文件中“2—7 小时”作为验证结论。GPU 显存和主机 RAM 必须分别预算。

本方法的主要对象随 chi 的四次方增长，随原始 bond dimension 多项式增长；扫描大 L 不再增加该存储。这正是其对大周长有价值之处。

## 7. 可以怎样比较旧方案

旧`tee_exact_cylinder.py`用 `D^(2W)` 的稠密边界向量，属于小圆柱直接参考；旧实现要求 W 为 3 的倍数，其切口方向与本图交替 gamma/beta 的 cut 不同。原始 `W` 与早期新记号 `2L` 不能直接逐项对等比较熵值。旧源码已列入清理，方法区别保存在[历史摘要](清理范围_20261009.md)。

旧`tee_boundary_mps.py`已经用 MPS 压缩边界，其存储也不是对周长指数增长；额外误差来自边界 bond `M` 和乘积 bond `product_M`。所以“新方法优于旧稠密参考的长周长内存”成立，“新方法在相同精度下必然优于旧 MPS”没有证明。旧实现退出生产路线，其程序与探索产物不再保留，必要经验见[历史摘要](清理范围_20261009.md)。

新方法复用同一组主谱计算很多长度，很适合大周长二阶熵；旧 MPS 对每个有限周长单独优化左右固定点，可能更直接控制有限圆柱修正。最终效率比较要匹配 cut、物理边界条件、扇区和误差目标，再记录收敛时间与峰值内存。本次尚未做这个性能 benchmark。

## 本地复现命令

```powershell
python -B tmp_tee_schematic_20261009/algorithm_checks/code_boundary_check.py
python -B tmp_tee_schematic_20261009/algorithm_checks/code_boundary_check.py --family signed --max-steps 600
```

这些命令只使用小尺寸随机 two-C3 张量做校验，不执行高 D 熵作业。
