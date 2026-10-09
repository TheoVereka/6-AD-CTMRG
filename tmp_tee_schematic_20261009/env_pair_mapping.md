# 三组 CTM environment 的左右配对与指标审计

本次完整阅读新的 [我们的数值背景.txt](我们的数值背景.txt)，并逐一核对当前 `core_C3.py` 的三个 environment、C₃ edge 轨道、输出轴与消费者，以及 `correlation_length.py` 的提取路径。

**结论：在本图交替 γ/β 的同方向 armchair cut 上，正确配对是 normal env1 ↔ swap env1、normal env2 ↔ swap env3、normal env3 ↔ swap env2。不是机械的同号配对。** 三组分别对应原六色网格上不同横向位置的切口；它们不是三个 C₃ 旋转方向。

以下 `normal` 指原始 `(a,b)`，`swap` 指原始 `(b,a)` 独立运行 CTMRG。审计没有修改生产代码。

## 1. 三个 environment 中实际保存了哪些 edge

[core_C3.py:2150](../src_code/scripts/core_C3.py:2150) 的返回顺序是：

```text
env1 = C21CD, T1F, T2A
env2 = C21EB, T1D, T2C
env3 = C21AF, T1B, T2E
```

源码明确只存每条 C₃ 轨道的一个代表。完整轨道及固定到本图 γ/β 方向后的颜色如下：

| normal env | 第一个 edge 的 C₃ 轨道 | 取 γ 方向的代表 | 第二个 edge 的 C₃ 轨道 | 取 β 方向的代表 |
|---|---|---|---|---|
| 1 | T1F,T2B,T3D | d(γ) | T2A,T3C,T1E | a(β) |
| 2 | T1D,T2F,T3B | b(γ) | T2C,T3E,T1A | c(β) |
| 3 | T1B,T2D,T3F | f(γ) | T2E,T3A,T1C | e(β) |

依据：[初始化轨道说明](../src_code/scripts/core_C3.py:1918)、[env1→env2](../src_code/scripts/core_C3.py:2041)、[env2→env3](../src_code/scripts/core_C3.py:2070)、[env3→env1](../src_code/scripts/core_C3.py:2095)。例如 env2 更新明确说明 `T3B` 属于 `T1D` 轨道；第三个更新明确说明 `T3F` 属于 `T1B` 轨道。

这一步只使用合法的 C₃ 空间旋转。没有假设同一 a/b 张量本身对腿置换不变，也没有添加反射或物理自旋旋转。

## 2. 为什么 swap 后必须交叉配对 env2/env3

由 [two-C₃ 构造](../src_code/scripts/core_C3.py:1294)：

```text
原始 six-site： a,b,c,d,e,f
交换 a,b 后：  b,a,d,c,f,e
```

因此 swap environment 在同一 γ/β 方向上提供：

| swap env | 第一位置，γ | 第二位置，β |
|---|---|---|
| 1 | c | b |
| 2 | a | d |
| 3 | e | f |

原始蜂窝连接为 `γ: CD,BE,AF`、`β: AB,CF,DE`。逐个 normal env 寻找切口对岸：

```text
normal env1： d(γ) -- c(γ)，a(β) -- b(β) → swap env1
normal env2： b(γ) -- e(γ)，c(β) -- f(β) → swap env3
normal env3： f(γ) -- a(γ)，e(β) -- d(β) → swap env2
```

在这组给定的方向和两行顺序下，每一行对岸颜色都唯一，所以 environment 配对也唯一。只按 a/b orbit 而忽略六色位置与切腿类型，无法得到这个结论。

## 3. 这三组在原始网格上的具体位置

采用用户原图从0开始编号的列：偶数行 `cfabed...`，奇数行 `bedcfa...`。

| pair | 两行左颜色 → 右颜色 | 切口列 | 向右的原始六列通道 |
|---|---|---|---|
| normal1 / swap1 | d→c，a→b | 5\|6 | 6,7,8,9,10,11 |
| normal2 / swap3 | b→e，c→f | 3\|4 | 4,5,6,7,8,9 |
| normal3 / swap2 | f→a，e→d | 1\|2 | 2,3,4,5,6,7 |

这些 cut 相差横向平移两个 zigzag site，始终切 γ/β 两种键；几何方向、两行周期与 `S₂(2L)` 的长度定义相同。

原始 two-C₃ ansatz 不自动施加这些横向平移对称，所以三组熵不必对任意输入张量相同。不能把它们预先平均、强制相等，或把一个 pair 的谱当作另两个 pair 的谱。

## 4. LMN 与 XYZ 到底接哪条蓝柱

原 edge 的 canonical 存储是 `(LMN,XYZ,ketbra)`：第一条 χ 腿属于 L/M/N 这一类，第二条属于 X/Y/Z 这一类。

在本图的边界链上：

```text
LMN = 长柱，几何长度2；XYZ = 短柱，几何长度1。
左边：γ位置 → β位置，间距2，用第一χ轴 ↔ 第一χ轴；
      β位置 → 下一γ位置，间距1，用第二χ轴 ↔ 第二χ轴。
右边：γ位置 → β位置，间距1，用第二χ轴 ↔ 第二χ轴；
      β位置 → 下一γ位置，间距2，用第一χ轴 ↔ 第一χ轴。
```

这由蜂窝 y 坐标直接确定，并对三条奇数列 cut 分别成立，不能根据相同 shape 把两类 χ 腿当作可互换。

例如原完整六 edge 能量收缩把 `T3D` 写成 `LXct`，把 `T2A` 写成 `LXbs`，见 [core.py:3235](../src_code/scripts/core.py:3235)、[core.py:3237](../src_code/scripts/core.py:3237)：同第一轴 L 是长柱，同第二轴 X 是短柱。

三个 C₃ 更新都会把 edge 原封不动地交给下一阶段的 canonical `MYa/LXb` 输入。更新表达式中 `T1D` 暂写 `YMa`、`T2C` 暂写 `XLb`，这些是更新前后的 frame 重命名，**不是待补做的轴转置**。这一点也由 [correlation_length.py:461](../src_code/scripts/correlation_length.py:461) 的注释与 [core_C3.py:2074](../src_code/scripts/core_C3.py:2074) 的真实消费者核实。

## 5. 三组 A/B/C/D 的明确排列

下面 `r(edge)=edge.reshape(chi,chi,bond_D,bond_D)`，后两轴按 ket、bra 顺序展开。每一组均已分别依据 cut 颜色、左右长短柱和源码 canonical 轴推导：

```python
# pair 1: normal env1 / swap env1
A = r(normal_T1F).permute(2,3,1,0)
D = r(normal_T2A).permute(2,3,0,1)
B = r(swap_T1F).permute(3,2,0,1)
C = r(swap_T2A).permute(3,2,1,0)

# pair 2: normal env2 / swap env3
A = r(normal_T1D).permute(2,3,1,0)
D = r(normal_T2C).permute(2,3,0,1)
B = r(swap_T1B).permute(3,2,0,1)
C = r(swap_T2E).permute(3,2,1,0)

# pair 3: normal env3 / swap env2
A = r(normal_T1B).permute(2,3,1,0)
D = r(normal_T2E).permute(2,3,0,1)
B = r(swap_T1D).permute(3,2,0,1)
C = r(swap_T2C).permute(3,2,1,0)
```

三组排列模式相同的原因是几何方向、左右柱长和 canonical 存储都相同，不是从第一个 pair 盲目复制。

代入新背景 `A[a,b,i,p],B[b,c,j,q],D[a,b,p,i],C[b,c,q,j]` 时：

```text
i = 左短 XYZ；p = 左长 LMN；j = 右长 LMN；q = 右短 XYZ。
ABAB 输入(i,j,k,l) = (左短,右长,左短,右长)。
ABAB 输出(p,q,r,s) = (左长,右短,左长,右短)。
DCDC 把这组长短腿重新映回输入顺序。
```

右侧红绿切口腿交换已包含在 `permute(3,2,...)` 中；双层原本已包含 bra 的共轭，不能再加一次共轭。

## 6. 三个 pair 分别与同 cut 的原始 PEPS 小圆柱核对

本次只读核验复用 [已有小圆柱 helper](algorithm_checks/code_boundary_check.py)，另在内存中分别构造上表起始列为6、4、2的原始六列 CP 通道。没有用第一个 cut 的参考值验证另两个 cut。

D=2、χ=8、`2L=4`，正且腿不对称的随机 two-C₃ 张量，NumPy/Torch seed=20261009；两次 CTMRG 均3步满足谱停止条件：

| normal/swap pair | 原始同 cut 圆柱 S₂ | 正确 pair S₂ | 差 | 错误同号 pair S₂ |
|---|---:|---:|---:|---:|
| 1/1 | 0.001377564856 | 0.001377564924 | 6.80e-11 | 同正确 pair |
| 2/3 | 0.000859922649 | 0.000859338034 | −5.85e-7 | 2/2：0.000922787735 |
| 3/2 | 0.000068945046 | 0.000068959150 | 1.41e-8 | 3/3：0.000926919655 |

左右周期 MPO 在各自圆柱通道下的残差分别约为 `5.0e-9/9.9e-9`、`1.28e-5/4.7e-8`、`8.5e-8/5.1e-6`。χ=4 也逐 pair 核对，正确配对得到同数量级差异。它们支持各自的 indexing；残余有限 χ/周期 CTM 近似误差没有被当作恒等精确。

## 7. 生产实现前已经解决与仍需保留的边界

三组颜色、切腿类型、environment 配对、LMN/XYZ 柱长和 A/B/C/D 轴序已得到独立几何推导及三条同 cut 小圆柱核对，未发现必须向用户追问的配对歧义。

生产脚本应从一次 normal 与一次 swap CTMRG 的完整9对象返回值中分别提取三组 pair，记录各 pair 的原 environment 名称与切口列相位。`correlation_length.obtain_4Ts()` 只取更新后的 env2，不提供本任务所需完整三组；其自身上下 row-transfer 配对不应覆盖这里的周期边界配对。

谱 solver、主谱截断、各熵 observable 的误差控制仍是之后的数值工作。本次只完成 mapping 审计，没有生成生产求谱脚本，也没有把有限 χ 小例子称为实际 0713 高 D 的精度证明。
