# 固定给定 edge 后，主谱求解的精度核查

## 可以继续实现，没有新的代数定义矛盾

这里固定用户给定的四个 edge、D=8、χ=64，讨论由它们定义的两个转移算子。新背景的 `T₁` 显式矩阵、`T₂` 八步 matrix-free matvec、两个 replica 交换扇区及归一化公式彼此相容。本核查不将其他 χ 或 D 的物理问题加到这次求谱任务里。

**实际需修正的是精度承诺：特征值稳定到 `1e−8` 加上保留数 `8→16→32` 稳定，是有用的数值验收，不能单独构成绝对 `|ΔS₂|≤1e−4` 的证明。**

## 真正的 block solver 已落地

[`spectral_prototype.py`](spectral_prototype.py) 提供：

```python
solve_block(operator_matvec, n, k, block_size=4, subspace=80,
            tol=1e-10, max_matvec=2000, device="cpu", seed=20261009,
            projector=None)
```

它使用真实的多向量 block Krylov 扩张、两次块正交化、thick restart、Rayleigh–Ritz，以及保留实不变子空间的有序实 Schur 重启。Q 与 AQ 是预分配 float64 数组；重启按行分块右乘小矩阵，不生成第二份完整 basis。随机初始块及补秩块都应用可选 projector。复共轭对和已发现的边界等模簇不强行切成恰好 k 根。

返回 `BlockResult`，含 `eigenvalues`、逐根 `relative_residuals`、`converged`、matvec 次数、重启次数、退出原因与正交误差。收敛的含义仅为返回的 Ritz 对通过真实空间残差检验；没有暗示遗漏谱已经受控。

这里没有将 SciPy `eigs(ncv=80)` 冒充 block Arnoldi。官方 API 的 `v0` 是一个起始向量，其算法为 implicitly restarted Arnoldi；`ncv` 是子空间规模。[SciPy eigs 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html)

## 测试结果

[`check_spectral_prototype.py`](check_spectral_prototype.py) 与 [`spectral_prototype_results.json`](spectral_prototype_results.json) 覆盖：

- 96维实正规矩阵，实际执行9次重启后收敛。
- 5重精确主简并，block_size=8，保持完整5根。
- 实矩阵的复共轭谱，k落在共轭对内部时自动保留整对。
- 非正规实矩阵，实际执行10次重启后收敛。
- 正交投影、随机补秩，以及 matvec 预算耗尽时明确返回失败。
- 缩放后的 `Tr(T^L)` 在 L=50、500、1000 时避免直接幂次溢出。

这些小矩阵上，返回根相对完整 dense spectrum 的绝对差小于 `1.5e−14`，真实空间相对残差小于 `1e−10`。未在本地执行 D=8、χ=64 的高维实测。

χ=64 时，一套80条 float64完整向量为10 GiB，Q+AQ为20 GiB。块 QR 与逐根复Ritz残差采用少量实向量临时空间；重启 tile 约不超过128 MiB。实际GPU峰值还包括调用者的 matvec 工作区、框架缓存及仍被调用者引用的张量，不能只用 basis 计总显存。

## 可验证的误差预算

令 `Zᵣ=Tr(Tᵣᴸ)`，保留谱给出 `Z̃ᵣ`。若已知

\[
|Z_r-\widetilde Z_r|\le\epsilon_r|\widetilde Z_r|,
\qquad 0\le\epsilon_r<1,
\]

则一个保守的熵误差界是

\[
|\Delta S_2|
\le -\ln(1-\epsilon_2)-2\ln(1-\epsilon_1).
\]

每个迹的相对误差预算取 `3e−5` 时，右端约为 `9.00014e−5`；再为幂次求和舍入保留约 `1e−5`。这是一个可以检查的分配。

若主要谱项同相、没有明显抵消，主本征值真实相对误差 `1e−8` 在 L=1000 时造成约 `1e−5` 的单迹相对误差。因此背景的数值目标本身有合理余量。但前提是“真实本征值误差”，而非仅把 solver `tol` 设成 `1e−8`。

### 1. 非正规误差

对非正规矩阵，小右残差不等于小本征值误差。测试文件给出了明确反例：

\[
T=\begin{pmatrix}1&10^6\\0&1-10^{-4}\end{pmatrix},
\quad\widetilde\lambda=1+10^{-3},
\quad v=(1,10^{-9})^T.
\]

它的相对右残差约 `5.5e−13`，但本征值误差仍为 `1e−3`。所以不能把 residual=`1e−10` 当成 forward error=`1e−10`。

实践中可以用更紧 tol、更大子空间及独立起始块比较 **S₂结果本身**；必要时计算相匹配的左右Ritz向量重叠，诊断本征值条件数 `||l||·||r||/|l†r|`。SciPy 明确区分左右本征向量的方程，不能以右向量代替左向量。[SciPy eig 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html)

### 2. 遗漏谱尾

若确实知道遗漏的所有本征值模长都不超过 β，则

\[
|Z_r-Z_{r,k}|\le(\dim T_r-k)\,\beta^L.
\]

这一界不要求正规性，但“β是所有遗漏根的上界”必须另有依据。一个额外计算到的 Ritz 根不自动给出这个保证。

以 χ=64、T₂维数 `64⁴`、保留32根、单迹尾项预算 `2e−5`、缩放后的保留迹约1为例，维数最坏界要求遗漏谱半径比例不大于：

| L（取幂次数） | β/主谱半径 |
|---:|---:|
| 50 | 0.57747 |
| 100 | 0.75991 |
| 500 | 0.94657 |
| 1000 | 0.97292 |

所以最短周长通常最难通过谱截断验收。`8→16→32` 的 S₂ 稳定性应输出为“谱数量稳定性已通过”，不应写成“严格绝对误差已证明”。完全覆盖的精确简并重数也需要核查；block_size=4不是未知重数的数学上界。

### 3. 迹求和的符号与相位

先除以各自谱半径，再计算复数幂次和，最后加回 `L log(radius)`。保留复共轭根的相位，记录 `sum(abs(terms))/abs(sum(terms))` 的抵消放大因子。正性或虚部检查失败时明确报错；不要静默取绝对值、只取模长或将负purity裁剪到0。

## 生产入口的建议状态

生产脚本可以持续实现并输出实际 S₂；建议分别记录：

1. `ritz_residuals_converged`：逐根真实空间残差是否通过。
2. `spectrum_count_stability`：保留谱数量增加后，整个请求周长区间的 S₂ 是否在容差内稳定。
3. `solver_parameter_stability`：更紧tol/不同block或独立起始块的 S₂ 是否稳定。
4. `absolute_error_certified`：只有给出了足够谱尾及本征值误差界才设为true；本次原型默认不认证。

没有必要因“不具备严格误差证书”停止构造和测试。也不能把尚未验证的 `1e−4` 精度写成已经保证。
