# 新数据的全 ansatz 合并工具

把各来源文件夹完整放入：

`D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713core\0730newdata`

先只扫描，不写入数据，也不打开 GUI：

```powershell
python .\integrate_newdata_all_ansatze.py --dry-run
```

确认识别出的 ansatz、J2、D、唯一结果、未完成结果和冲突无误后运行：

```powershell
python .\integrate_newdata_all_ansatze.py
```

旧入口 `integrate_0730_twoc3.py` 仍然可用，但现在也处理全部 ansatz。

## 数据选择层级

每个产物目录里的同一 `(ansatz,J2,D)` 是一个物理 run。该 run 的多个 chi
仍使用 `summarize_0713core.py` 的既有选择规则归并成一个候选；不同产物目录
才是不同 run。

- 某 `(ansatz,J2,D)` 只有一个完成候选且 `0713summary` 不存在：自动导入。
- `0713summary` 不存在且 run 没有有效非-lookahead observable：把所有
  `*_best.pt` 和同目录 `.log` 重命名后复制到
  `_unfinished_all_ansatze_for_rerun`。重复的未完成 run 全部保留。
- 完成候选重复：严格按 `(ansatz,J2)` 逐张弹出 GUI；不会把多个 ansatz
  放进同一张 figure。

每张 GUI 只有当前 ansatz 的一个 plot column，只为当前 `(ansatz,J2)` 有冲突
的 D 显示单选按钮。每个 D 始终恰好选择一个来源；更改后立即重画 energy、
`m_Neel` 和 NN correlations。按 **Confirm** 后先完整 staging 当前组的所有
选择，再整体替换并打开下一个 `(ansatz,J2)`；任一 D staging 失败都不会部分
替换。直接关闭窗口会停止后续确认。

选项文字包含来源细分文件夹、run 时间、计算时长和 energy per site。

正常运行会写
`newdata_all_ansatze_integration_report.json`，记录自动导入、未完成归档、
全部冲突候选和已确认选择。只执行自动部分而暂不打开 GUI：

```powershell
python .\integrate_newdata_all_ansatze.py --prepare-only
```

`0801core` 两个 main 当前出现的全部 yaml ansatz 名均已纳入后备识别；未来新增
ansatz 只要 `hyperparams.yaml` 含有 `ansatz` 和 `J2`，无需修改该工具。
