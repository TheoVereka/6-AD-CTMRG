# 0730 2C3 数据合并工具

把所有来源文件夹完整放进：

`D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713core\0730newdata`

建议先只扫描：

```powershell
python .\integrate_0730_twoc3.py --dry-run
```

确认识别出的 J2、D、唯一数据、未完成数据和冲突无误后，正常运行：

```powershell
python .\integrate_0730_twoc3.py
```

脚本只处理 2C3 ansatz。唯一的已完成 `(J2,D)` 会直接按
`summarize_0713core.py` 的格式写入 `0713summary`。`0713summary` 原先没有、
且没有任何有效 observable 的 run，其所有 `*_best.pt` 和同目录 `.log`
会被重命名后复制到 `0730newdata\_unfinished_2C3_for_rerun`；重复的未完成
run 不会去重。

每个存在已完成重复项的 J2 会依次弹出一个窗口。左边每个冲突 D 都是一组
单选按钮（因此始终恰好选择一个来源）；选项显示来源细分文件夹、run 时间、
计算时长和 energy per site。更换选择会立即重画右边的 energy、m_Neel 和
NN correlations。按 **Confirm** 后才会把该 J2 的选择写入
`0713summary`，随后关闭并打开下一个 J2。直接关闭窗口会中止后续 J2，不会
默认确认当前选择。

每次正常运行会写
`0730newdata\0730_twoc3_integration_report.json`，记录自动导入、未完成归档、
全部候选和已经确认的选择。若只想先执行自动部分、暂不打开窗口，可用：

```powershell
python .\integrate_0730_twoc3.py --prepare-only
```
