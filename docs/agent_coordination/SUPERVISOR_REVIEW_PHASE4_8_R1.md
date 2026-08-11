# Supervisor Review — Phase 4.8 R1（Real-Path Admissibility）

更新日期：2026-08-04
审查人：Supervisor（Project Director）
判定：**R1 = CONDITIONAL PASS / NO VETO**

## 1. 独立复核

1. Phase 4.7 结论未改判（R0/H5/H6 PASS、H7 FAIL、H2 FAIL、Phase 5 CLOSED；候选 profile 冻结）✓；
2. 环境实测：1× RTX 2080 Ti（驱动 580.76.05、CUDA 13）、torch 2.8.0+cu128、NCCL 2.27.3、CUPTI 存在、无 nsys/ncu、无多 GPU、无多节点 ✓；
3. 等级判定：当前仓库 = L0（NumPy proxy）；硬件可达 = **L1**（高保真单机）；L2/L3 需新硬件 ✓；
4. R1 判据核对：
   - ≥L1：PASS（L1 可达，需构建高保真层）；
   - 候选 profile 映射到真实/高保真事件：PASS（EXECUTION_PATH_MAP）；
   - baseline 与候选同路径公平比较：PASS（唯一区别 = reveal 参数）；
   - reveal/control/sync 成本可测：PASS（measurement capability table；M 证据等级）；
   - scheduler/execution/comm 时间可区分：PASS（critical-path 分解）；
   - 不依赖未来真值：PASS（partial 语义）；
   - 稳定环境：PASS（含服务器不稳定性风险登记）；
   - 允许进入实现：条件性（见下）。
5. 未执行禁止项：未改 production、未实现 checkpoint 8、未加 profiler、未运行 microbenchmark/pilot、未生成正式 corpus、未创建额外 Subagent、未启动 scheduler 优化 ✓。

## 2. 判定

**R1 = CONDITIONAL PASS / NO VETO**：当前可达最高等级为 **L1（高保真单机）**。进入 Phase 4.8-1 的条件：

1. 必须先构建高保真执行层（router shim、dispatch/GEMM kernel、reveal/sync 计时）后测量；
2. 所有结论限定为高保真/单机验证，**不得声称 L2/L3**；
3. reveal/control/sync 成本必须实测（M），不得置 0；
4. I1（插桩等价）通过前不得进入 microbenchmark；
5. 用户批准后才进入 Phase 4.8-1。

## 3. 结论

R1 = **CONDITIONAL PASS**。允许将 D1 draft protocol 提交用户审核；在用户批准并完成 L1 高保真层构建前，不得实现 profile 或运行任何实验。
