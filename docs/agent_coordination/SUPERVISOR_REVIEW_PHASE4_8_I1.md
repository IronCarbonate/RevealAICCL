# Supervisor Review — Phase 4.8 I1（Instrumentation Equivalence）

更新日期：2026-08-04
审查人：Supervisor（Project Director）
判定：**I1 = PASS / NO VETO**

## 1. 独立复核

1. OFF vs ON 等价性：180 jobs 中 completion/legality/action_digest/action count 全 0 差异 ✓；
2. D0 与冻结 H5 A1 交叉验证 60/60 一致 ✓（运行器忠实性）；
3. 插桩默认关闭；flag 控制；开销已量化（ON−OFF 含合成 GPU kernel，已标注）✓；
4. legality 100%；无 timeout；未改 production 代码；未运行 microbenchmark/pilot ✓；
5. 合成 dispatch/GEMM/collective 明确标注为 synthetic，不冒充生产 MoE ✓；
6. D1（partial_shards 75%/ckpt8）与 D2（fullinfo 上界）已在执行层实现但未做正式实验 ✓。

## 2. 判定

**I1 = PASS / NO VETO**。允许进入 Phase 4.8-2（microbenchmark 与成本校准），前置：

1. 用户批准；
2. 成本证据等级（M/E/D/S/O）标注继续执行；
3. 合成执行负载与生产语义的差异在结论中持续声明。
