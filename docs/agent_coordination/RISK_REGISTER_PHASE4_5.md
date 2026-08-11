# Risk Register — Phase 4.5

更新日期：2026-08-04

| # | 风险 | 等级 | 状态 | 缓解/处置 |
|---|---|---|---|---|
| R1 | 服务器连接不稳定（region-42 今日多次 SSH 超时，最近一次持续 ≥5 分钟不可达） | 高 | **已发生** | 已确认数据盘在 `/root/autodl-tmp`（关机保留 15 天）；恢复后立即补跑 read-back/测试并上传文档；如实例被释放则用本地 payload 重建 |
| R2 | 正式 artifacts 被意外覆盖/污染 | 高 | 已控制 | `FORMAL_RESULT_FREEZE.md` 明确只读；新分析强制使用 `outputs/phase4_5/` 新目录 |
| R3 | H2 结论被局部优势误导（completion/CVaR-only） | 高 | 已控制 | 冻结文档明确禁止将局部优势描述为 H2 成功 |
| R4 | H2a profiling 污染事件 hash / RNG / 方法顺序 | 高 | 未开始（待批准） | 按指令 A2：flag 控制、默认关闭、量化 profiler 开销、equivalence 测试 |
| R5 | 多线程 vs 单线程数值差异影响分析可比性 | 中 | 已记录 | 正式运行使用 MULTI_THREADED_DEFAULT（admission-003）；H2b 以该正式 artifact 为准，不做跨配置混比 |
| R6 | 旧服务器（无卡模式）证据/归档不可达 | 中 | 已发生 | 两次失败运行的归档均在旧服务器 `/root/autodl-tmp/phase4-archive/`，旧服务器多次不可达；新正式结果已完整发布在本机，旧归档仅作补充证据 |
| R7 | 环境重建偏差（Python/pip-freeze 不一致） | 中 | 已排除 | venv 重建后 python hash `0c05a22b...` 与 pip-freeze hash `6f27b26b...` 与冻结值完全一致 |
| R8 | 测试期间产生新 staging/destination 污染 | 中 | 已控制 | 测试脚本断言 destination=false、staging=0；read-back 校验 manifest |
| R9 | 四象限决策被绕过（跳过 H2a/H2b 直接重跑） | 高 | 已控制 | 执行顺序由 Supervisor 把关，文档明确禁止 |
| R10 | Systems Performance Agent 越权（修改语义/关闭 checker/扩大范围） | 中 | 未创建 | 只在其获批后创建，职责单一（H2a），生命周期受限，完成后退出 |
| R13 | Systems Performance Agent 工具失效（连续 4 次未执行任务，仅待命/询问方向） | 中 | **已发生** | 已中断并释放槽位；由主 Agent 在相同硬约束下接管完成 H2a；后续如再创建子代理需验证其任务接收可靠性 |
| R11 | H2b 分桶样本不足导致伪结论 | 高 | 未开始 | 指令 B3：≥5 条独立 sequence、禁止行级当独立样本、禁止只挑正向桶 |
| R12 | 新实验覆盖正式结论的可信度 | 中 | 已控制 | 任何新 H2 重评需重新预注册协议与新输出目录 |
