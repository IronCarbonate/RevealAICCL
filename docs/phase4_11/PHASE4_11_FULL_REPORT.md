# 完整研究报告：流量感知自适应集合通信控制（AICCL）

**覆盖范围：Phase 3B → Phase 4.10（含 Phase 4.11 汇总）**

更新日期：2026-08-10（Phase R0 证据口径修正）
语言：通俗中文（关键数字与正式记录一致）
状态：只读汇总报告；未运行新实验、未修改任何生产组件

> **R0 修正**：P10-1D 只测得逐 chunk CUDA router timing，随后将
> readiness 量化/replay 给 scheduler；它不是 concurrent pipeline。
> 419.84µs 仅是 replay/quantized candidate actionable window。
> fast-path 数值为 step-only 1,043.1µs、含 bind/checker 1,139.5µs、
> 含 digest 2,047.2µs，均不是严格理论下界。历史 replay-based P10-1
> formal 保持 CLOSED，但不禁止建立新的 concurrent/event-driven architecture。

---

## 摘要（一句话版）

这套研究想回答一个问题：**能不能通过"提前看到一点流量信息、更早决定通信动作"来让 GPU 集群的集合通信更快？**

答案分三层：

1. **调度算法本身没有剩余价值**——预测未来、多场景稳健规划、换排序、加风险开关、自适应调整，全部无效或得不偿失；
2. **"信息更早到手"确实有用**——固定方案（75% 分片提前揭示 + 第 8 时隙全量揭示 + 简单调度器）在单卡和双卡真机上都有稳定的端到端收益；
3. **旧 replay-based 参考路由器路径未获准入**——调度器单步决策约 ~12ms，而 replay/quantized candidate window 为 ~0.42ms；当前实现的 fast-path estimates 为 ~1.04/~1.14/~2.05ms。该历史 formal 路径已关闭，但真实并发架构尚未测量，不能由此排除。

---

## 1. 背景：我们在解决什么问题

现代 GPU 集群做集合通信（allreduce、allgather 等）时，消息要在多条链路上流动，而链路带宽、拓扑和共享带宽组是固定的，流量是动态的。传统调度器"看到什么做什么"。

我们研究的是：如果通信层能提前观察到一部分流量信息（比如哪些 token 已经到达、要发给谁），调度器能不能**更早开始有用的传输**，从而整体更快。

为了严格验证，整个项目采用"预注册 + 门控"流程：先冻结协议和统计口径，再运行实验，结果无论好坏都如实记录。整个研究跨 Phase 3B 到 Phase 4.10，共 20 个门（Gate）与实验，使用 5 套互不重叠的数据集（corpus）和真实计时。

---

## 2. 研究路线总览

| 阶段 | 研究问题 | 判定 |
|---|---|---|
| Phase 3B | 无预测的流量歧义集是否可审计可复现 | PASS |
| H1 | 历史流量能否被预测 | **FAIL** |
| H2 / H2a / H2b | 多场景稳健规划是否有价值 | **FAIL**（可实现性条件通过，算法价值失败） |
| 象限 2（W1/W2/W3） | 静态预计算/排序/风险门控是否有价值 | 无价值（W1 等价性通过） |
| Route A | 揭示时机与粒度是否主导完成时间 | PASS |
| H5 / H6 / H7 | 可实现揭示成本；固定预算选择器；自适应 | PASS / PASS / **FAIL** |
| L1 部署 | 单机高保真真实时间 | **PASS** |
| L2-S 部署 | 双卡真实 NCCL | **PASS** |
| Phase 4.9-F | L2 收尾与 read-back | PASS |
| P10-R0 / S0 / I1 | 生产路径审计；子strate 选型；参考 router 正确性 | CONDITIONAL PASS / PASS / PASS（17/17） |
| P10-P0 / T0 | pilot 机制；三臂计时稳定化 | CONDITIONAL PASS / PASS |
| P10-F0-v1 | 正式准入证明 | **FAIL**（P4 决定性失败） |
| P10-SF0-A / SF0-B | 调度器快速路径审计 / 优化门 | PASS / **FAIL** |
| Phase 4.10-F | 收尾；P10-1 formal | PASS；formal = **CLOSED** |

---

## 3. 分阶段详细结果

### 3.1 Phase 3B：无预测的流量歧义集（PASS）

**问题**：不靠预测，只从历史流量构造"可能场景集合"，是否可靠？

**结果**：`boundary_scenarios` 方法、K=8 场景、校准半径 0.3433。预注册条件 1–6 全部通过（联合覆盖率 0.94、配对改善 CI>0、LOFO 无系统性恶化、尾部分位数充足、宽度合理、完整 read-back 通过）。十份正式产物、25 万行原始数据，read-back 零差异。

**结论**：无预测的歧义集可作为后续研究基座，但这不等于任何调度收益。

### 3.2 H1：历史预测（FAIL）

**问题**：用最近历史训练 MLP 预测下一时刻流量，能否比"直接沿用上一刻"更好？

**结果**：不能。测试集上 MLP 总误差 1.6468，比 previous-value 的 1.5678 更差；配对差 −0.0790（95% CI [−0.1133, −0.0478]，全为负）；5 个流量族只有 1 个略好；留一族（LOFO）验证 0/5 正向。

**结论**：历史预测路线失败，冻结为负结果。

### 3.3 H2 / H2a / H2b：多场景稳健规划（FAIL）

**问题**：同时考虑多种未来场景（K=8）选"最稳妥"动作，是否更好？

**H2 结果**：完成时间比"等待"略好（20.49 vs 25.88 时隙），与"看到什么做什么"（partial）相当（20.61）；但端到端耗时约 1042ms，是被动基线（~104–116ms）的 9–10 倍。条件 1/3/6 失败，H2 = FAIL。

**H2a（可实现性）**：1042ms 中 92.3% 已被冻结计时组件解释——歧义集构建 479.7ms（46.0%）+ 前缀综合 450.9ms（43.3%）。理论上存在 7.1× 加速路径，条件性 PASS。

**H2b（算法价值）**：即使只算调度（不算实现开销），稳健规划只比 partial 好 **+0.11 时隙**（CI [0.087, 0.140]），且动作集合 98.5% 相同、首动作 71% 一致、完整序列一致仅 11.7%、丢弃动作 0。不存在收益足够大的工作区间。H2b = FAIL。

**结论**：多场景思考基本没有带来新决策，纯属拖慢系统。

### 3.4 象限 2：静态预计算 / 排序 / 风险门控（无价值）

- **W1**：全对最短路径预计算与冻结语义逐位一致，单次查询加速 302×；但完成时间不变（语义不变，只是查询更快）。等价性 PASS。
- **W2**：按剩余距离/容量余量换候选顺序——结果与基线完全相同；lookahead 只改善 +0.030 时隙且置信区间跨 0，不显著，还多花 65% 计算。
- **W3**：风险检测门控在留出集上 100% 选择"立刻行动"，等待从不占优，wasted actions 为 0。门控空转。

**结论**：调度决策侧（排序、选择、自适应、多场景）在冻结语义下没有剩余价值。

### 3.5 Route A：揭示时机是主导瓶颈（PASS）

**问题**：信息"什么时候全量揭示、分几步揭示"对完成时间影响多大？

**结果**（新 corpus 1042/1142/1242，300 坐标/档）：

| 档位 | 全量揭示时隙 | 完成时间（partial） |
|---|---:|---:|
| S4 | 32 | 36.18 |
| S0（现状） | 16 | 20.95 |
| S5 | 8（粗粒度） | 16.30 |
| S1 | 8（细粒度） | 14.92 |
| S2 | 4 | 13.40 |
| S3 | 1 | **11.80** |
| fullinfo 上界 | — | 10.80 |

配对改善（越早越好）：S1 +6.03、S2 +7.56、S3 +9.16 时隙（CI 全 >0）；推迟到 S4 则 −15.22。S3 距全信息上界仅差 1.0 时隙；细粒度（S1）比粗粒度（S5）好 1.38 时隙。

**结论**：信息揭示节奏是完成时间的主导瓶颈；信息利用（调度）本身接近最优。

### 3.6 H5 / H6 / H7：可实现揭示、预算、自适应（PASS / PASS / FAIL）

**H5（成本计入）**：把提前揭示的真实成本（计算、控制消息、同步）计入端到端后：

| 臂 | ΔJ vs 基线 | 95% CI |
|---|---:|---:|
| A2 粗粒度提前揭示 | +6.06ms | [+5.50, +6.59] |
| A3 A2+全局直方图 | +5.98ms | [+5.42, +6.51] |
| A4 rank-local 流式 | **+9.22ms** | [+8.26, +10.13] |
| A5 全局聚合 | **−0.13ms** | 全序列为负 |

A4 收益约 15%，成本只占收益 ~1%；全局聚合信息反而变差。H5 = PASS。

**H6（固定预算选择器）**：在 25%/50%/75% 预算下，partial_shards（按 token 分片揭示）稳定优于随机（+0.60/+0.81/+0.57ms，CI>0，5/5 族、3/3 种子）；entry 级选择器（先揭示哪个完整 entry）无差异。H6 = PASS。

**H7（自适应）**：规则控制器在所有特征桶都选 75%，与固定方案完全等价（Δ=0.0000ms）；理论 oracle 上界也只多 0.0014ms。H7 = FAIL——没有值得自适应的异质性，保留固定 75%。

### 3.7 L1 部署验证（PASS）

在真实时间执行层（合成 shim、单 rank、RTX 2080 Ti）上，300 jobs/臂正式测试：

| 指标 | D0 | D1 | Δ |
|---|---:|---:|---:|
| 完成时间（时隙） | 20.34 | 13.91 | +6.43（CI [+5.68, +7.12]） |
| E2E 墙钟 | 55,195µs | 44,271µs | **+10,953µs**（CI [+3,598, +23,148]） |
| 吞吐 | 18.12 | 22.59 | +24.7% |
| legality | 100% | 100% | — |

3/3 种子正向；4/5 流量族正向（hotspot_random_walk −1.1ms 为负，记为适用边界）。D1 = PASS（L1）。

### 3.8 L2-S 部署验证（PASS）

换成 2× V100 + 真实 2-rank NCCL（通信成本用真实 allreduce 62–87µs、allgather 122–136µs 实测替代假设值）：

| 指标 | D0 | D1 | Δ |
|---|---:|---:|---:|
| 完成时间 | 20.34 | 13.91 | +6.43 |
| E2E 墙钟 | 53,729µs | 47,272µs | **+6,458µs**（CI [+3,409, +9,385]） |
| 吞吐 | 18.61 | 21.15 | +13.7% |
| GPU busy（含真实 collective） | 4,375µs | 2,239µs | −49% |

D1 = PASS（L2-S）。Phase 4.9-F read-back 0 差异，L2-F0 = PASS。

### 3.9 Phase 4.10：生产路径审计（从选型到关闭）

**P10-R0（审计）**：仓库没有真实 MoE router / expert GEMM / DeepEP；NCCL（torch.distributed）真实可用；DeepEP 需要 Ampere+（V100 sm_70 不支持）；MSCCL 工具因缺 msccl 不可编译。CONDITIONAL PASS。

**P10-S0 / P10-I1（选型与正确性）**：选定"最小 PyTorch reference router"（L2-R 命名，不称生产 router）。历史 17/17 等价性检查通过；Phase R0 又完成 19/19 强化检查：actual 75% partial view、独立 token→traffic oracle、真实未揭示反事实扰动/no-leak、token loss/duplication 与确定性 tie tests 均通过。

**P10-P0 / P10-T0（pilot 与计时稳定化）**：
- pilot（20 jobs/臂）：completion +1.95 时隙；但 E2E Δ −19.7ms（被每任务 ~80–100ms 固定 setup 主导），hotspot −32.8ms；机制可准入，E2E 收益未确立；
- 三臂（B0 batched / C0 chunked / C1 chunked+75%@8）：C1 完成时间 22.9 vs C0/B0 28.1（+5.2 时隙，稳健）；稳态 E2E C1≈B0（151.4 vs 150.0ms）；router 成本 934.8µs vs 257.6µs。该实现先完成 router timing，再量化/replay readiness，不是 router 与 scheduler 的真实并发执行。

**P10-F0-v1（正式准入证明，FAIL）**：

| 性质 | 结果 |
|---|---|
| P1：窗口内有 ≥3 个可行动就绪事件 | PASS（8 个 chunk CUDA 完成） |
| P2：首次提交早于最终 router 完成 | PASS（slot 4 < slot 8） |
| P3：75% 就绪不早于调度器启动 | PASS（p75 @ slot 6 ≥ slot 4） |
| P4：replay/quantized candidate window > 调度器单步 p95 | **FAIL（419.8µs < 12,290µs）** |

**P10-SF0-A / SF0-B（快速路径审计）**：
- 复现：单步 p95 11,290–12,933µs（P10-1E 记录 12,290µs）；
- 分解：enumerate 83.5%、pack 8.9%、gate 0.1%（p95 口径合计 92.4%、均值 99.2%）；enumerate 内部 90.1% 是静态可缓存的 BFS 距离重算；
- 首提交准备 p95 = 8,674µs（含确定性 checker）；
- implementation fast-path estimates：step-only **1,043.1µs**、含 bind/checker **1,139.5µs**、含 digest **2,047.2µs**；这些不是严格理论下界；
- 预注册目标 <336µs；旧 replay 配置的 SF0-B = FAIL；
- Phase 4.10-F = PASS；历史 replay-based P10-1 formal = CLOSED。该结论不禁止新建 concurrent/event-driven architecture。

---

## 4. 四层结论严格区分

| 层 | 含义 | 结论 |
|---|---|---|
| L1 | 单机高保真（RTX 2080 Ti，合成 shim，单 rank） | **有收益**：ΔE2E +10.95ms |
| L2-S | 双卡真实 NCCL（合成 shim/合成 GEMM） | **有收益**：ΔE2E +6.46ms |
| L2-R 正确性 | reference router substrate（真实 CUDA top-k/shard-ready） | **正确**（17/17），但不等于生产收益 |
| L2-R replay 路径 E2E | 冻结调度器 + 参考 router timing replay | **不可证明/不可准入**（candidate window 419.8µs；未测真实并发窗口） |

---

## 5. 负结果与适用边界（全部保留）

| 负结果 | 关键数字 |
|---|---|
| H1 历史预测 | Δ −0.0790 RMSE，CI 全负，LOFO 0/5 |
| H2 稳健规划 | E2E Δ −938.58ms，慢 9–10× |
| H2b 算法价值 | +0.11 时隙，动作重合 98.5% |
| W2/W3 排序与门控 | lookahead CI 跨 0；门控空转 |
| A5 全局聚合 | −0.13ms，全序列为负 |
| H7 自适应 | ≡ 固定 B75，oracle 上界 0.0014ms |
| hotspot_random_walk | L1 −1.1ms；L2 −1.7ms；pilot −32.8ms；1D −0.59ms/−3.6ms（适用边界） |
| P10-1C pilot E2E | −19.7ms（setup 主导；completion 仍 +1.95） |
| P10-1D E2E 稳态 | C1≈B0（收益未确立） |
| P10-F0-v1 | 正式不可准入（P4 FAIL） |
| P10-SF0-B | 旧 replay 配置目标 <336µs 不可认证（step-only estimate 1,043.1µs；含 bind/checker 1,139.5µs） |

---

## 6. 关键数字汇总（速查表）

| 项目 | 数字 |
|---|---|
| H1 配对差 | −0.0790 RMSE，CI [−0.1133, −0.0478] |
| H2 E2E 差 | −938.58ms（robust 1042ms vs 基线 ~104–116ms） |
| H2b 调度价值 | +0.11 时隙，CI [0.087, 0.140] |
| Route A 最佳 | S3 11.80 vs fullinfo 10.80（差 1.0 时隙） |
| H5 最佳 | A4 +9.22ms，CI [+8.26, +10.13] |
| H6 最佳选择器 | partial_shards（+0.57~+0.81ms，CI>0） |
| H7 | 自适应 ≡ 固定（Δ=0）；oracle 0.0014ms |
| L1 正式 | ΔE2E +10,953µs，CI [+3,598, +23,148]；吞吐 +24.7% |
| L2-S 正式 | ΔE2E +6,458µs，CI [+3,409, +9,385]；吞吐 +13.7% |
| P10-1D | C1 22.9 vs 28.1（+5.2 时隙）；稳态 E2E≈0 |
| P10-1E | replay/quantized candidate window 419.84µs；scheduler p95 12,290.03µs；P4 FAIL |
| P10-1F | 单步 11.29–12.93ms；分解 92.4%；首提交 8,673.8µs；fast-path estimates 1,043.1/1,139.5/2,047.2µs |

---

## 7. 论文可用主张与证据（摘要版）

完整的 Claims-to-Evidence Matrix（C1–C20，每条含文档 + artifact + 统计）见
`docs/phase4_11/CLAIMS_TO_EVIDENCE_MATRIX.md`。要点：

- 调度侧干预（预测/稳健规划/排序/门控/自适应）无剩余价值：C1–C3、C8、C9；
- 揭示时机与粒度是主导杠杆：C4–C7；
- L1/L2-S 部署收益成立：C10、C11；
- L2-R 参考 router 正确：C12、C13；但 E2E 不可证明：C14–C18；
- hotspot 与全部负结果保留：C19、C20。

**Fail-closed（禁止表述）**：生产 MoE；L3/DeepEP/RDMA 已验证；L2-R E2E 收益；scheduler <336µs 可认证；
P10-1 formal 通过；自适应/预测有价值；排序/门控有收益；L1/L2 外推至多节点或生产 SLA；completion 收益等于 E2E 收益；选择性删除负结果。

---

## 8. 限制

1. L3（多节点 RDMA/NVSHMEM/DeepEP）未验证；V100（sm_70）不支持 DeepEP；
2. L2-R router 是参考实现，不是生产 MoE；真实 expert GEMM/packing/combine 未实现；
3. L1/L2-S 结论基于合成 shim，限定单节点尺度，不构成生产 SLA；
4. scheduler fast-path estimates 基于冻结 Python 实现，不是理论下界；向量化/记忆化未实现（SF0-B FAIL），可行性未证明；
5. 419.8µs 是冻结 workload 的 replay/quantized candidate window（48-token 世界、8 chunk、Rear4GPU），不是直接测得的 concurrent pipeline window；
6. 控制消息 RTT 为 localhost 实测，真实 fabric 未测；
7. completion 与 E2E 分别报告，二者不一定同向（1D 稳态 E2E≈0 即为例）；
8. 统计为预注册配对 bootstrap；绝对墙钟受机器负载影响，跨 run 不直接比较。

---

## 9. 结论

1. **信息揭示时机是唯一被证据支持的杠杆**：更早、更细、token 级分片揭示（partial_shards @ 75%、checkpoint 8）在单卡和双卡真机上带来稳定端到端收益；
2. **调度算法侧无剩余价值**：预测、稳健规划、排序、门控、自适应全部失败或无效，负结果完整保留；
3. **历史 L2-R replay 路径不可准入**：冻结调度器单步延迟约 12ms，candidate window 为 0.42ms；当前 fast-path estimates 为约 1.04/1.14/2.05ms。历史 formal 路径关闭，但真实 concurrent/event-driven pipeline 尚未测量且不在关闭范围内；
4. 全部 artifacts、复现脚本、负结果与适用边界（hotspot_random_walk）已固化，可作为论文与后续研究的基础。

---

## 附录 A：Artifact 与复现

见 `docs/phase4_11/RESULT_ARTIFACT_INDEX.md`（结果表/artifact 索引）与
`docs/phase4_11/ARTIFACT_OVERVIEW.md`（分阶段 artifact、复现命令、硬件要求、期望结果）。

## 附录 B：数据与语料隔离

| 阶段 | base seeds |
|---|---|
| H1/H2 | 642 / 742 / 842 |
| Route A | 1042 / 1142 / 1242 |
| H5–H7 | 2042 / 2142 / 2242 |
| L1/L2 正式 | 3042 / 3142 / 3242 |
| P10-1 pilot/计时 | 4042（dev/val） |
| P10-1 formal corpus | 5042 / 5142 / 5242（**从未生成**） |

## 附录 C：门链判定速查

Phase 3B PASS → H1 FAIL → H2 FAIL → H2a CONDITIONAL PASS → H2b FAIL → W1–W3 无价值 →
Route A PASS → H5 PASS → H6 PASS → H7 FAIL → L1 D1 PASS → L2-S D1 PASS → L2-F0 PASS →
P10-R0 CONDITIONAL PASS → S0 PASS → I1 PASS（17/17）→ P0 CONDITIONAL PASS → T0 PASS →
F0-v1 FAIL → SF0-A PASS → SF0-B FAIL → Phase 4.10-F PASS → P10-1 formal CLOSED。
