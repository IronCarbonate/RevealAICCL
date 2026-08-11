# Route A 结果：Reveal 机制敏感性（信息价值量化）

更新日期：2026-08-04
协议：`docs/phase4_6/ROUTE_A_REVEAL_SENSITIVITY_PROTOCOL.md`（预注册）
判定：**H-A1 / H-A2 / H-A3 / H-A4 全部成立**——信息揭示节奏是 completion 的主导瓶颈，且已量化

## 1. 执行与验证

- 新 corpus：base seeds `(1042, 1142, 1242)`，45 条 sequence 与正式 45 条 digest 零交集（已校验）；
- 等价性门：S0（冻结节奏）在新运行器上于**正式 corpus 300/300 逐项复现**正式 `partial_current_only`（completion/first_action/legality/actions 0 差异）；
- 主实验：新 corpus test split 15 sequence × 20 coordinates = 300 坐标，6 档揭示 × 3 方法 = 5,400 episode，legality 100%、timeout 0。

## 2. 主结果（completion mean，slots）

| 档位 | full-reveal slot | partial | wait | fullinfo |
|---|---:|---:|---:|---:|
| S4 | 32 | 36.18 | 42.80 | 10.80 |
| S0（冻结） | 16 | 20.95 | 26.80 | 10.80 |
| S5 | 8 | 16.30 | 18.80 | 10.80 |
| S1 | 8 | 14.92 | 18.80 | 10.80 |
| S2 | 4 | 13.40 | 14.80 | 10.80 |
| S3 | 1 | **11.80** | 11.80 | 10.80 |

配对 bootstrap（partial，S_i vs S0；正值=更早揭示更好）：

| 对比 | mean | 95% CI |
|---|---:|---:|
| S1（slot 8） | +6.03 | [+5.55, +6.50] |
| S2（slot 4） | +7.56 | [+6.88, +8.25] |
| S3（slot 1） | +9.16 | [+8.36, +9.94] |
| S5（slot 8，粗粒度） | +4.66 | [+4.11, +5.20] |
| S4（slot 32） | −15.22 | [−15.51, −14.91] |

## 3. 假设检验

### H-A1（单调性）：成立

completion 随 full-reveal slot 提前严格下降：36.18 → 20.95 → 14.92 → 13.40 → 11.80；所有配对 CI 排除 0。平均而言，full-reveal 每提前 1 slot 节省约 **0.4–0.75 completion slot**（slots 16→8 约 0.75/slot；8→4 约 0.4/slot）。

### H-A2（信息差距随提前揭示归零）：成立

S3（slot 1）partial = 11.80，与 fullinfo 10.80 仅差 **1.0 slot**；S2（slot 4）差 2.6 slots。信息延迟差距从 S0 的 ~10.2 slots 压缩到 S3 的 ~1 slot。

### H-A3（剩余差距为调度效率、与揭示无关）：成立

- 新 corpus LB = 4.25；fullinfo = 10.80（regret vs LB = **6.55**，与正式 corpus 的 ~6.5 一致，且与揭示节奏无关）；
- S3 的 partial 相对 fullinfo 仅 +1.0 slot——信息几乎即时时，简单调度器的信息利用已接近完美；剩余 regret 主体是"全信息下 direct scheduler 与 provable LB 的效率差距"，属于独立于信息的研究问题。

### H-A4（mode 影响远小于节奏）：成立

- S0→S3 各 mode 增益 8.07–9.48 slots（5 档接近），模式差异 <1.5 slots，节奏效应是 mode 效应的 6 倍以上。

## 4. 次要发现

- **揭示粒度也有价值**：S1（细粒度 (0,.25,.5,.75,1)）vs S5（粗粒度 (0,.5,1)）同为 full slot 8，细粒度好 1.38 slots——不仅"何时全揭示"重要，"分几步揭示"也重要。
- wait 在除 S3 外所有档位都差于 partial：提前行动的价值在揭示越慢时越大（S4 差 6.6、S0 差 5.9、S1 差 3.9、S3 差 0）。

## 5. 结论与含义

1. **信息延迟是 completion regret 的主导瓶颈，已定量**：在本 proxy 上，把 full reveal 从 slot 16 提前到 slot 8/4/1 分别节省约 6.0/7.6/9.2 slots；推迟到 32 增加 15.2 slots。
2. 简单调度器（partial）在信息可用时已接近最优（S3 距全信息参照仅 1 slot），信息利用不是问题。
3. 剩余 ~6.5 slots 的"调度效率差距"（fullinfo vs LB）与信息无关，是 direct scheduler 相对 provable 下界的固有差距，可作独立研究问题（离线最优/更优 packing）。
4. 对 AICCL 研究的直接含义：**提高 reveal 频率/密度（更早、更细）是唯一被证据支持的可提升 completion 的杠杆**；在现有揭示语义下，调度侧已无可榨取的收益。

## 6. 约束

- 未修改 production 代码；正式 artifacts 只读；新 corpus/新输出目录；legality 100%；H2=FAIL、Phase 5 CLOSED 维持。
