# Phase R3-A0/C0：Real Variable-Size AlltoAllv Substrate and Correctness

更新日期：2026-08-10  
状态：**R3-A0/C0 COMPLETE / PENDING SUPERVISOR**  
边界：未运行 R3-P0、formal R3 E2E、expert GEMM、return combine 或 production MoE runtime。

## 1. 结论

R3-A0 第一次建立并实际运行了以下链路：

`reference Router top-k → router-derived destination lists → delta sendcounts/offsets →`
`reference contiguous packing → real NCCL uneven-split all_to_all_single → receive/unpack verification`

该 primitive 命名为 **A2Av-T0**。它使用
`torch.distributed.all_to_all_single(async_op=True)`，并为每个 progressive descriptor
传入真实不等的 `input_split_sizes` 与 `output_split_sizes`。它不是固定 64-byte descriptor
all-reduce，不是 P2P intervention，也不是 production MoE packing。

R3-A0/C0 correctness gate 全部通过：

- 7 个冻结 coverage cases，2 ranks，C/D 两 arms，共 28 arm-rank-cases；
- 196/196 compiled scheduler/checker steps legal；
- 114,688/114,688 sent tokens 被验证接收；
- lost/duplicate/wrong-destination/corruption = **0/0/0/0**；
- unrevealed/future/duplicate-dispatch/stale-dispatch = **0/0/0/0**；
- C/D exact semantic equivalence = **14/14** rank-case pairs；
- local independent read-back = **PASS**。

建议进入 **R3-P0 Supervisor review**，但本轮未运行 P0，也不把 A0/C0 diagnostics
解释为 early-vs-delayed 性能收益。

## 2. Sendcounts provenance

运行时不存在 externally supplied final sendcounts。每个 token 的流程为：

1. reference Router 对实际 token embedding 执行 deterministic top-k；
2. top-k expert ID 映射为 `destination_rank = expert_id % world_size`；
3. 只有已 completed 且已 revealed 的 chunk assignment 才能进入 destination lists；
4. `sendcounts[d] = len(destination_lists[d])`；
5. offsets 由 sendcounts 的 exclusive prefix sum 产生；
6. payload 按 `(destination_rank, token_id, chunk_id, chunk_offset)` 确定性排序并连续化。

`ProgressivePackingState` 不提供传入 sendcounts/offsets 的接口。future、unrevealed、
duplicate 或 stale chunk/token 均在 layout 生成前 fail closed。因此 final sendcounts 不是
先验输入，也没有人工生成 final traffic 后反向匹配 Router。

## 3. Payload identity 与接收验证

reference payload 每个 token 使用 8×int64 record：

`token_id, source_rank, destination_rank, expert_id, chunk_id, chunk_offset, payload_word, checksum`

接收端按照所有 source 的实际发送记录构造 expected registry，并独立验证：

- token identity 与 source；
- destination 必须等于当前接收 rank；
- expert-derived destination；
- payload word 与 checksum；
- expected/received multiset 完全一致。

这是 deterministic reference packing，不称为 production MoE packing。

## 4. Progressive 75% / checkpoint8

每个 rank/case 含 8 个 router chunks：

- chunk 0–5：各自完成后 reveal，并各生成一个 delta descriptor；
- chunk 6：CUDA complete 后仅进入 pending-ready，不可打包、不可发送；
- chunk 7：完成后与 chunk 6 一起 reveal；最后 descriptor 固定覆盖 `[6,7]`；
- 每个 arm 共 7 个 descriptors；fast path runtime BFS=0、full rebuild=0。

所有 14 个 early rank-case arms 的第一次真实 payload API call 都早于 final router host
completion；margin p50 = 25,465.364µs，最小 = 16,931.085µs。所有 delayed arms 的
98 个 payload calls 均在 final router completion 后。该事实只证明 incremental substrate
不依赖 full final counts；它不是 R3-P0 paired performance 或 device-overlap 结论。

## 5. C/D exact correctness equivalence

每个 case、每个 rank 的 C 与 D 均逐项比较：

- same Router inputs and top-k digests；
- same router assignment multiset；
- same final destination mapping；
- same seven ordered delta descriptors；
- same sendcounts、offsets、token counts、bytes 与 payload multiset digest；
- same compiled scheduler action sequence；
- same final payload multiset and total bytes。

14/14 comparisons 全部通过。唯一差异为通信调用时机。

## 6. Coverage 与 variable traffic

每个 case/rank/arm 的 Router token 总数固定为 4,096；coverage fixture 在运行前冻结，
仅用于满足明确的 correctness edge cases，不按性能收益选择 workload。

| case | frozen property | early destination totals `[rank0, rank1]` | result |
|---|---|---:|:---:|
| balanced | default unbiased reference Router | [4,073, 4,119] | PASS |
| skewed | fixed expert-0 bias | [6,808, 1,384] | PASS |
| all_to_one_like | fixed expert-0/2 bias | [8,192, 0] | PASS |
| zero_sized_pair | fixed expert-1/3 bias | [0, 8,192] | PASS |
| empty_shard | chunk 2 active size 0；总 token 不变 | [4,018, 4,174] | PASS |
| single_token_shard | chunk 1 active size 1；总 token 不变 | [4,133, 4,059] | PASS |
| multiple_progressive_shards | fixed sizes 128/256/384/512/640/768/512/896 | [4,041, 4,151] | PASS |

Early descriptors 的 196 个 src→dst pair sizes：

- p50/p95/p99/max = **257/764/1,024/1,024 tokens**；
- nonzero min = **1 token**；
- zero-sized pairs = **34**；
- distinct sizes = **95**。

因此实际通信具有不等 src→dst token/byte counts，而非固定大小伪装的 AlltoAllv。

## 7. Diagnostics（非 PASS 条件）

| metric | count | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|---:|
| router final host latency | 28 | 36,305.365 | 43,418.814 | 43,604.691 | 43,623.954 |
| per-chunk router CUDA | 224 | 6,489.904 | 7,832.008 | 9,587.102 | 11,216.672 |
| count/offset construction | 196 | 622.978 | 1,624.360 | 1,756.547 | 1,768.497 |
| reference packing | 196 | 1,651.776 | 4,462.444 | 5,021.426 | 5,156.336 |
| payload H2D | 196 | 115.290 | 201.262 | 242.530 | 278.657 |
| delta-count exchange completion | 196 | 220.017 | 649.246 | 2,432.946 | **166,851.153** |
| AlltoAllv async submit | 196 | 100.337 | 156.917 | 189.300 | 213.572 |
| AlltoAllv completion | 196 | 107.099 | 174.035 | 199.337 | 225.062 |
| unpack + Python verification | 28 | 32,688.532 | 66,252.855 | 156,586.628 | 189,652.463 |

总 reference payload bytes（C+D、两 ranks、七 cases）= **7,340,032 bytes**。

Count exchange 的 166.851ms max tail 与 Python verifier 的长尾必须保留。它们不阻塞
A0/C0 correctness，但后续 P0 必须将 count exchange、packing 与 verification scope
清楚分离，不得从本轮 diagnostics 推断 E2E 收益。

## 8. Independent read-back

本地 analyzer 不信任 canonical summary，重新执行：

- 196 descriptor structure/count/offset/byte checks；
- 196 recv-split = sendcount transpose checks；
- 28 semantic arm-rank-case checks；
- 14 C/D equivalence checks；
- 14 early-first-call-before-final checks；
- 98 delayed-call-after-final checks；
- 7 coverage checks；
- traffic distribution 与 diagnostics exact recomputation。

全部 PASS。canonical result 的服务器/本地 SHA-256 相同。

## 9. 禁止项与停止点

未实现或运行：expert GEMM、return-path combine、production MoE runtime、DeepEP、
PCCL production integration、新 transport variant、predictor/robust/adaptive、formal R3
E2E 或 R3-P0。未修改 AICCL scheduling semantics、partial_shards@75%、checkpoint8；
未人工延长 Router；未按收益选择 workload；未创建 Subagent。

R3-A0/C0 到此停止。技术上建议 Supervisor 授权下一阶段 **R3-P0：progressive early
AlltoAllv vs identical delayed AlltoAllv**，但不自动进入。
