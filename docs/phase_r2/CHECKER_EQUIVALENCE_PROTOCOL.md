# Old Checker ↔ Compiled Checker Equivalence Protocol

更新日期：2026-08-10  
状态：协议已在 R2-C0 执行；技术结果 PASS，待 Supervisor

## 1. 权威定义

旧路径 `bind_action + commit_proposal` 是 reference oracle。在 Supervisor 接受 R2-C0
之前，compiled checker/guard 不能成为唯一权威，任何 mismatch 必须 fail closed，并保存
完整 reproducer。

## 2. 配对输入

每个 case 从同一 immutable base world 克隆两个 state，输入必须逐字节/逐字段一致：

- topology、capacity、shared-group、checkpoint；
- revealed ready bitmap；
- revealed token/source/router-derived destination；
- state version 与历史 committed actions；
- structural action order 和 deterministic tie key。

hidden suffix 不传给任一路径；counterfactual suffix 只用于 no-leak metamorphic test。

## 3. 必须完全相等的输出

- legal/reject；
- reject reason class；
- bound opaque token IDs 与 transfer order；
- applied/committed action count；
- next state version；
- holder bitsets、edge/group remaining capacity；
- token seen/committed bitsets；
- token loss/duplication counters；
- serialized canonical next-state digest。

浮点近似不适用；这些均要求 exact equality。

## 4. Test matrix

至少覆盖：

1. 0→6 revealed chunks 的每个 progressive prefix 与 checkpoint8；
2. actual 75% partial view；
3. independently reconstructed token→traffic oracle；
4. deterministic tie cases 和重复运行；
5. hidden suffix counterfactual perturbation；
6. unrevealed token/future top-k access；
7. duplicate token、missing token、already committed token；
8. stale/future state version；
9. invalid edge、source/holder mismatch；
10. edge capacity 与 shared-group capacity overflow；
11. same-slot forwarding 禁止条件；
12. empty proposal、maximal legal proposal、首个非法 action 位于不同位置；
13. token loss/duplication 与 fail-closed exception injection；
14. 两 rank、不同 deterministic seeds、边界 topology/state corpus。

## 5. 执行模式

### Mode A：offline differential

在 clone state 上分别运行 reference 与 compiled path；比较第 3 节全部字段。用于 exhaustive/
property corpus，不进入 timed E2E。

### Mode B：online shadow

旧 checker 产生唯一实际 commit；compiled checker 在 clone state 上 shadow 执行。mismatch
阻止后续通信并落盘。该模式只用于 equivalence gate，不得把 shadow timing 计为 fast-path 收益。

### Mode C：candidate authority

只有 R2-C0 获 Supervisor PASS 后才可申请。即使获批，DynamicGuard mismatch/unknown/error
仍必须 fail closed，并可回退到 reference checker；不得静默接受。

## 6. Acceptance criteria

- 所有 valid/invalid cases exact equivalence = 100%；
- no future access assertions = 100%；
- hidden suffix perturbation 不改变 prefix action/state digest；
- deterministic repetition/tie tests = 100%；
- token integrity、legality = 100%；
- artifact read-back 与 corpus/hash 一致；
- mismatch、timeout、exception 均为 0；
- Supervisor PASS。

性能只在 correctness gate 之后测量；不能以更快为理由容忍任何 semantic mismatch。

## 7. R2-C0 execution result

- E1 Static：360 tests / 0 mismatch；
- E2 Single-step：212 tests / 0 mismatch；
- E3 Trajectory：36 tests / 0 mismatch，含 524 个逐 step 比较；
- checker accept/reject：736 comparisons / 0 mismatch；
- ordered candidate/action：724 comparisons / 0 divergence；
- hidden pending-ready：192 checks / 0 influence；
- hidden suffix：12 paired trajectories / 0 prefix-action change。

Gate 状态为 TECHNICAL PASS / PENDING SUPERVISOR；旧 checker 仍保留为 oracle，
本结果不自动授权 R2-F0。
