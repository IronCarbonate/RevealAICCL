#!/usr/bin/env python
"""Convert a learned chunk schedule (real_strategy_list) into MSCCL XML.

This translates each (t, chunk, edge) > 0 decision into MSCCLang `chunk.copy()` operations,
then lowers to MSCCL XML via msccl-tools.

Assumptions for this converter (matching inference usage):
- B == 1 (single graph/topology instance).
- Switch semantics: a switch can only forward chunks it received in the PREVIOUS epoch.
    Concretely, if a switch sends chunk c at epoch t, then chunk c must have been delivered to that
    switch at epoch t-1 (switch has no persistent buffer and cannot forward within the same epoch).

Key limitation: MSCCL runtime ranks map to GPU processes; switches are virtual vertices.
We eliminate switch vertices by backtracing each switch->GPU send at epoch t to the GPU that injected
that chunk into the switch pipeline at earlier epochs, and emit an equivalent GPU->GPU `copy` at epoch t.

Input loading:
- --input <path>: torch.load() file containing a dict with keys:
    real_strategy_list, edge_src_idx, edge_dst_idx, pre_condition, (optional) is_switch, chunk_mask

Output:
- Prints XML to stdout or writes to --out.

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch


def _ensure_local_msccl_tools_on_path() -> None:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        cand = p / "msccl-tools"
        if (cand / "msccl").is_dir():
            sys.path.insert(0, str(cand))
            return


_ensure_local_msccl_tools_on_path()

from msccl.language import Buffer, Check, MSCCLProgram, XML, chunk  # noqa: E402
from msccl.language.collectives import AllGather, AllToAll  # noqa: E402
from msccl.topologies import fully_connected  # noqa: E402
from msccl.topologies.topology import Topology  # noqa: E402


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: int
    origin_rank: int  # mapped rank id [0..nranks)
    origin_index: int  # within origin input buffer
    global_index: int  # output buffer index


def _to_bool(x: Optional[torch.Tensor], *, V: int, device: torch.device) -> torch.Tensor:
    if x is None:
        return torch.zeros((V,), dtype=torch.bool, device=device)
    if x.shape != (V,):
        raise ValueError(f"is_switch must have shape (V,), got {tuple(x.shape)}")
    return x.to(device=device, dtype=torch.bool)


def _require_b1(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() < 1:
        raise ValueError(f"{name} must have batch dim; got shape {tuple(x.shape)}")
    if x.shape[0] != 1:
        raise ValueError(f"{name} must have B==1 for inference/compilation; got B={x.shape[0]}")
    return x[0]


def _infer_gpu_rank_map(is_switch: torch.Tensor) -> Tuple[List[int], Dict[int, int]]:
    gpu_vertices = [int(v) for v in range(int(is_switch.numel())) if not bool(is_switch[v].item())]
    if not gpu_vertices:
        raise ValueError("No GPU vertices found: is_switch is True for all V")
    v2r = {v: i for i, v in enumerate(gpu_vertices)}
    return gpu_vertices, v2r


def _infer_allgather_chunk_meta(
    *,
    pre_condition_b: torch.Tensor,  # (C,V)
    is_switch: torch.Tensor,  # (V,)
    chunk_mask_b: Optional[torch.Tensor],  # (C,)
) -> Tuple[List[ChunkMeta], int, Dict[int, int]]:
    if pre_condition_b.dim() != 2:
        raise ValueError(f"pre_condition[b] must have shape (C,V), got {tuple(pre_condition_b.shape)}")
    C, V = pre_condition_b.shape

    gpu_vertices, v2r = _infer_gpu_rank_map(is_switch)
    nranks = len(gpu_vertices)

    if chunk_mask_b is None:
        valid = torch.ones((C,), dtype=torch.bool, device=pre_condition_b.device)
    else:
        if chunk_mask_b.shape != (C,):
            raise ValueError(f"chunk_mask[b] must have shape (C,), got {tuple(chunk_mask_b.shape)}")
        valid = chunk_mask_b.to(dtype=torch.bool)

    # For each chunk, find exactly one GPU origin.
    owners: List[Optional[int]] = [None] * C  # mapped origin rank
    for c in range(C):
        if not bool(valid[c].item()):
            continue
        row = pre_condition_b[c]
        src_vs = (row > 0).nonzero(as_tuple=False).view(-1).tolist()
        src_vs = [int(v) for v in src_vs if not bool(is_switch[v].item())]
        if len(src_vs) != 1:
            raise ValueError(
                f"Chunk {c}: expected exactly 1 GPU owner in pre_condition, got {len(src_vs)} (gpu_vs={src_vs})."
            )
        v = src_vs[0]
        owners[c] = v2r[v]

    # Infer chunk_factor: must be uniform across ranks for AllGather.
    counts = [0] * nranks
    for c in range(C):
        if owners[c] is not None:
            counts[int(owners[c])] += 1
    nonzero_counts = [cnt for cnt in counts if cnt != 0]
    if not nonzero_counts:
        raise ValueError("No valid chunks found to infer chunk_factor")
    chunk_factor = max(nonzero_counts)
    if any(cnt not in (0, chunk_factor) for cnt in counts):
        raise ValueError(
            "Non-uniform chunks per rank in pre_condition; cannot model as AllGather. "
            f"Per-rank counts={counts}, inferred chunk_factor={chunk_factor}."
        )

    # Assign origin_index within each rank by chunk_id ordering.
    per_rank_chunks: List[List[int]] = [[] for _ in range(nranks)]
    for c in range(C):
        r = owners[c]
        if r is None:
            continue
        per_rank_chunks[int(r)].append(c)
    for r in range(nranks):
        per_rank_chunks[r].sort()
        if per_rank_chunks[r] and len(per_rank_chunks[r]) != chunk_factor:
            raise ValueError(
                "Rank has chunks but not equal to chunk_factor; cannot build AllGather layout. "
                f"rank={r}, got={len(per_rank_chunks[r])}, chunk_factor={chunk_factor}."
            )

    chunk_id_to_global_index: Dict[int, int] = {}
    metas: List[ChunkMeta] = []
    for r in range(nranks):
        for origin_index, c in enumerate(per_rank_chunks[r]):
            global_index = r * chunk_factor + origin_index
            chunk_id_to_global_index[int(c)] = int(global_index)
            metas.append(
                ChunkMeta(
                    chunk_id=int(c),
                    origin_rank=int(r),
                    origin_index=int(origin_index),
                    global_index=int(global_index),
                )
            )

    metas.sort(key=lambda m: m.chunk_id)
    return metas, int(chunk_factor), chunk_id_to_global_index


@dataclass(frozen=True)
class AllToAllChunkMeta:
    """Metadata for AllToAll chunk mapping.
    
    For AllToAll with N ranks and chunk_factor C:
    - Total chunks = N * N * C
    - Chunk at input[r][i] (rank r, index i) goes to output[i // C][r * C + i % C]
    - i.e., chunk from rank r destined for rank (i // C) ends up at output index (r * C + i % C)
    """
    chunk_id: int
    origin_rank: int      # source rank
    origin_index: int     # index within source input buffer
    dest_rank: int        # destination rank
    dest_index: int       # index within destination output buffer


def _infer_alltoall_chunk_meta(
    *,
    pre_condition_b: torch.Tensor,  # (C,V)
    post_condition_b: torch.Tensor,  # (C,V)
    is_switch: torch.Tensor,  # (V,)
    chunk_mask_b: Optional[torch.Tensor],  # (C,)
) -> Tuple[List[AllToAllChunkMeta], int, Dict[int, Tuple[int, int, int, int]]]:
    """Infer AllToAll chunk metadata from pre/post conditions.
    
    For AllToAll:
    - Each chunk has exactly one source GPU (pre_condition)
    - Each chunk has exactly one destination GPU (post_condition)
    - chunk_factor = chunks per (src, dst) pair
    
    Returns:
        metas: List of AllToAllChunkMeta
        chunk_factor: chunks per rank pair
        chunk_id_to_info: dict mapping chunk_id -> (origin_rank, origin_index, dest_rank, dest_index)
    """
    if pre_condition_b.dim() != 2:
        raise ValueError(f"pre_condition[b] must have shape (C,V), got {tuple(pre_condition_b.shape)}")
    C, V = pre_condition_b.shape

    gpu_vertices, v2r = _infer_gpu_rank_map(is_switch)
    nranks = len(gpu_vertices)

    if chunk_mask_b is None:
        valid = torch.ones((C,), dtype=torch.bool, device=pre_condition_b.device)
    else:
        if chunk_mask_b.shape != (C,):
            raise ValueError(f"chunk_mask[b] must have shape (C,), got {tuple(chunk_mask_b.shape)}")
        valid = chunk_mask_b.to(dtype=torch.bool)

    # For each chunk, find source and destination GPU
    chunk_info: List[Optional[Tuple[int, int]]] = [None] * C  # (src_rank, dst_rank)
    for c in range(C):
        if not bool(valid[c].item()):
            continue
        # Find source
        pre_row = pre_condition_b[c]
        src_vs = (pre_row > 0).nonzero(as_tuple=False).view(-1).tolist()
        src_vs = [int(v) for v in src_vs if not bool(is_switch[v].item())]
        if len(src_vs) != 1:
            raise ValueError(
                f"AllToAll Chunk {c}: expected exactly 1 GPU source in pre_condition, got {len(src_vs)}"
            )
        src_rank = v2r[src_vs[0]]
        
        # Find destination
        post_row = post_condition_b[c]
        dst_vs = (post_row > 0).nonzero(as_tuple=False).view(-1).tolist()
        dst_vs = [int(v) for v in dst_vs if not bool(is_switch[v].item())]
        if len(dst_vs) != 1:
            raise ValueError(
                f"AllToAll Chunk {c}: expected exactly 1 GPU destination in post_condition, got {len(dst_vs)}"
            )
        dst_rank = v2r[dst_vs[0]]
        chunk_info[c] = (src_rank, dst_rank)

    # Group chunks by (src, dst) pair to infer chunk_factor
    pair_chunks: Dict[Tuple[int, int], List[int]] = {}
    for c in range(C):
        if chunk_info[c] is None:
            continue
        src_rank, dst_rank = chunk_info[c]
        key = (src_rank, dst_rank)
        if key not in pair_chunks:
            pair_chunks[key] = []
        pair_chunks[key].append(c)

    # Infer chunk_factor (should be uniform across all pairs)
    if not pair_chunks:
        raise ValueError("No valid chunks found for AllToAll")
    
    counts = [len(chunks) for chunks in pair_chunks.values()]
    chunk_factor = max(counts)
    # For AllToAll, we expect all (src, dst) pairs to have the same chunk_factor
    # But some pairs might have 0 chunks (no communication needed)
    
    # Build metadata
    # For AllToAll: input buffer has N*chunk_factor chunks per rank
    # Chunk at input[r][i] goes to rank (i // chunk_factor), output index (r * chunk_factor + i % chunk_factor)
    metas: List[AllToAllChunkMeta] = []
    chunk_id_to_info: Dict[int, Tuple[int, int, int, int]] = {}
    
    # Sort chunks within each pair for deterministic ordering
    for key in pair_chunks:
        pair_chunks[key].sort()
    
    # Assign origin_index and dest_index based on AllToAll layout
    # For each (src, dst) pair, chunks are indexed within that pair
    for (src_rank, dst_rank), chunks in pair_chunks.items():
        for idx_in_pair, c in enumerate(chunks):
            # origin_index: position in source's input buffer
            # For AllToAll: input[src][dst * chunk_factor + idx_in_pair]
            origin_index = dst_rank * chunk_factor + idx_in_pair
            # dest_index: position in destination's output buffer
            # For AllToAll: output[dst][src * chunk_factor + idx_in_pair]
            dest_index = src_rank * chunk_factor + idx_in_pair
            
            meta = AllToAllChunkMeta(
                chunk_id=int(c),
                origin_rank=int(src_rank),
                origin_index=int(origin_index),
                dest_rank=int(dst_rank),
                dest_index=int(dest_index),
            )
            metas.append(meta)
            chunk_id_to_info[int(c)] = (src_rank, origin_index, dst_rank, dest_index)
    
    metas.sort(key=lambda m: m.chunk_id)
    return metas, int(chunk_factor), chunk_id_to_info


def _collect_used_edges_for_chunk(sent_b: torch.Tensor, c: int) -> List[int]:
    # sent_b: (C,E)
    row = sent_b[c]
    return (row > 0).nonzero(as_tuple=False).view(-1).tolist()


def _build_incoming_edges_by_dst(edge_dst_idx: torch.Tensor) -> List[List[int]]:
    E = int(edge_dst_idx.numel())
    V = int(edge_dst_idx.max().item()) + 1 if E > 0 else 0
    incoming: List[List[int]] = [[] for _ in range(V)]
    for e in range(E):
        incoming[int(edge_dst_idx[e].item())].append(e)
    return incoming


def _backtrace_switch_source(
    *,
    t: int,
    c: int,
    sw: int,
    sent_bool: List[torch.Tensor],
    edge_src_idx: torch.Tensor,
    incoming_edges: List[List[int]],
    is_switch: torch.Tensor,
) -> Tuple[int, int]:
    """Given a switch sw that is the *sender* at epoch t, find the GPU that injected the chunk.

    Switch semantics: to have chunk at sw at start of epoch t, it must have been delivered to sw at epoch t-1.
    If that delivery was from another switch, continue backtracing.
    
    Returns:
        (gpu_vertex_id, gpu_send_epoch): The GPU that originally sent the chunk and the epoch it sent.
    """
    cur_t = int(t)
    cur_node = int(sw)
    while bool(is_switch[cur_node].item()):
        if cur_t <= 0:
            raise ValueError(f"Switch {cur_node} sends chunk {c} at t=0, impossible under t->t+1 forwarding")
        prev_t = cur_t - 1
        candidates = []
        for e in incoming_edges[cur_node]:
            if bool(sent_bool[prev_t][c, e].item()):
                candidates.append(int(e))
        if not candidates:
            raise ValueError(
                f"Backtrace failed: switch {cur_node} sends chunk {c} at t={cur_t}, but no incoming edge at t={prev_t} delivered it"
            )
        # Deterministic choice if multiple sources fed the same switch inbox.
        e0 = candidates[0]
        cur_node = int(edge_src_idx[e0].item())
        cur_t = prev_t

    # cur_t is now the epoch when the GPU actually sent the chunk
    return int(cur_node), int(cur_t)


def compute_transfers(
    *,
    real_strategy_list: Sequence[torch.Tensor],
    edge_src_idx: torch.Tensor,
    edge_dst_idx: torch.Tensor,
    pre_condition: torch.Tensor,
    is_switch: torch.Tensor,
    chunk_mask: Optional[torch.Tensor],
) -> List[List[Tuple[int, int, int]]]:
    """Compute per-round transfers as (src_vertex, dst_vertex, chunk_id).

    These are GPU-vertex ids (not mapped ranks yet).
    """

    if not real_strategy_list:
        raise ValueError("real_strategy_list is empty")

    src = edge_src_idx.long().cpu()
    dst = edge_dst_idx.long().cpu()

    pre_b = _require_b1(pre_condition, "pre_condition").detach().cpu()  # (C,V)
    C, V = pre_b.shape
    is_sw = is_switch.detach().cpu().to(dtype=torch.bool)

    if pre_condition.shape[0] != 1:
        raise ValueError("This converter requires inference B==1; please pass a single-instance schedule")

    if chunk_mask is None:
        valid = torch.ones((C,), dtype=torch.bool)
    else:
        valid = _require_b1(chunk_mask, "chunk_mask").detach().cpu().to(dtype=torch.bool)
        if valid.shape != (C,):
            raise ValueError(f"chunk_mask must have shape (B,C) with B==1; got {tuple(chunk_mask.shape)}")

    # initial GPU holders per chunk
    holders: List[Set[int]] = [set() for _ in range(C)]
    for c in range(C):
        if not bool(valid[c].item()):
            continue
        vs = (pre_b[c] > 0).nonzero(as_tuple=False).view(-1).tolist()
        for v in vs:
            v = int(v)
            if bool(is_sw[v].item()):
                continue
            holders[c].add(v)

    # Normalize schedule to CPU bool tensors with shape (T, C, E)
    sent_bool: List[torch.Tensor] = []
    E = int(src.numel())
    for t, s in enumerate(real_strategy_list):
        if not torch.is_tensor(s) or s.dim() != 3:
            raise ValueError(f"real_strategy_list[{t}] must be a (B,C,E) tensor")
        if s.shape[0] != 1:
            raise ValueError(f"real_strategy_list[{t}] must have B==1; got B={s.shape[0]}")
        if s.shape[1] != C or s.shape[2] != E:
            raise ValueError(
                f"real_strategy_list[{t}] shape mismatch: expected (1,{C},{E}), got {tuple(s.shape)}"
            )
        sent_bool.append((s[0].detach().cpu() > 0) & valid.view(C, 1))

    incoming_edges = _build_incoming_edges_by_dst(dst)
    if len(incoming_edges) < V:
        # edge_dst_idx might not cover all vertices; pad.
        incoming_edges = incoming_edges + [[] for _ in range(V - len(incoming_edges))]

    # 1) Forward validate source availability with switch t->t+1 inbox semantics.
    gpu_buf = torch.zeros((C, V), dtype=torch.bool)
    sw_inbox = torch.zeros((C, V), dtype=torch.bool)
    for c in range(C):
        if not bool(valid[c].item()):
            continue
        gpu_buf[c] = (pre_b[c] > 0) & (~is_sw)

    for t in range(len(sent_bool)):
        avail = gpu_buf | sw_inbox
        # Source availability check for every scheduled edge.
        for e in range(E):
            u = int(src[e].item())
            chosen_chunks = sent_bool[t][:, e].nonzero(as_tuple=False).view(-1).tolist()
            for c in chosen_chunks:
                if not bool(avail[c, u].item()):
                    raise ValueError(
                        f"Invalid schedule: t={t}, edge={e} ({u}->{int(dst[e].item())}), chunk={c} not available at source at round start"
                    )

        # Compute received (boolean OR) for this epoch.
        received = torch.zeros((C, V), dtype=torch.bool)
        for e in range(E):
            v = int(dst[e].item())
            chosen_chunks = sent_bool[t][:, e].nonzero(as_tuple=False).view(-1).tolist()
            for c in chosen_chunks:
                received[c, v] = True

        recv_gpu = received & (~is_sw)
        recv_sw = received & is_sw
        gpu_buf = (gpu_buf | recv_gpu)
        sw_inbox = recv_sw  # overwrite each epoch

    # 2) Emit GPU-only transfers per epoch by backtracing switch senders.
    # Key change: place the transfer at the epoch when the SOURCE GPU actually sends,
    # not when the destination GPU receives (which may be later due to switch hops).
    T = len(sent_bool)
    per_round: List[List[Tuple[int, int, int]]] = [[] for _ in range(T)]
    
    # We also need to track when each (src_gpu, dst_gpu, chunk) transfer's data arrives at dst.
    # For validation, we update holders at the arrival epoch, not the send epoch.
    # arrival_events[t] = list of (dst_gpu, chunk) that arrive at epoch t
    arrival_events: List[List[Tuple[int, int]]] = [[] for _ in range(T)]
    
    for t in range(T):
        for e in range(E):
            u = int(src[e].item())
            v = int(dst[e].item())
            if bool(is_sw[v].item()):
                # Deliveries to switches are virtual; they only matter for later switch->GPU sends.
                continue
            chosen_chunks = sent_bool[t][:, e].nonzero(as_tuple=False).view(-1).tolist()
            for c in chosen_chunks:
                if bool(is_sw[u].item()):
                    u_gpu, send_epoch = _backtrace_switch_source(
                        t=t,
                        c=int(c),
                        sw=u,
                        sent_bool=sent_bool,
                        edge_src_idx=src,
                        incoming_edges=incoming_edges,
                        is_switch=is_sw,
                    )
                    # Place transfer at the epoch when GPU actually sends
                    per_round[send_epoch].append((int(u_gpu), int(v), int(c)))
                    # Data arrives at dst at epoch t (when switch delivers to GPU)
                    arrival_events[t].append((int(v), int(c)))
                else:
                    # Direct GPU->GPU transfer: send and arrive in same epoch
                    per_round[t].append((int(u), int(v), int(c)))
                    arrival_events[t].append((int(v), int(c)))

    # Validate GPU sources and update holders at arrival time
    # First pass: validate that source GPU has the chunk at send time
    holders_at_send: List[Set[int]] = [set(h) for h in holders]  # copy initial state
    for t in range(T):
        for (u_gpu, _v_gpu, c) in per_round[t]:
            if u_gpu not in holders_at_send[c]:
                raise ValueError(
                    f"Invalid GPU-only transfer after switch elimination: t={t}, chunk={c} src_gpu={u_gpu} not available at round start"
                )
        # Update holders at arrival time (not send time)
        for (v_gpu, c) in arrival_events[t]:
            holders_at_send[c].add(v_gpu)

    return per_round


def build_msccl_xml(
    *,
    per_round_transfers: List[List[Tuple[int, int, int]]],
    pre_condition: torch.Tensor,
    post_condition: Optional[torch.Tensor] = None,
    is_switch: torch.Tensor,
    chunk_mask: Optional[torch.Tensor],
    program_name: str,
    topology_mode: str,
    do_check: bool,
    collective_type: str = "allgather",
    instances: int = 1,
    instr_fusion: bool = False,
) -> str:
    """Build MSCCL XML from per-round transfers.
    
    Args:
        per_round_transfers: List of transfers per round, each transfer is (src_vertex, dst_vertex, chunk_id)
        pre_condition: (1, C, V) tensor of pre-conditions
        post_condition: (1, C, V) tensor of post-conditions (required for alltoall)
        is_switch: (V,) bool tensor indicating switch nodes
        chunk_mask: (1, C) bool tensor of valid chunks
        program_name: Name for the MSCCL program
        topology_mode: "fully_connected" or "from-transfers"
        do_check: Whether to run MSCCLang Check()
        collective_type: "allgather" or "alltoall"
        instances: Number of instances for MSCCL
        instr_fusion: Whether MSCCLang fuses ops (e.g., recv+send -> rcs)
    """
    pre_b = _require_b1(pre_condition, "pre_condition").detach().cpu()  # (C,V)
    C, V = pre_b.shape
    is_sw = is_switch.detach().cpu().to(dtype=torch.bool)

    cm_b = None
    if chunk_mask is not None:
        cm_b = _require_b1(chunk_mask, "chunk_mask").detach().cpu().to(dtype=torch.bool)

    gpu_vertices, v2r = _infer_gpu_rank_map(is_sw)
    nranks = len(gpu_vertices)

    # Build topology.
    if topology_mode == "fully_connected":
        topo = fully_connected(nranks)
    elif topology_mode == "from-transfers":
        links = [[0] * nranks for _ in range(nranks)]
        for round_tr in per_round_transfers:
            for (u, v, _c) in round_tr:
                ru = v2r[u]
                rv = v2r[v]
                if ru == rv:
                    continue
                links[rv][ru] = 1
        topo = Topology(f"Transfers(n={nranks})", links)
    else:
        raise ValueError(f"Unknown topology-mode: {topology_mode}")

    collective_type_lower = collective_type.lower()
    
    if collective_type_lower == "allgather":
        metas, chunk_factor, chunk_id_to_global = _infer_allgather_chunk_meta(
            pre_condition_b=pre_b,
            is_switch=is_sw,
            chunk_mask_b=cm_b,
        )
        collective = AllGather(nranks, chunk_factor, False)

        with MSCCLProgram(program_name, topo, collective, instances, instr_fusion=instr_fusion):
            # 0) Place each origin input chunk into its global output slot.
            # This mirrors the standard MSCCL allgather layout.
            for m in metas:
                c_ref = chunk(m.origin_rank, Buffer.input, m.origin_index)
                if c_ref is None:
                    raise ValueError(
                        f"Chunk {m.chunk_id}: origin input ref missing at rank={m.origin_rank}, index={m.origin_index}."
                    )
                c_ref.copy(m.origin_rank, buffer=Buffer.output, index=m.global_index)

            # 1) Apply schedule transfers in round order.
            for _t, round_tr in enumerate(per_round_transfers):
                for (u, v, c) in round_tr:
                    ru = v2r[u]
                    rv = v2r[v]
                    gidx = chunk_id_to_global.get(int(c))
                    if gidx is None:
                        # invalid/padded chunk
                        continue
                    src_ref = chunk(ru, Buffer.output, int(gidx))
                    if src_ref is None:
                        raise ValueError(
                            f"At round transfer, source ref missing: t? chunk={c}, src_rank={ru}, out_index={gidx}."
                        )
                    src_ref.copy(rv, buffer=Buffer.output, index=int(gidx))

            xml = _xml_and_optional_check(do_check)

    elif collective_type_lower == "alltoall":
        if post_condition is None:
            raise ValueError("post_condition is required for AllToAll collective")
        post_b = _require_b1(post_condition, "post_condition").detach().cpu()
        
        metas, chunk_factor, chunk_id_to_info = _infer_alltoall_chunk_meta(
            pre_condition_b=pre_b,
            post_condition_b=post_b,
            is_switch=is_sw,
            chunk_mask_b=cm_b,
        )
        collective = AllToAll(nranks, chunk_factor, inplace=False)

        with MSCCLProgram(program_name, topo, collective, instances, instr_fusion=instr_fusion):
            # For AllToAll: each chunk goes from input[src][origin_index] to output[dst][dest_index]
            # First, copy each chunk from input to output at destination
            for m in metas:
                c_ref = chunk(m.origin_rank, Buffer.input, m.origin_index)
                if c_ref is None:
                    raise ValueError(
                        f"AllToAll Chunk {m.chunk_id}: origin input ref missing at rank={m.origin_rank}, index={m.origin_index}."
                    )
                # Copy to destination's output buffer
                c_ref.copy(m.dest_rank, buffer=Buffer.output, index=m.dest_index)

            xml = _xml_and_optional_check(do_check)
    else:
        raise ValueError(f"Unknown collective_type: {collective_type}. Supported: 'allgather', 'alltoall'")

    return xml


def tensors_to_msccl_xml(
    *,
    real_strategy_list: Sequence[torch.Tensor],
    pre_condition: torch.Tensor,
    edge_src_idx: torch.Tensor,
    edge_dst_idx: torch.Tensor,
    is_switch: Optional[torch.Tensor] = None,
    chunk_mask: Optional[torch.Tensor] = None,
    post_condition: Optional[torch.Tensor] = None,
    program_name: str = "aiccl_strategy",
    topology_mode: str = "from-transfers",
    do_check: bool = True,
    collective_type: str = "allgather",
    instances: int = 1,
    instr_fusion: bool = False,
) -> str:
    """Convert in-memory tensors directly to MSCCL XML.

    Expected shapes:
    - pre_condition: (1, C, V)
    - post_condition: (1, C, V) - required for alltoall
    - each strategy tensor: (1, C, E)
    - edge_src_idx/edge_dst_idx: (E,)
    - is_switch: (V,) bool (optional)
    - chunk_mask: (1, C) bool (optional)
    
    Args:
        collective_type: "allgather" or "alltoall"
        instances: Number of instances for MSCCL
    """

    if pre_condition.dim() != 3 or pre_condition.shape[0] != 1:
        raise ValueError(f"pre_condition must have shape (1,C,V); got {tuple(pre_condition.shape)}")
    V = int(pre_condition.shape[2])
    is_sw = _to_bool(is_switch, V=V, device=pre_condition.device)

    per_round = compute_transfers(
        real_strategy_list=real_strategy_list,
        edge_src_idx=edge_src_idx,
        edge_dst_idx=edge_dst_idx,
        pre_condition=pre_condition,
        is_switch=is_sw,
        chunk_mask=chunk_mask,
    )
    return build_msccl_xml(
        per_round_transfers=per_round,
        pre_condition=pre_condition,
        post_condition=post_condition,
        is_switch=is_sw,
        chunk_mask=chunk_mask,
        program_name=program_name,
        topology_mode=topology_mode,
        do_check=do_check,
        collective_type=collective_type,
        instances=instances,
        instr_fusion=instr_fusion,
    )


def strategy_dict_to_msccl_xml(
    payload: Dict[str, object],
    *,
    program_name: str = "aiccl_strategy",
    topology_mode: str = "from-transfers",
    do_check: bool = True,
    key_strategy: str = "real_strategy_list",
    key_pre: str = "pre_condition",
    key_src: str = "edge_src_idx",
    key_dst: str = "edge_dst_idx",
    key_switch: str = "is_switch",
    key_chunk_mask: str = "chunk_mask",
) -> str:
    """Convert a dict payload (e.g., torch.load result) into MSCCL XML."""

    if key_strategy not in payload:
        raise ValueError(f"Missing key {key_strategy!r}")
    strategy = payload[key_strategy]
    if not isinstance(strategy, (list, tuple)):
        raise ValueError(f"{key_strategy} must be a list/tuple of tensors")

    def _req_tensor(k: str) -> torch.Tensor:
        v = payload.get(k)
        if not torch.is_tensor(v):
            raise ValueError(f"Missing tensor key {k!r}")
        return v

    pre = _req_tensor(key_pre)
    src = _req_tensor(key_src)
    dst = _req_tensor(key_dst)
    is_sw = payload.get(key_switch)
    cm = payload.get(key_chunk_mask)
    return tensors_to_msccl_xml(
        real_strategy_list=[s for s in strategy],
        pre_condition=pre,
        edge_src_idx=src,
        edge_dst_idx=dst,
        is_switch=is_sw if torch.is_tensor(is_sw) else None,
        chunk_mask=cm if torch.is_tensor(cm) else None,
        program_name=program_name,
        topology_mode=topology_mode,
        do_check=do_check,
    )


def _xml_and_optional_check(do_check: bool) -> str:
    # XML() prints; we want string.
    # Use program.generate_xml() directly.
    from msccl.language import _curr  # type: ignore

    prog = _curr()
    xml = prog.generate_xml()
    if do_check:
        ok = Check()
        if ok is False:
            raise ValueError("MSCCLang Check() failed")
    return xml


def _load_from_torch(path: str) -> Dict[str, object]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    return {"_": obj}


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert real_strategy_list to MSCCL XML")
    ap.add_argument("--input", type=str, required=True, help="torch.load() file containing schedule dict")
    ap.add_argument("--out", type=str, default="", help="output XML path (default: stdout)")
    ap.add_argument("--program-name", type=str, default="aiccl_strategy", help="MSCCL algorithm name")
    ap.add_argument(
        "--topology-mode",
        type=str,
        default="fully_connected",
        choices=["fully_connected", "from-transfers"],
        help="MSCCLang topology used for link existence checks",
    )
    ap.add_argument("--no-check", action="store_true", help="skip MSCCLang Check()")

    # Optional key overrides
    ap.add_argument("--key-strategy", type=str, default="real_strategy_list")
    ap.add_argument("--key-pre", type=str, default="pre_condition")
    ap.add_argument("--key-src", type=str, default="edge_src_idx")
    ap.add_argument("--key-dst", type=str, default="edge_dst_idx")
    ap.add_argument("--key-switch", type=str, default="is_switch")
    ap.add_argument("--key-chunk-mask", type=str, default="chunk_mask")

    args = ap.parse_args()

    d = _load_from_torch(args.input)

    if args.key_strategy not in d:
        raise SystemExit(f"Missing key {args.key_strategy!r} in {args.input}")
    real_strategy_list = d[args.key_strategy]
    if not isinstance(real_strategy_list, (list, tuple)):
        raise SystemExit(f"{args.key_strategy} must be a list/tuple of tensors")

    def _get_tensor(key: str, required: bool = True) -> Optional[torch.Tensor]:
        if key in d and torch.is_tensor(d[key]):
            return d[key]
        if required:
            raise SystemExit(f"Missing tensor key {key!r} in {args.input}")
        return None

    pre = _get_tensor(args.key_pre, required=True)
    edge_src = _get_tensor(args.key_src, required=True)
    edge_dst = _get_tensor(args.key_dst, required=True)

    # infer V from pre
    if pre.dim() != 3:
        raise SystemExit(f"pre_condition must have shape (B,C,V), got {tuple(pre.shape)}")
    V = int(pre.shape[2])

    is_switch = _get_tensor(args.key_switch, required=False)
    is_switch = _to_bool(is_switch, V=V, device=pre.device)

    chunk_mask = _get_tensor(args.key_chunk_mask, required=False)

    # Normalize schedule tensors
    schedule: List[torch.Tensor] = []
    for i, s in enumerate(real_strategy_list):
        if not torch.is_tensor(s):
            raise SystemExit(f"real_strategy_list[{i}] is not a tensor")
        schedule.append(s)

    per_round = compute_transfers(
        real_strategy_list=schedule,
        edge_src_idx=edge_src,
        edge_dst_idx=edge_dst,
        pre_condition=pre,
        is_switch=is_switch,
        chunk_mask=chunk_mask,
    )

    xml = build_msccl_xml(
        per_round_transfers=per_round,
        pre_condition=pre,
        is_switch=is_switch,
        chunk_mask=chunk_mask,
        program_name=args.program_name,
        topology_mode=args.topology_mode,
        do_check=not args.no_check,
    )

    if args.out:
        Path(args.out).write_text(xml)
    else:
        sys.stdout.write(xml)


if __name__ == "__main__":
    main()
