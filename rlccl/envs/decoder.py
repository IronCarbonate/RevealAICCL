"""Slot decoder for autoregressive schedule generation."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .problem import compute_received_chunks
from ..traffic.matrix_utils import validate_traffic_matrix


MOMENT_NODE_FEAT_DIM = 7
GLOBAL_MOMENT_FEAT_DIM = 8
CANDIDATE_MOMENT_FEAT_DIM = 4
MOMENT_Z_CLIP = 10.0


def _coefficient_of_variation(values, eps=1e-6):
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=0) / (abs(values.mean()) + eps))


def get_moment_node_features(moment_context, z_clip=MOMENT_Z_CLIP):
    """Build the fixed seven node moment features from a MomentContext."""
    send_mean = np.asarray(moment_context.send_mean, dtype=np.float64)
    recv_mean = np.asarray(moment_context.recv_mean, dtype=np.float64)
    send_std = np.asarray(moment_context.send_std, dtype=np.float64)
    recv_std = np.asarray(moment_context.recv_std, dtype=np.float64)
    mean_scale = max(float(np.mean(np.concatenate([send_mean, recv_mean]))), 1e-6)
    std_scale = max(float(np.mean(np.concatenate([send_std, recv_std]))), 1e-6)
    confidence = np.full_like(send_mean, float(moment_context.confidence))
    return np.stack(
        [
            send_mean / mean_scale,
            recv_mean / mean_scale,
            send_std / std_scale,
            recv_std / std_scale,
            np.clip(moment_context.current_send_z, -z_clip, z_clip) / z_clip,
            np.clip(moment_context.current_recv_z, -z_clip, z_clip) / z_clip,
            confidence,
        ],
        axis=1,
    ).astype(np.float32)


def get_candidate_moment_node_arrays(moment_context, z_clip=MOMENT_Z_CLIP):
    """Return node-level arrays sufficient to replay candidate moment features."""
    send_mean = np.asarray(moment_context.send_mean, dtype=np.float64)
    recv_mean = np.asarray(moment_context.recv_mean, dtype=np.float64)
    mean_scale = max(float(np.mean(np.concatenate([send_mean, recv_mean]))), 1e-6)
    return {
        "source_current_send_z": (
            np.clip(moment_context.current_send_z, -z_clip, z_clip) / z_clip
        ).astype(np.float32),
        "destination_current_recv_z": (
            np.clip(moment_context.current_recv_z, -z_clip, z_clip) / z_clip
        ).astype(np.float32),
        "source_expected_send": (send_mean / mean_scale).astype(np.float32),
        "destination_expected_recv": (recv_mean / mean_scale).astype(np.float32),
    }


def get_global_moment_features(moment_context, current_matrix=None, max_entry=8.0):
    """Build eight global moment/current-traffic features without mutating history."""
    if current_matrix is not None:
        validate_traffic_matrix(current_matrix)
        matrix = np.asarray(current_matrix, dtype=np.float64)
        num_nodes = matrix.shape[0]
        off_diagonal = matrix[~np.eye(num_nodes, dtype=bool)]
        sparsity = float(np.mean(off_diagonal == 0)) if off_diagonal.size else 1.0
        source_load = matrix.sum(axis=1)
        destination_load = matrix.sum(axis=0)
        max_value = float(matrix.max()) if matrix.size else 0.0
    else:
        # Current aggregate loads can be reconstructed from history mean/std and
        # z-scores; pair sparsity/max-entry remain unavailable and use zero.
        source_load = np.asarray(moment_context.send_mean) + (
            np.asarray(moment_context.current_send_z) * np.asarray(moment_context.send_std)
        )
        destination_load = np.asarray(moment_context.recv_mean) + (
            np.asarray(moment_context.current_recv_z) * np.asarray(moment_context.recv_std)
        )
        sparsity = 0.0
        max_value = 0.0
    return np.asarray(
        [
            np.clip(moment_context.mean_drift, 0.0, 10.0),
            np.clip(moment_context.var_drift, 0.0, 10.0),
            np.clip(moment_context.confidence, 0.0, 1.0),
            np.clip(
                moment_context.history_length / max(moment_context.window_size, 1),
                0.0,
                1.0,
            ),
            sparsity,
            _coefficient_of_variation(source_load),
            _coefficient_of_variation(destination_load),
            max_value / max(float(max_entry), 1e-6),
        ],
        dtype=np.float32,
    )


def get_candidate_moment_features(cand_e, edge_src, edge_dst, node_arrays):
    """Build four candidate features from compact node-level replay arrays."""
    candidate_src = edge_src[cand_e]
    candidate_dst = edge_dst[cand_e]
    return np.stack(
        [
            node_arrays["source_current_send_z"][candidate_src],
            node_arrays["destination_current_recv_z"][candidate_dst],
            node_arrays["source_expected_send"][candidate_src],
            node_arrays["destination_expected_recv"][candidate_dst],
        ],
        axis=1,
    ).astype(np.float32)

class SlotDecoder:
    """
    Slot Decoder: Autoregressive construction of Y_t.
    """
    def __init__(self, topology_info):
        self.topo = topology_info
        self.V = topology_info.V
        self.E = topology_info.E
        
        self.edge_src = np.asarray(topology_info.edge_src, dtype=np.int64)
        self.edge_dst = np.asarray(topology_info.edge_dst, dtype=np.int64)
        self.capacities = np.asarray(topology_info.capacities, dtype=np.float32)
        
        if hasattr(self.topo, 'dist_matrix') and self.topo.dist_matrix is not None:
            self.dist_matrix = np.asarray(self.topo.dist_matrix, dtype=np.float32)
        else:
            # Simple fallback BFS
            inf = 1e9
            dist = np.full((self.V, self.V), inf, dtype=np.float32)
            np.fill_diagonal(dist, 0)
            for i in range(self.E):
                u, v = self.edge_src[i], self.edge_dst[i]
                dist[u, v] = 1
            for k in range(self.V):
                for i in range(self.V):
                    for j in range(self.V):
                        dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
            self.dist_matrix = dist
        
        self.group_to_edges = []
        self.edge_to_groups = [[] for _ in range(self.E)]
        self.group_limits = []
        
        if hasattr(self.topo, 'shared_constraints') and self.topo.shared_constraints:
            for g_idx, (edges, limit) in enumerate(self.topo.shared_constraints):
                self.group_to_edges.append(np.asarray(edges, dtype=np.int64))
                self.group_limits.append(float(limit))
                for e in edges:
                    self.edge_to_groups[e].append(g_idx)
        
        self.group_limits = np.array(self.group_limits, dtype=np.float32) if self.group_limits else np.array([1.0], dtype=np.float32)
        self.num_groups = len(self.group_limits) if len(self.group_to_edges) > 0 else 0
        
        self.out_deg = np.zeros(self.V, dtype=np.float32)
        self.in_deg = np.zeros(self.V, dtype=np.float32)
        np.add.at(self.out_deg, self.edge_src, 1)
        np.add.at(self.in_deg, self.edge_dst, 1)
        self.f_out = self.out_deg / max(1, self.E)
        self.f_in = self.in_deg / max(1, self.E)
        
        self.max_steps = int(self.capacities.sum()) + 5
        self.edge_src_t = torch.tensor(self.edge_src, dtype=torch.long)
        self.edge_dst_t = torch.tensor(self.edge_dst, dtype=torch.long)
        
        # Static info shared across all slots (stored once, not per-slot)
        # This saves O(batch_target * E) memory in the buffer
        self._static_info_cpu = {
            'edge_src_t': self.edge_src_t.clone(),
            'edge_dst_t': self.edge_dst_t.clone(),
            'capacities': self.capacities.copy(),
            'max_steps': self.max_steps,
            'group_limits': self.group_limits.copy(),
            'edge_to_groups': self.edge_to_groups,  # list of lists, shared reference is fine
            'num_groups': self.num_groups,
            'V': self.V,
            'E': self.E,
        }

    def get_static_info(self):
        """Return static topology info (shared across all slots)."""
        return self._static_info_cpu

    def compute_dist_to_demand(self, demands):
        C, V = demands.shape
        inf_dist = V + 1.0
        dist_to_demand = np.full((C, V), inf_dist, dtype=np.float32)
        for v in range(V):
            cs = np.where(demands[:, v])[0]
            if len(cs) > 0:
                dists = self.dist_matrix[:, v]
                dist_to_demand[cs, :] = np.minimum(dist_to_demand[cs, :], dists)
        return dist_to_demand

    def get_node_features(self, state, demands, t, T):
        C, V = state.shape
        f_held = state.sum(axis=0) / max(1, C)
        f_need = demands.sum(axis=0) / max(1, C)
        f_time = np.full(V, t / max(1, T), dtype=np.float32)
        return np.stack([f_held, f_need, self.f_out, self.f_in, f_time], axis=1).astype(np.float32)

    def get_chunk_features(self, state, demands):
        C, V = state.shape
        holders = state.sum(axis=1)
        f_rarity = 1.0 - (holders / max(1, V))
        demanders = demands.sum(axis=1)
        f_urgency = demanders / max(1, V)
        return np.stack([f_rarity, f_urgency], axis=1).astype(np.float32)

    def get_edge_features(self, edge_usage):
        f_cap_rem = (self.capacities - edge_usage) / np.maximum(self.capacities, 1e-6)
        f_cap_static = self.capacities / max(np.max(self.capacities), 1e-6)
        return np.stack([f_cap_rem, f_cap_static], axis=1).astype(np.float32)

    def get_candidate_dynamic_features(self, cand_c, cand_e, demands, dist_to_demand,
                                        edge_usage, group_usage, step, max_steps):
        cand_src = self.edge_src[cand_e]
        cand_dst = self.edge_dst[cand_e]
        
        f_is_demand = demands[cand_c, cand_dst].astype(np.float32)
        d_src = dist_to_demand[cand_c, cand_src]
        d_dst = dist_to_demand[cand_c, cand_dst]
        f_dist_red = (d_src - d_dst) / max(1.0, self.V)
        
        edge_rem = self.capacities[cand_e] - edge_usage[cand_e]
        f_edge_rem = edge_rem / np.maximum(self.capacities[cand_e], 1e-6)
        
        f_group_rem = np.ones(len(cand_e), dtype=np.float32)
        if self.num_groups > 0:
            for i, e in enumerate(cand_e):
                groups = self.edge_to_groups[e]
                if groups:
                    min_rem = 1.0
                    for g in groups:
                        rem = (self.group_limits[g] - group_usage[g]) / max(self.group_limits[g], 1e-6)
                        min_rem = min(min_rem, rem)
                    f_group_rem[i] = min_rem
        
        f_step_progress = np.full(len(cand_e), step / max(max_steps, 1), dtype=np.float32)
        
        return np.stack([f_is_demand, f_dist_red, f_edge_rem, f_group_rem, f_step_progress], axis=1).astype(np.float32)

    def decode_slot(
        self,
        model,
        state,
        demands,
        t,
        T,
        train=True,
        moment_context=None,
        current_matrix=None,
        moment_max_entry=8.0,
        observation_demands=None,
    ):
        device = next(model.parameters()).device
        C, V = state.shape
        E = self.E

        # ``demands`` remains the authoritative ground-truth execution state.
        # Phase C partial-observation experiments may provide a separate policy
        # view for feature construction.  The default path is byte-for-byte the
        # existing full-demand behavior.
        policy_demands = (
            demands
            if observation_demands is None
            else np.asarray(observation_demands, dtype=np.float32)
        )
        if policy_demands.shape != demands.shape:
            raise ValueError(
                "observation_demands must have the same chunk/node shape as demands"
            )
        if not np.all(np.isfinite(policy_demands)) or np.any(policy_demands < 0):
            raise ValueError("observation_demands must be finite and nonnegative")
        
        if self.edge_src_t.device != device:
            self.edge_src_t = self.edge_src_t.to(device)
            self.edge_dst_t = self.edge_dst_t.to(device)
        
        # 1. Encode slot-level state
        dist_to_demand = self.compute_dist_to_demand(policy_demands)
        node_feats_np = self.get_node_features(state, policy_demands, t, T)
        chunk_feats_np = self.get_chunk_features(state, policy_demands)
        edge_feats_np = self.get_edge_features(np.zeros(E, dtype=np.float32))

        moment_enabled = getattr(model, "global_moment_feat_dim", 0) > 0
        global_moment_feats_np = None
        candidate_moment_node_arrays = None
        node_moment_feats_np = None
        if moment_enabled:
            if moment_context is None:
                node_moment_feats_np = np.zeros(
                    (V, MOMENT_NODE_FEAT_DIM), dtype=np.float32
                )
                global_moment_feats_np = np.zeros(
                    GLOBAL_MOMENT_FEAT_DIM, dtype=np.float32
                )
                candidate_moment_node_arrays = {
                    name: np.zeros(V, dtype=np.float32)
                    for name in (
                        "source_current_send_z",
                        "destination_current_recv_z",
                        "source_expected_send",
                        "destination_expected_recv",
                    )
                }
            else:
                node_moment_feats_np = get_moment_node_features(moment_context)
                global_moment_feats_np = get_global_moment_features(
                    moment_context, current_matrix, moment_max_entry
                )
                candidate_moment_node_arrays = get_candidate_moment_node_arrays(
                    moment_context
                )
            node_feats_np = np.concatenate(
                [node_feats_np, node_moment_feats_np], axis=1
            )

        expected_node_dim = getattr(model, "node_feat_dim", node_feats_np.shape[1])
        if node_feats_np.shape[1] != expected_node_dim:
            raise ValueError(
                f"Decoder produced {node_feats_np.shape[1]} node features, "
                f"model expects {expected_node_dim}"
            )
        
        node_feats = torch.tensor(node_feats_np, dtype=torch.float32, device=device)
        chunk_feats = torch.tensor(chunk_feats_np, dtype=torch.float32, device=device)
        edge_feats = torch.tensor(edge_feats_np, dtype=torch.float32, device=device)
        global_moment_feats = (
            torch.tensor(global_moment_feats_np, dtype=torch.float32, device=device)
            if global_moment_feats_np is not None
            else None
        )
        
        with torch.no_grad():
            h_v, h_e, h_c, g_ctx = model.encode_state(
                node_feats,
                edge_feats,
                self.edge_src_t,
                self.edge_dst_t,
                chunk_feats,
                global_moment_feats=global_moment_feats,
            )
            value = model.get_value(g_ctx)
        
        # 2. Initialize candidates
        src_has = state[:, self.edge_src]
        dst_lacks = 1 - state[:, self.edge_dst]
        valid_matrix = (src_has == 1) & (dst_lacks == 1)
        all_c_idxs, all_e_idxs = np.where(valid_matrix)
        
        # state_info: only store DYNAMIC per-slot data
        # Static topology info is accessed via decoder.get_static_info()
        state_info = {
            'node_feats': node_feats.cpu(),
            'edge_feats': edge_feats.cpu(),
            'chunk_feats': chunk_feats.cpu(),
            'demands': policy_demands.copy(),
            'dist_to_demand': dist_to_demand.copy(),
            'moment_enabled': moment_enabled,
            'global_moment_feats': (
                global_moment_feats.cpu() if global_moment_feats is not None else None
            ),
            'node_moment_feats': (
                node_moment_feats_np.copy() if node_moment_feats_np is not None else None
            ),
            'candidate_moment_node_arrays': (
                {key: value.copy() for key, value in candidate_moment_node_arrays.items()}
                if candidate_moment_node_arrays is not None
                else None
            ),
        }
        
        if len(all_c_idxs) == 0:
            Y_t = np.zeros((C, E), dtype=int)
            logp_slot = torch.tensor(0.0, device=device)
            entropy_slot = torch.tensor(0.0, device=device)
            return Y_t, logp_slot, entropy_slot, value, state_info, []
        
        Y_t = np.zeros((C, E), dtype=int)
        edge_usage = np.zeros(E, dtype=np.float32)
        group_usage = np.zeros(self.num_groups, dtype=np.float32) if self.num_groups > 0 else np.array([], dtype=np.float32)
        received_mask = np.zeros((V, C), dtype=bool)
        active_mask = np.ones(len(all_c_idxs), dtype=bool)
        cand_dst_all = self.edge_dst[all_e_idxs]
        
        # 3. Autoregressive decoding
        micro_actions = []
        logp_list = []
        entropy_list = []
        
        for step in range(self.max_steps):
            mask_edge = (edge_usage[all_e_idxs] < self.capacities[all_e_idxs])
            mask_recv = ~received_mask[cand_dst_all, all_c_idxs]
            mask_group = np.ones(len(all_c_idxs), dtype=bool)
            
            if self.num_groups > 0:
                full_groups = group_usage >= self.group_limits
                if np.any(full_groups):
                    blocked_edges = np.zeros(E, dtype=bool)
                    for g in np.where(full_groups)[0]:
                        blocked_edges[self.group_to_edges[g]] = True
                    mask_group = ~blocked_edges[all_e_idxs]
            
            current_mask = active_mask & mask_edge & mask_recv & mask_group
            if not np.any(current_mask):
                break
            
            active_indices = np.where(current_mask)[0]
            
            # Pruning
            max_cands = 256 if train else 512
            if len(active_indices) > max_cands:
                active_c = all_c_idxs[active_indices]
                active_e = all_e_idxs[active_indices]
                cand_src = self.edge_src[active_e]
                cand_dst = self.edge_dst[active_e]
                
                is_demand_score = demands[active_c, cand_dst].astype(np.float32) * 10.0
                d_src = dist_to_demand[active_c, cand_src]
                d_dst = dist_to_demand[active_c, cand_dst]
                dist_red_score = (d_src - d_dst) / max(1.0, self.V)
                heuristic_scores = is_demand_score + dist_red_score
                
                top_k_local = np.argsort(heuristic_scores)[-max_cands:]
                active_indices = active_indices[top_k_local]
            
            cand_c = all_c_idxs[active_indices]
            cand_e = all_e_idxs[active_indices]
            
            cand_dyn_feats_np = self.get_candidate_dynamic_features(
                cand_c, cand_e, policy_demands, dist_to_demand,
                edge_usage, group_usage, step, self.max_steps
            )
            if moment_enabled:
                candidate_moment_feats = get_candidate_moment_features(
                    cand_e,
                    self.edge_src,
                    self.edge_dst,
                    candidate_moment_node_arrays,
                )
                cand_dyn_feats_np = np.concatenate(
                    [cand_dyn_feats_np, candidate_moment_feats], axis=1
                )

            expected_candidate_dim = getattr(
                model, "cand_feat_dim", cand_dyn_feats_np.shape[1]
            )
            if cand_dyn_feats_np.shape[1] != expected_candidate_dim:
                raise ValueError(
                    f"Decoder produced {cand_dyn_feats_np.shape[1]} candidate features, "
                    f"model expects {expected_candidate_dim}"
                )
            
            with torch.no_grad():
                cand_dyn_feats = torch.tensor(cand_dyn_feats_np, dtype=torch.float32, device=device)
                cand_e_t = torch.tensor(cand_e, dtype=torch.long, device=device)
                cand_c_t = torch.tensor(cand_c, dtype=torch.long, device=device)
                
                logits = model.get_candidate_logits(
                    h_v, h_e, h_c, g_ctx, self.edge_src_t, self.edge_dst_t,
                    cand_e_t, cand_c_t, cand_dyn_feats
                )
                
                probs = F.softmax(logits, dim=0)
                dist = Categorical(probs)
                
                if train:
                    action_idx = dist.sample()
                else:
                    action_idx = torch.argmax(logits)
                    
                log_prob = dist.log_prob(action_idx)
                entropy = dist.entropy()
            
            logp_list.append(log_prob)
            entropy_list.append(entropy)
            
            idx = action_idx.item()
            best_global_idx = active_indices[idx]
            c = all_c_idxs[best_global_idx]
            e = all_e_idxs[best_global_idx]
            v = cand_dst_all[best_global_idx]
            
            micro_actions.append({
                'action_idx': idx,
                'cand_c': cand_c.copy(),
                'cand_e': cand_e.copy(),
                'selected_e': e,
                'step': step,
            })
            
            Y_t[c, e] = 1
            active_mask[best_global_idx] = False
            edge_usage[e] += 1
            for g in self.edge_to_groups[e]:
                group_usage[g] += 1
            received_mask[v, c] = True
        
        if len(logp_list) > 0:
            logp_slot = torch.stack(logp_list).sum()
            entropy_slot = torch.stack(entropy_list).mean()
        else:
            logp_slot = torch.tensor(0.0, device=device)
            entropy_slot = torch.tensor(0.0, device=device)
        
        return Y_t, logp_slot, entropy_slot, value, state_info, micro_actions


def recompute_logp_slot(model, state_info, micro_actions, device, static_info):
    """
    Batched recomputation of logp_slot and entropy.
    Constructs a single large batch of all candidates across all micro-steps
    to avoid sequential NN forward passes.
    
    Args:
        model: The policy model
        state_info: Dynamic per-slot state (node_feats, edge_feats, chunk_feats, demands, dist_to_demand)
        micro_actions: List of micro-action dicts from decode_slot
        device: torch device
        static_info: Static topology info from decoder.get_static_info()
    """
    # Recover dynamic state info
    node_feats = state_info['node_feats'].to(device)
    edge_feats = state_info['edge_feats'].to(device)
    chunk_feats = state_info['chunk_feats'].to(device)
    demands = state_info['demands']
    dist_to_demand = state_info['dist_to_demand']
    moment_enabled = bool(state_info.get('moment_enabled', False))
    global_moment_feats = state_info.get('global_moment_feats')
    if global_moment_feats is not None:
        global_moment_feats = global_moment_feats.to(device)
    candidate_moment_node_arrays = state_info.get(
        'candidate_moment_node_arrays'
    )
    
    # Recover static topology info (shared across all slots)
    edge_src_t = static_info['edge_src_t'].to(device)
    edge_dst_t = static_info['edge_dst_t'].to(device)
    V = static_info['V']
    E = static_info['E']
    max_steps = static_info['max_steps']
    capacities = static_info['capacities']
    group_limits = static_info['group_limits']
    edge_to_groups = static_info['edge_to_groups']
    num_groups = static_info['num_groups']
    
    # 1. GNN Encoding (Once per slot)
    h_v, h_e, h_c, g_ctx = model.encode_state(
        node_feats,
        edge_feats,
        edge_src_t,
        edge_dst_t,
        chunk_feats,
        global_moment_feats=global_moment_feats,
    )
    value_new = model.get_value(g_ctx)
    
    if len(micro_actions) == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), value_new
    
    # 2. Batched Feature Generation (CPU replay)
    edge_usage = np.zeros(E, dtype=np.float32)
    group_usage = np.zeros(num_groups, dtype=np.float32) if num_groups > 0 else np.array([], dtype=np.float32)
    
    all_cand_e = []
    all_cand_c = []
    all_dyn_feats = []
    segment_sizes = []
    actions_taken = []
    
    # Replay loop to collect all inputs
    # Use numpy for feature calculation (fast)
    edge_src_np = edge_src_t.cpu().numpy()
    edge_dst_np = edge_dst_t.cpu().numpy()
    
    for ma in micro_actions:
        cand_c = ma['cand_c']
        cand_e = ma['cand_e']
        selected_e = ma['selected_e']
        step = ma['step']
        
        # Recalculate dynamic features using current usage
        cand_src = edge_src_np[cand_e]
        cand_dst = edge_dst_np[cand_e]
        
        f_is_demand = demands[cand_c, cand_dst].astype(np.float32)
        d_src = dist_to_demand[cand_c, cand_src]
        d_dst = dist_to_demand[cand_c, cand_dst]
        f_dist_red = (d_src - d_dst) / max(1.0, V)
        
        edge_rem = capacities[cand_e] - edge_usage[cand_e]
        f_edge_rem = edge_rem / np.maximum(capacities[cand_e], 1e-6)
        
        f_group_rem = np.ones(len(cand_e), dtype=np.float32)
        if num_groups > 0:
            for i, e in enumerate(cand_e):
                groups = edge_to_groups[e]
                if groups:
                    min_rem = 1.0
                    for g in groups:
                        rem = (group_limits[g] - group_usage[g]) / max(group_limits[g], 1e-6)
                        min_rem = min(min_rem, rem)
                    f_group_rem[i] = min_rem
        
        f_step_progress = np.full(len(cand_e), step / max(max_steps, 1), dtype=np.float32)
        cand_dyn_feats_np = np.stack([f_is_demand, f_dist_red, f_edge_rem, f_group_rem, f_step_progress], axis=1).astype(np.float32)
        if moment_enabled:
            if candidate_moment_node_arrays is None:
                raise ValueError("Missing candidate moment replay arrays")
            candidate_moment_feats = get_candidate_moment_features(
                cand_e,
                edge_src_np,
                edge_dst_np,
                candidate_moment_node_arrays,
            )
            cand_dyn_feats_np = np.concatenate(
                [cand_dyn_feats_np, candidate_moment_feats], axis=1
            )
        
        all_cand_e.append(cand_e)
        all_cand_c.append(cand_c)
        all_dyn_feats.append(cand_dyn_feats_np)
        segment_sizes.append(len(cand_e))
        actions_taken.append(ma['action_idx'])
        
        # Update usage for next step
        edge_usage[selected_e] += 1
        for g in edge_to_groups[selected_e]:
            group_usage[g] += 1
            
    # 3. Batched NN Inference
    flat_cand_e = torch.tensor(np.concatenate(all_cand_e), dtype=torch.long, device=device)
    flat_cand_c = torch.tensor(np.concatenate(all_cand_c), dtype=torch.long, device=device)
    flat_dyn_feats = torch.tensor(np.concatenate(all_dyn_feats), dtype=torch.float32, device=device)
    
    # Single forward pass for all candidates in the slot
    # get_candidate_logits expands g_ctx to match flat_cand_e length
    all_logits = model.get_candidate_logits(
        h_v, h_e, h_c, g_ctx, edge_src_t, edge_dst_t,
        flat_cand_e, flat_cand_c, flat_dyn_feats
    )
    
    # 4. Compute Logprobs and Entropy
    logp_list = []
    entropy_list = []
    current_offset = 0
    
    # Iterate over segments (fast CPU loop over tensor slices)
    for i, size in enumerate(segment_sizes):
        segment_logits = all_logits[current_offset : current_offset + size]
        current_offset += size
        
        probs = F.softmax(segment_logits, dim=0)
        dist = Categorical(probs)
        
        # The action index is relative to the segment
        action_t = torch.tensor(actions_taken[i], dtype=torch.long, device=device)
        logp_list.append(dist.log_prob(action_t))
        entropy_list.append(dist.entropy())
        
    logp_slot_new = torch.stack(logp_list).sum()
    entropy_slot_new = torch.stack(entropy_list).mean()
    
    return logp_slot_new, entropy_slot_new, value_new


def train(train_problems, topology_info_dict):
    """
    Slot-level PPO Training.
    """
    model = SlotLevelPolicy(node_feat_dim=5, edge_feat_dim=2, cand_feat_dim=5, 
                            chunk_feat_dim=2, hidden_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    slot_buffer = SlotBuffer()
    
    epochs = 4
    ppo_epochs = 3
    clip_eps = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_coef = 0.5
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_problems))
        
        slot_buffer.clear()
        slots_collected = 0
        batch_target = 200
        
        for idx in indices:
            if slots_collected >= batch_target:
                break
            
            scenario_id, problem = train_problems[idx]
            if np.sum(problem.demands) == 0:
                continue
            
            topo_info = getattr(problem, 'topology_info', None)
            if topo_info is None:
                continue
            
            decoder = SlotDecoder(topo_info)
            state = problem.initial_state.copy()
            demands = problem.demands.copy()
            initial_total_demands = max(1.0, float(np.sum(problem.demands)))
            
            # Get static info once per problem
            static_info = decoder.get_static_info()
            
            for t in range(problem.T):
                Y_t, logp_slot, entropy_slot, value, state_info, micro_actions = decoder.decode_slot(
                    model, state, demands, t, problem.T, train=True
                )
                
                N_t = compute_received_chunks(Y_t, topo_info.edge_dst, topo_info.V)
                state = np.maximum(state, N_t)
                demands = demands * (1 - N_t)
                
                remaining = float(np.sum(demands))
                slot_reward = -remaining / initial_total_demands
                
                episode_success = (np.sum(demands) == 0)
                episode_timeout = (t == problem.T - 1)
                episode_end = episode_success or episode_timeout
                
                slot_buffer.add(
                    state_info=state_info,
                    actions=micro_actions,
                    logprob_slot=logp_slot.detach().cpu(),
                    value=value.detach().cpu(),
                    reward=slot_reward,
                    done=episode_end,
                    static_info=static_info
                )
                
                slots_collected += 1
                if episode_end:
                    break
        
        if len(slot_buffer) == 0:
            continue
        
        # PPO Update
        rewards = torch.tensor(slot_buffer.slot_rewards, dtype=torch.float32)
        values = torch.tensor([v.item() for v in slot_buffer.slot_values], dtype=torch.float32)
        dones = torch.tensor(slot_buffer.slot_dones, dtype=torch.float32)
        
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0.0
                next_val = 0.0
            else:
                next_val = values[t + 1].item() if t + 1 < len(values) else 0.0
            
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        buffer_size = len(slot_buffer)
        indices_ppo = np.arange(buffer_size)
        
        for _ in range(ppo_epochs):
            np.random.shuffle(indices_ppo)
            mb_size = 16
            
            for start in range(0, buffer_size, mb_size):
                end = min(start + mb_size, buffer_size)
                batch_indices = indices_ppo[start:end]
                
                optimizer.zero_grad()
                loss_accum = 0.0
                
                for idx_sample in batch_indices:
                    state_info = slot_buffer.slot_states[idx_sample]
                    micro_actions = slot_buffer.slot_actions[idx_sample]
                    static_info = slot_buffer.slot_static_infos[idx_sample]
                    old_logp = slot_buffer.slot_logprobs[idx_sample].to(device)
                    advantage = advantages[idx_sample].to(device)
                    ret = returns[idx_sample].to(device)
                    
                    logp_new, entropy_new, value_new = recompute_logp_slot(
                        model, state_info, micro_actions, device, static_info
                    )
                    
                    ratio = torch.exp(logp_new - old_logp)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    
                    value_loss = F.mse_loss(value_new.squeeze(), ret)
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_new
                    loss_accum += loss
                
                loss_accum = loss_accum / len(batch_indices)
                loss_accum.backward()
                optimizer.step()
    
    return model




def solve_with_model(problem, model, topology_info):
    """
    Inference function.
    """
    model.eval()
    topo = getattr(problem, 'topology_info', topology_info)
    decoder = SlotDecoder(topo)
    
    state = problem.initial_state.copy()
    demands = problem.demands.copy()
    schedule = []
    
    with torch.no_grad():
        for t in range(problem.T):
            Y_t, _, _, _, _, _ = decoder.decode_slot(
                model, state, demands, t, problem.T, train=False
            )
            schedule.append(Y_t)
            
            N_t = compute_received_chunks(Y_t, topo.edge_dst, topo.V)
            state = np.maximum(state, N_t)
            demands = demands * (1 - N_t)
            
            if np.sum(demands) == 0:
                break
    
    while len(schedule) < problem.T:
        schedule.append(np.zeros((problem.C, problem.E), dtype=int))
    
    return schedule
