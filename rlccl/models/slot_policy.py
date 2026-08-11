"""Slot-level policy network for collective communication scheduling."""

import torch
import torch.nn as nn

from .gnn_layers import ECDUGNNLayer
from .moment_encoder import MomentEncoder


class SlotLevelPolicy(nn.Module):
    """Slot-level Policy Network for collective communication scheduling.
    
    This network encodes the topology state and outputs:
    1. Actor logits for selecting (chunk, edge) pairs
    2. Critic value estimation for PPO training
    
    Args:
        node_feat_dim: Dimension of node features (default: 5)
        edge_feat_dim: Dimension of edge features (default: 2)
        cand_feat_dim: Dimension of candidate dynamic features (default: 5)
        chunk_feat_dim: Dimension of chunk features (default: 2)
        hidden_dim: Hidden dimension for all layers (default: 64)
    """
    
    def __init__(
        self, 
        node_feat_dim: int = 5, 
        edge_feat_dim: int = 2, 
        cand_feat_dim: int = 5, 
        chunk_feat_dim: int = 2, 
        hidden_dim: int = 64,
        global_moment_feat_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_feat_dim = int(node_feat_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.cand_feat_dim = int(cand_feat_dim)
        self.chunk_feat_dim = int(chunk_feat_dim)
        self.global_moment_feat_dim = int(global_moment_feat_dim)
        
        # Encoders
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # GNN layers
        self.layer1 = ECDUGNNLayer(hidden_dim, hidden_dim, hidden_dim)
        self.layer2 = ECDUGNNLayer(hidden_dim, hidden_dim, hidden_dim)
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # These modules do not exist in baseline mode, preserving old state-dict
        # keys and exact 5/2/5/2 model construction behavior.
        if self.global_moment_feat_dim > 0:
            self.moment_encoder = MomentEncoder(
                self.global_moment_feat_dim, hidden_dim
            )
            self.context_fusion = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LeakyReLU(0.1),
            )
        
        # Actor: scores candidate (c, e) pairs
        self.actor = nn.Sequential(
            nn.Linear(4 * hidden_dim + cand_feat_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Critic: V(s_t) at slot level
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def encode_state(
        self,
        node_feats,
        edge_feats,
        edge_src,
        edge_dst,
        chunk_feats,
        global_moment_feats=None,
    ):
        """Encode state and return embeddings for actor/critic.
        
        Args:
            node_feats: Node features, shape (V, node_feat_dim)
            edge_feats: Edge features, shape (E, edge_feat_dim)
            edge_src: Source node indices, shape (E,)
            edge_dst: Destination node indices, shape (E,)
            chunk_feats: Chunk features, shape (C, chunk_feat_dim)
            
        Returns:
            h_v: Node embeddings, shape (V, hidden_dim)
            h_e: Edge embeddings, shape (E, hidden_dim)
            h_c: Chunk embeddings, shape (C, hidden_dim)
            g_ctx: Global context, shape (hidden_dim,)
        """
        num_nodes = node_feats.size(0)
        h_v = self.node_encoder(node_feats)
        h_e = self.edge_encoder(edge_feats)
        
        h_v, h_e = self.layer1(h_v, h_e, edge_src, edge_dst, num_nodes)
        h_v, h_e = self.layer2(h_v, h_e, edge_src, edge_dst, num_nodes)
        
        graph_ctx = self.global_pool(h_v.mean(dim=0))
        if self.global_moment_feat_dim > 0:
            if global_moment_feats is None:
                global_moment_feats = torch.zeros(
                    self.global_moment_feat_dim,
                    dtype=node_feats.dtype,
                    device=node_feats.device,
                )
            moment_ctx = self.moment_encoder(global_moment_feats)
            g_ctx = self.context_fusion(torch.cat([graph_ctx, moment_ctx], dim=-1))
        else:
            g_ctx = graph_ctx
        h_c = self.chunk_encoder(chunk_feats)
        
        return h_v, h_e, h_c, g_ctx

    def get_value(self, g_ctx):
        """Critic: V(s_t) for slot-level value estimation.
        
        Args:
            g_ctx: Global context, shape (hidden_dim,)
            
        Returns:
            Value estimate, shape (1,)
        """
        return self.critic(g_ctx)

    def get_candidate_logits(
        self, 
        h_v, 
        h_e, 
        h_c, 
        g_ctx, 
        edge_src, 
        edge_dst, 
        cand_e, 
        cand_c, 
        cand_dynamic_feats
    ):
        """Actor: compute logits for candidate (c, e) pairs.
        
        Args:
            h_v: Node embeddings, shape (V, hidden_dim)
            h_e: Edge embeddings, shape (E, hidden_dim)
            h_c: Chunk embeddings, shape (C, hidden_dim)
            g_ctx: Global context, shape (hidden_dim,)
            edge_src: Source node indices, shape (E,)
            edge_dst: Destination node indices, shape (E,)
            cand_e: Candidate edge indices, shape (N_cand,)
            cand_c: Candidate chunk indices, shape (N_cand,)
            cand_dynamic_feats: Dynamic features for candidates, shape (N_cand, cand_feat_dim)
            
        Returns:
            Logits for candidates, shape (N_cand,)
        """
        cand_edge_emb = h_e[cand_e]
        cand_src_emb = h_v[edge_src[cand_e]]
        cand_dst_emb = h_v[edge_dst[cand_e]]
        cand_chunk_emb = h_c[cand_c]
        
        # Expand global context to match number of candidates
        g_ctx_expanded = g_ctx.unsqueeze(0).expand(len(cand_e), -1)
        
        actor_input = torch.cat([
            cand_edge_emb, cand_src_emb, cand_dst_emb, 
            cand_chunk_emb, cand_dynamic_feats, g_ctx_expanded
        ], dim=-1)
        
        return self.actor(actor_input).squeeze(-1)
