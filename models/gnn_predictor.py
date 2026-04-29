import torch
import torch.nn as nn
import torch.nn.functional as F
from models.egnn_layer import EGNNLayer
from models.multiscale_edges import MultiScaleEdgeBuilder
from models.attention_pool import AttentionPool

class GNNPredictor(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int = 47,
        hidden_dim: int = 512,
        n_layers: int = 4,
        edge_dim: int = 16,
        dropout: float = 0.05,
        use_protein_encoder: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_protein_encoder = use_protein_encoder

        # 1. Physics Path: Starting features for EGNN
        self.input_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 2. Local Physics Engine (EGNN)
        self.edge_builder = MultiScaleEdgeBuilder(edge_dim=edge_dim)
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(node_dim=hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])

        # 3. Identity Path: Residual skip connection of raw chemistry to Transformer
        self.identity_proj = nn.Linear(atom_feat_dim, hidden_dim)

        # 4. Contextual Path: ESM-2 Projection
        if use_protein_encoder:
            self.protein_proj = nn.Sequential(
                nn.Linear(1280, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim)
            )

        # 5. Global Reasoning: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 6. Output
        self.pool = AttentionPool(in_dim=hidden_dim, hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x, pos, batch, ligand_mask=None, protein_res_idx=None, protein_embs=None):
        # A. PRE-PROCESS
        x = torch.nan_to_num(x, nan=0.0)
        pos = torch.nan_to_num(pos, nan=0.0)

        # B. STAGE 1: LOCAL PHYSICS (GNN)
        h = self.input_proj(x)
        edge_index, edge_attr = self.edge_builder(pos, batch)
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        # C. STAGE 2: FEATURE FUSION (Identity + Context)
        # 1. Add back the "Fresh" Chemical Identity (Residual)
        h = h + self.identity_proj(x)

        # 2. Add the "Fresh" ESM-2 Protein context
        if self.use_protein_encoder and protein_embs is not None and ligand_mask is not None:
            prot_mask = ~ligand_mask
            unique_batches = torch.unique(batch)
            for i, b_id in enumerate(unique_batches):
                graph_prot_mask = prot_mask & (batch == b_id)
                if not graph_prot_mask.any() or i >= len(protein_embs): continue
                
                p_emb = self.protein_proj(protein_embs[i].to(h.device))
                indices = torch.clamp(protein_res_idx[graph_prot_mask], 0, p_emb.size(0)-1)
                h[graph_prot_mask] = h[graph_prot_mask] + p_emb[indices]

        # D. STAGE 3: GLOBAL REASONING (Transformer)
        unique_batches = torch.unique(batch)
        h_refined = torch.zeros_like(h)
        for b_id in unique_batches:
            mask = (batch == b_id)
            h_graph = h[mask].unsqueeze(0)
            h_trans = self.transformer(h_graph)
            h_refined[mask] = h_trans.squeeze(0)
        h = h_refined

        # E. STAGE 4: POOLING
        num_graphs = int(batch.max().item()) + 1
        if ligand_mask is not None:
            graph_emb = self.pool(h[ligand_mask], batch[ligand_mask], num_graphs=num_graphs)
        else:
            graph_emb = self.pool(h, batch, num_graphs=num_graphs)

        return torch.clamp(self.head(graph_emb), min=-20.0, max=20.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
