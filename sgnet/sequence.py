"""SGNet sequence-only baseline encoder."""
from __future__ import absolute_import
import torch
import torch.nn as nn
from torch_scatter import scatter


def _batch(data):
    value = getattr(data, "batch", None)
    if value is None:
        return torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
    return value.long()

def _num_graphs(batch):
    return int(batch.max().item()) + 1 if batch.numel() else 0

def _mean_max(hidden, batch, size):
    mean = scatter(hidden, batch, dim=0, dim_size=size, reduce="mean")
    maximum = scatter(hidden, batch, dim=0, dim_size=size, reduce="max")
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, maximum

def interleave_partner_residues(hidden_a, batch_a, hidden_b, batch_b):
    rows = []
    for graph_id in range(max(_num_graphs(batch_a), _num_graphs(batch_b))):
        rows.extend((hidden_a[batch_a == graph_id], hidden_b[batch_b == graph_id]))
    return torch.cat(rows, dim=0) if rows else hidden_a.new_zeros((0, hidden_a.shape[-1]))

class ResidueFFN(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(nn.Linear(channels, 2 * channels), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(2 * channels, channels))
    def forward(self, hidden):
        return hidden + self.net(self.norm(hidden))

class FrozenSequenceEncoder(nn.Module):
    """Frozen ESM2 branch with no edge, coordinate, or mask argument."""
    def __init__(self, in_channels=1280, hidden_channels=256,
                 head_channels=256, dropout=0.4):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(in_channels),
                                   nn.Linear(in_channels, hidden_channels), nn.SiLU())
        self.residue_ffn = ResidueFFN(hidden_channels, dropout)
        self.graph_encoder = nn.Sequential(
            nn.LayerNorm(5 * hidden_channels),
            nn.Linear(5 * hidden_channels, 2 * hidden_channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels))
        self.affinity_head = nn.Sequential(nn.Linear(hidden_channels, head_channels),
                                           nn.SiLU(), nn.Dropout(dropout),
                                           nn.Linear(head_channels, 1))
    def _encode(self, data):
        return self.residue_ffn(self.input(torch.nan_to_num(data.x.float()))), _batch(data)
    def forward_features(self, sequence_a, sequence_b):
        hidden_a, batch_a = self._encode(sequence_a)
        hidden_b, batch_b = self._encode(sequence_b)
        size = max(_num_graphs(batch_a), _num_graphs(batch_b))
        mean_a, max_a = _mean_max(hidden_a, batch_a, size)
        mean_b, max_b = _mean_max(hidden_b, batch_b, size)
        pair = torch.cat((mean_a + mean_b, torch.abs(mean_a - mean_b),
                          max_a + max_b, torch.abs(max_a - max_b), mean_a * mean_b), dim=-1)
        graph_hidden = self.graph_encoder(pair)
        residue_hidden = interleave_partner_residues(hidden_a, batch_a, hidden_b, batch_b)
        return self.affinity_head(graph_hidden), residue_hidden, graph_hidden
    def forward(self, sequence_a, sequence_b):
        return self.forward_features(sequence_a, sequence_b)[0]

