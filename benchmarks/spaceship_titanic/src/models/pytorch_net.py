import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class PyTorchSpaceshipNet(nn.Module):
    """
    PyTorch Deep Tabular Network with Categorical Entity Embeddings,
    Dense Residual Blocks, BatchNorm, and Dropout for Spaceship Titanic Transported Classification.
    """
    def __init__(self, cat_dims: Dict[str, int], emb_dims: Dict[str, int], num_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.25):
        super(PyTorchSpaceshipNet, self).__init__()

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_classes, emb_dim)
            for col, (num_classes, emb_dim) in emb_dims.items()
        })

        total_emb_dim = sum(emb_dim for _, emb_dim in emb_dims.values())
        total_input_dim = total_emb_dim + num_dim

        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, cat_inputs: Dict[str, torch.Tensor], num_inputs: torch.Tensor) -> torch.Tensor:
        emb_outputs = []
        for col, emb_layer in self.embeddings.items():
            emb_outputs.append(emb_layer(cat_inputs[col]))

        cat_concat = torch.cat(emb_outputs, dim=1)
        x = torch.cat([cat_concat, num_inputs], dim=1)

        h = self.input_layer(x)
        h = F.silu(h + self.res_block1(h))
        logits = self.head(h)
        return logits.squeeze(-1)
