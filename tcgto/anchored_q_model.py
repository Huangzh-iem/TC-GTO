from __future__ import annotations
import copy
import torch
from torch import nn

class AnchoredQModel(nn.Module):
    """R1 operator with an explicitly anchored displacement parameterization."""
    def __init__(self, base: nn.Module, anchor_index: int = 144):
        super().__init__()
        self.base = base
        self.anchor_index = int(anchor_index)
        self.q0_head = self._clone_q_path(base.response_head)
        self.delta_q_head = self._clone_q_path(base.response_head)

    @staticmethod
    def _clone_q_path(source: nn.Sequential) -> nn.Sequential:
        head = copy.deepcopy(source)
        final = source[-1]
        replacement = nn.Linear(final.in_features, 1, bias=final.bias is not None)
        with torch.no_grad():
            replacement.weight.copy_(final.weight[0:1])
            if final.bias is not None:
                replacement.bias.copy_(final.bias[0:1])
        head[-1] = replacement
        return head

    def forward(self, sparse, mask, coords, valid_node_mask, lf):
        out = self.base(sparse, mask, coords, valid_node_mask, lf)
        latent = out["floor_latent"]
        if not 0 <= self.anchor_index < latent.shape[1]:
            raise ValueError(f"anchor_index={self.anchor_index} outside T={latent.shape[1]}")
        q0 = self.q0_head(latent[:, self.anchor_index]).squeeze(-1)
        raw_delta = self.delta_q_head(latent).squeeze(-1)
        delta = raw_delta - raw_delta[:, self.anchor_index:self.anchor_index + 1]
        response = out["response"].clone()
        response[..., 0] = q0[:, None, :] + delta
        out["response"] = response
        out["q0"] = q0
        out["delta_q"] = delta
        return out
