import torch
import torch.nn as nn
from torch.nn import LayerNorm


class MLP(nn.Module):
    def __init__(self, embed_channels: int, hidden_size: int) -> None:
        self.mlp = nn.Sequential(
            nn.Linear(embed_channels, embed_channels),
            nn.GELU(),
            nn.Linear(embed_channels, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# borrowed from https://github.com/huggingface/transformers/blob/94df0e65602922be2831b3faa457a2bde78b936b/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L255
# Qwen2-VL PatchMerger
class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


if __name__ == "__main__":
    # test
    x = torch.randn(1, 4096, 512)  # B, L, C
    patch_merger = PatchMerger(dim=896, context_dim=512, spatial_merge_size=2)
    y = patch_merger(x)
    print(y.shape)  # should be (1024, 896)
