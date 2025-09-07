import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")


@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 3000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # HellaSwag evaluation
    hellaswag_eval_every: int | None = None  # e.g., 500; None disables
    hellaswag_max_samples: int = 1000
    hellaswag_split: str = "validation"

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class MoEModelConfig(ModelConfig):
    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    moe_layers: str = "alternate"  # "all", "alternate", or "last_half"
    load_balancing_weight: float = 0.01

    def should_use_moe(self, layer_idx: int) -> bool:
        """Determine if a specific layer should use MoE"""
        if self.moe_layers == "all":
            return True
        elif self.moe_layers == "alternate":
            return layer_idx % 2 == 1  # Every other layer
        elif self.moe_layers == "last_half":
            return layer_idx >= self.n_layers // 2
        return False


