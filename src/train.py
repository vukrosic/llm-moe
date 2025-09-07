import math
import os
import pickle
import time
import json
from datetime import datetime
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import ModelConfig, MoEModelConfig, set_seed
from .model import MinimalLLM, MoEMinimalLLM


class TextTokenDataset(Dataset):
    def __init__(self, tokens, seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size
        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])
    print(f"Loaded {len(texts)} documents")
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens


def load_hellaswag(split: str = "validation"):
    """Load HellaSwag dataset split from Hugging Face hub."""
    return load_dataset("AIGym/hellaswag", split=split)


@torch.no_grad()
def evaluate_hellaswag(
    model: torch.nn.Module,
    tokenizer,
    config: ModelConfig,
    max_samples: int = 1000,
    split: str = "validation",
    device: Optional[torch.device] = None,
):
    """Evaluate multiple-choice accuracy on HellaSwag.

    Scores each candidate (ctx + ending) with average token log-likelihood.
    """
    if device is None:
        device = next(model.parameters()).device

    ds = load_hellaswag(split)
    n = min(max_samples, len(ds))
    correct = 0
    total = 0

    model.eval()
    for i in range(n):
        ex = ds[i]
        context = ex.get("ctx", "")
        endings = ex.get("endings", [])
        label = int(ex.get("label", 0))
        if not endings or len(endings) != 4:
            continue

        scores = []
        for ending in endings:
            text = (context + " " + ending).strip()
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                scores.append(-1e9)
                continue
            x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(device)
            with autocast(enabled=config.use_amp):
                out = model(x) if not (hasattr(model, 'config') and isinstance(model.config, MoEModelConfig)) else model(x, return_aux_loss=False)
                logits = out[0] if isinstance(out, tuple) else out
                logprobs = torch.log_softmax(logits, dim=-1)
                token_ll = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                avg_ll = token_ll.mean().item()
                scores.append(avg_ll)

        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == label)
        total += 1

    acc = correct / max(1, total)
    model.train()
    return {"hellaswag_accuracy": acc, "hellaswag_samples": total}


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"]) 
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])  # type: ignore
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def setup_muon_optimizer(model: torch.nn.Module, config: ModelConfig):
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if (param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name and param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")
    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)
    return [muon_optimizer, adamw_optimizer]


def evaluate_model(model: torch.nn.Module, val_loader: DataLoader, config: ModelConfig):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)
            with autocast(enabled=config.use_amp):
                if hasattr(model, 'config') and isinstance(model.config, MoEModelConfig):
                    logits = model(x, return_aux_loss=False)
                else:
                    logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}


def calculate_model_flops(config: ModelConfig, is_moe: bool = False) -> int:
    """Calculate approximate FLOPs per forward pass for reporting/fair comparisons."""
    batch_size = config.batch_size
    seq_len = config.max_seq_len
    d_model = config.d_model
    n_heads = config.n_heads
    n_layers = config.n_layers
    d_ff = config.d_ff

    embedding_flops = batch_size * seq_len * d_model

    attention_flops_per_layer = (
        3 * batch_size * seq_len * d_model**2 +
        batch_size * n_heads * seq_len**2 * (d_model // n_heads) +
        batch_size * seq_len * d_model**2
    )

    if is_moe and hasattr(config, 'num_experts'):
        active_experts = getattr(config, 'expert_top_k', 1)
        ff_flops_per_layer = (
            batch_size * seq_len * d_model * d_ff * active_experts +
            batch_size * seq_len * d_ff * d_model * active_experts
        )
    else:
        ff_flops_per_layer = (
            batch_size * seq_len * d_model * d_ff +
            batch_size * seq_len * d_ff * d_model
        )

    total_flops = (
        embedding_flops +
        n_layers * (attention_flops_per_layer + ff_flops_per_layer) +
        batch_size * seq_len * d_model * (config.vocab_size or d_model)
    )
    return int(total_flops)


def _ensure_dirs():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def _save_checkpoint(model, step: int):
    _ensure_dirs()
    path = os.path.join("checkpoints", f"ckpt_step_{step}.pt")
    torch.save(model.state_dict(), path)
    return path


def _save_final_model(model, name: str):
    _ensure_dirs()
    path = os.path.join("outputs", f"{name}.pt")
    torch.save(model.state_dict(), path)
    return path


def _log_event(event: dict):
    _ensure_dirs()
    event["ts"] = datetime.utcnow().isoformat()
    with open(os.path.join("logs", "train.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader, tokenizer=None):
    print(f"\n🚀 Training Small model with Muon optimizer")
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")
    optimizers = setup_muon_optimizer(model, config)
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)
    scaler = GradScaler() if config.use_amp else None
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    pbar = tqdm(total=config.max_steps, desc="Training")
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            x, y = x.to(device), y.to(device)
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, Val Acc: {eval_metrics['val_accuracy']:.4f}, Val PPL: {eval_metrics['val_perplexity']:.2f}")
                _log_event({"type": "eval", "step": step, **eval_metrics})
                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                ck = _save_checkpoint(model, step)
                _log_event({"type": "checkpoint", "step": step, "path": ck})
            if step in getattr(config, 'log_milestones', ()):    
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\n🧪 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}")
            # Optional HellaSwag eval
            if getattr(config, 'hellaswag_eval_every', None) and tokenizer is not None and step > 0 and step % int(config.hellaswag_eval_every) == 0:
                hs = evaluate_hellaswag(model, tokenizer, config, max_samples=getattr(config, 'hellaswag_max_samples', 1000), split=getattr(config, 'hellaswag_split', 'validation'), device=device)
                print(f"  📚 HellaSwag@{step}: acc={hs['hellaswag_accuracy']:.3f} on {hs['hellaswag_samples']} samples")
            step += 1
            if step % 100 == 0:
                pbar.update(100)
    pbar.close()
    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")
    _log_event({"type": "final", **final_eval, "seconds": training_time})
    final_path = _save_final_model(model, name="final_standard")
    _log_event({"type": "final_model", "path": final_path})
    return model, final_eval


def train_moe_model(config: MoEModelConfig, train_loader: DataLoader, val_loader: DataLoader, tokenizer=None):
    print(f"\n🚀 Training MoE model with {config.num_experts} experts (top-{config.expert_top_k})")
    set_seed(42)
    model = MoEMinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for n, p in model.named_parameters() if 'expert' not in n)
    expert_params = total_params - active_params
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")
    optimizers = setup_muon_optimizer(model, config)
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)
    scaler = GradScaler() if config.use_amp else None
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training MoE")
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            x, y = x.to(device), y.to(device)
            if config.use_amp:
                with autocast():
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    loss = total_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss
                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}'
                })
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, Val Acc: {eval_metrics['val_accuracy']:.4f}, Val PPL: {eval_metrics['val_perplexity']:.2f}")
                _log_event({"type": "eval", "step": step, **eval_metrics})
                ck = _save_checkpoint(model, step)
                _log_event({"type": "checkpoint", "step": step, "path": ck})
            if step in getattr(config, 'log_milestones', ()):    
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\n🧪 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}")
            # Optional HellaSwag eval
            if getattr(config, 'hellaswag_eval_every', None) and tokenizer is not None and step > 0 and step % int(config.hellaswag_eval_every) == 0:
                hs = evaluate_hellaswag(model, tokenizer, config, max_samples=getattr(config, 'hellaswag_max_samples', 1000), split=getattr(config, 'hellaswag_split', 'validation'), device=device)
                print(f"  📚 HellaSwag@{step}: acc={hs['hellaswag_accuracy']:.3f} on {hs['hellaswag_samples']} samples")
            step += 1
            if step % 100 == 0:
                pbar.update(100)
    pbar.close()
    final_eval = evaluate_model(model, val_loader, config)
    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    _log_event({"type": "final", **final_eval})
    final_path = _save_final_model(model, name="final_moe")
    _log_event({"type": "final_model", "path": final_path})
    return model, final_eval

