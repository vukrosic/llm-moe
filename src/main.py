import time
from torch.utils.data import DataLoader
import torch

from .config import ModelConfig, MoEModelConfig, set_seed
from .train import (
    TextTokenDataset,
    load_and_cache_data,
    train_model,
    train_moe_model,
    calculate_model_flops,
)


def main():
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    set_seed(42)

    temp_config = ModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size

    base_config = ModelConfig(
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        vocab_size=vocab_size,
    )

    moe_config = MoEModelConfig(
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        batch_size=24,
        vocab_size=vocab_size,
        num_experts=8,
        expert_top_k=2,
        moe_layers="alternate",
        load_balancing_weight=0.01,
    )

    standard_flops = calculate_model_flops(base_config, is_moe=False)
    moe_flops = calculate_model_flops(moe_config, is_moe=True)

    print(f"\n🔢 FLOP Analysis:")
    print(f"   Standard Model FLOPs per step: {standard_flops:,}")
    print(f"   MoE Model FLOPs per step: {moe_flops:,}")
    print(f"   FLOP ratio (MoE/Standard): {moe_flops/standard_flops:.2f}x")

    target_total_flops = 8e14
    standard_steps = int(target_total_flops / standard_flops)
    moe_steps = int(target_total_flops / moe_flops)

    print(f"\n⚖️ Fair Training Setup:")
    print(f"   Target total FLOPs: {target_total_flops:,.0e}")
    print(f"   Standard model steps: {standard_steps:,}")
    print(f"   MoE model steps: {moe_steps:,}")

    base_config.max_steps = standard_steps
    base_config.eval_every = max(500, standard_steps // 20)
    base_config.log_milestones = (standard_steps // 4, standard_steps // 2, 3 * standard_steps // 4)

    moe_config.max_steps = moe_steps
    moe_config.eval_every = max(500, moe_steps // 20)
    moe_config.log_milestones = (moe_steps // 4, moe_steps // 2, 3 * moe_steps // 4)

    models_to_test = [
        ("Regular Transformer", base_config),
        ("Mixture of Experts", moe_config),
    ]

    dataset = TextTokenDataset(tokens, temp_config.max_seq_len)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=temp_config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=temp_config.batch_size, shuffle=False, num_workers=2)
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    for model_name, config in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {model_name}")
        print(f"{'='*60}")

        if isinstance(config, MoEModelConfig):
            print(f"\n📋 MoE Model Configuration:")
            print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
            print(f"   MoE: {config.num_experts} experts, top-{config.expert_top_k} routing")
            print(f"   MoE Layers: {config.moe_layers}")
            print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
            print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")
        else:
            print(f"\n📋 Regular Transformer Configuration:")
            print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
            print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
            print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

        start_time = time.time()
        if isinstance(config, MoEModelConfig):
            model, final_metrics = train_moe_model(config, train_loader, val_loader, tokenizer=tokenizer)
        else:
            model, final_metrics = train_model(config, train_loader, val_loader, tokenizer=tokenizer)
        total_time = time.time() - start_time

        total_flops_used = config.max_steps * calculate_model_flops(config, is_moe=isinstance(config, MoEModelConfig))
        flops_per_second = total_flops_used / max(1e-6, total_time)

        print(f"\n🎯 {model_name} Results:")
        print(f"⏱️ Training time: {total_time/60:.1f} minutes")
        print(f"🔢 Total FLOPs used: {total_flops_used:,.0e}")
        print(f"⚡ FLOPs per second: {flops_per_second:,.0e}")
        print(f"🏆 Final Results:")
        print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()


