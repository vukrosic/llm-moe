# LLM Research: Transformer vs Mixture of Experts

**🔍 Looking for bugs and improvements in the MoE implementation!** Please review the code and suggest fixes or enhancements.

This repository implements and compares two transformer architectures for language modeling. The `llm.py` file contains both standard Feed-Forward (FF) and Mixture of Experts (MoE) implementations for direct comparison.

## 📊 Final Results Summary (3000 training steps each)

| Model | Parameters | Active Params | Param Utilization | Val Loss | Val Acc | Val PPL | Training Time |
|-------|------------|---------------|------------------|----------|---------|---------|---------------|
| Regular Transformer | ~29M | 29M | 100% | 0.1365 | 0.9766 | 1.15 | 1.9 min |
| MoE (8 experts) | ~54M | ~25M | **47%** | 0.0758 | 0.9857 | 1.08 | 3.6 min |

## 🚀 Features

- **Regular Transformer**: Standard transformer with feed-forward networks
- **Mixture of Experts (MoE)**: Sparse transformer using top-k expert routing
- **Muon Optimizer**: Novel matrix-based optimizer for efficient training
- **RoPE Positional Embeddings**: Rotary position embeddings for better sequence understanding
- **RMSNorm**: Root mean square layer normalization
- **Automatic Mixed Precision**: For faster training with reduced memory usage

## 📊 Architecture Comparison

| Model | Parameters | Active Params | Efficiency |
|-------|------------|---------------|------------|
| Regular | ~29M | 29M | 100% |
| MoE (8 experts) | ~54M | ~25M | ~47% |

## 🧪 Experimental Setup

- **Dataset**: 500K tokens from SmolLM corpus
- **Model Size**: 384d, 6L, 8H, 1536ff
- **Training**: 3000 steps, batch size 24
- **Sequence Length**: 512 tokens
- **Evaluation**: Validation loss, accuracy, perplexity

## 🔧 Key Components

- `MinimalLLM`: Standard transformer implementation
- `MoEMinimalLLM`: Transformer with selective MoE layers
- `TopKRouter`: Sparse routing mechanism for experts
- `MixtureOfExperts`: Feed-forward layer with expert specialization
- `Muon`: Orthogonalized momentum optimizer

## 📈 Results Summary

Both models achieve strong performance with the MoE variant showing better parameter efficiency through sparse activation while maintaining competitive perplexity scores.

## 🛠️ Usage

```bash
python llm.py
```

The script automatically:
1. Downloads and tokenizes data
2. Trains both regular and MoE models
3. Compares their performance metrics
4. Reports training time and final results

## 📚 Dependencies

- PyTorch
- Transformers (HuggingFace)
- Torchtune (RoPE implementation)
- Datasets (HuggingFace)
- tqdm, numpy