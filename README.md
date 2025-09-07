# LLM Research: Transformer vs Mixture of Experts

**🔍 Looking for bugs and improvements in the MoE implementation!** Please review the code and suggest fixes or enhancements.

This repository implements and compares two transformer architectures for language modeling. The `llm-same-flops-data.py` file contains both standard Feed-Forward (FF) and Mixture of Experts (MoE) implementations with **fair FLOP-based comparison**.

## 📊 Latest Results Summary (Fair FLOP Comparison) - Model is small, so regular FF works better

| Model | Parameters | Active Params | Param Utilization | Val Loss | Val Acc | Val PPL | Training Time | Total FLOPs |
|-------|------------|---------------|------------------|----------|---------|---------|---------------|-------------|
| Regular Transformer | ~29M | 29M | 100% | **0.4018** | **0.9289** | **1.49** | 1.5 min | 8e14 |
| MoE (8 experts, top-2) | ~54M | ~25M | **47.8%** | 0.4639 | 0.9164 | 1.59 | 2.2 min | 8e14 |

> **Key Finding**: With longer training and equal FLOP budgets, the regular transformer achieved better final performance despite MoE's parameter efficiency advantage.

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

- **Dataset**: 500K tokens from SmolLM corpus (cosmopedia-v2)
- **Model Size**: 384d, 6L, 8H, 1536ff
- **Training**: Equal FLOP budgets (8e14 total FLOPs each)
  - Regular Transformer: 2,122 steps
  - MoE: 1,724 steps
- **Batch Size**: 24, **Sequence Length**: 512 tokens
- **Fair Comparison**: Both models use identical total computational budget
- **Evaluation**: Validation loss, accuracy, perplexity at regular intervals

## 🔧 Key Components

- `MinimalLLM`: Standard transformer implementation
- `MoEMinimalLLM`: Transformer with selective MoE layers
- `TopKRouter`: Sparse routing mechanism for experts
- `MixtureOfExperts`: Feed-forward layer with expert specialization
- `Muon`: Orthogonalized momentum optimizer

## 📈 Results Analysis & Key Insights

### 🔍 **Surprising Results: Regular Transformer Wins**

**Counterintuitive Finding**: Despite MoE's theoretical advantages (47.8% parameter efficiency, sparse activation), the regular transformer achieved better final performance after extended training with equal computational budgets.

### 📊 **Performance Comparison**
- **Regular Transformer**: Better final metrics (Loss: 0.40 vs 0.46, PPL: 1.49 vs 1.59)
- **MoE Model**: Slower convergence, required more training time despite fewer active parameters
- **Both models**: Achieved excellent perplexity (< 2.0), demonstrating strong language modeling capabilities

### 💡 **Possible Explanations**
1. **Training Dynamics**: MoE's routing complexity may require different optimization strategies
2. **Data Scale**: For smaller datasets (500K tokens), regular transformers may converge faster
3. **Expert Specialization**: The 8 experts may not provide sufficient benefit for this task complexity
4. **Load Balancing**: Despite auxiliary losses, expert utilization may not be optimal

### 📈 **Training Progression Analysis**

**Regular Transformer Convergence:**
- Step 500: Loss 5.19, PPL 180 → Step 1000: Loss 3.22, PPL 25 → Step 2000: Loss 0.51, PPL 1.67
- **Rapid improvement**: Perplexity dropped from 180 to 1.67 over 1500 steps
- **Final**: Loss 0.40, PPL 1.49 (excellent convergence)

**MoE Model Convergence:**
- Step 500: Loss 4.96, PPL 143 → Step 1000: Loss 2.86, PPL 17 → Step 1500: Loss 0.93, PPL 2.53
- **Slower improvement**: Required more steps for similar perplexity reduction
- **Final**: Loss 0.46, PPL 1.59 (good but slower convergence)

### ⏱️ **Timing Analysis: FLOPs vs Wall-Clock Time**

**Key Finding**: Despite identical FLOP budgets, MoE took ~47% longer per training step

**Why the timing difference?**
- **Implementation Inefficiency**: Original MoE used sequential expert processing with complex indexing
- **Memory Access Patterns**: Irregular memory access due to conditional token routing
- **Kernel Launch Overhead**: Multiple tensor operations and auxiliary loss computation
- **Optimization Opportunity**: Recent vectorized implementation should reduce this gap

**Before vs After Optimization**:
- **Before**: Sequential expert loops + conditional processing
- **After**: Vectorized operations with `index_add_` for efficient aggregation

### 🎯 **Implications**
- **Convergence Speed**: Regular transformer converged faster despite higher parameter count
- **Optimization Challenge**: MoE routing complexity may hinder training dynamics
- **Parameter efficiency ≠ Performance**: Sparse activation doesn't always translate to better results
- **Architecture Choice**: Depends on training budget, data scale, and computational constraints
- **Further Research Needed**: Test with larger datasets, different expert counts, improved routing

## 🛠️ Usage

### Main Fair Comparison Script (Recommended)
```bash
python llm-same-flops-data.py
```

**Features**:
- ✅ **Fair FLOP-based comparison** - both models use identical computational budgets
- ✅ **Automatic step calculation** - adjusts training steps to equalize FLOPs
- ✅ **Comprehensive evaluation** - validation at milestones and final results
- ✅ **Detailed reporting** - FLOPs per second, parameter efficiency metrics

**What it does**:
1. Downloads and caches tokenized data from SmolLM corpus
2. Calculates fair training steps based on model FLOPs
3. Trains both models with equal total computational budget
4. Reports detailed performance metrics and efficiency comparisons

### Legacy Script
```bash
python llm.py
```
Original implementation with fixed training steps (not FLOP-equalized).

## 🔬 Future Research Directions

- **Scale Testing**: Evaluate with larger datasets (10M+ tokens) where MoE benefits may emerge
- **Expert Count Optimization**: Test different numbers of experts (4, 16, 32) and routing strategies
- **Improved Routing**: Implement more sophisticated routing mechanisms (e.g., learned temperature, expert capacity)
- **Load Balancing**: Enhance expert utilization through better auxiliary loss functions
- **Hybrid Approaches**: Combine MoE with other efficiency techniques (quantization, distillation)
- **Task-Specific MoE**: Domain-adaptive expert specialization for different tasks

## 🐛 Known Issues & Improvements Needed

- **MoE Convergence**: Investigate why MoE converges slower than expected
- **Expert Utilization**: Monitor and improve expert load balancing during training
- **Memory Efficiency**: Optimize MoE memory usage for larger models
- **Routing Stability**: Address potential routing instabilities in early training
- **✅ FIXED - Timing Inefficiency**: Vectorized expert processing implemented to reduce wall-clock time gap with regular transformer

## 📚 Dependencies

- PyTorch
- Transformers (HuggingFace)
- Torchtune (RoPE implementation)
- Datasets (HuggingFace)
- tqdm, numpy

## 📄 Files

- `llm-same-flops-data.py`: Main fair comparison script with FLOP-equalized training
- `llm.py`: Legacy implementation with fixed training steps
- `gpu_monitor.py`: GPU monitoring utilities
- `requirements.txt`: Python dependencies