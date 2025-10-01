# TensorBoard Training Visualization Guide

## Overview

TensorBoard is now fully integrated into the AutoStock Trader training pipeline, providing real-time visualization of model training progress.

## Quick Start

### 1. Train with TensorBoard enabled

```bash
# Train all models with TensorBoard
python train_models.py --tensorboard

# Train specific models
python train_models.py --models lstm:attention gru:residual --tensorboard
```

### 2. View TensorBoard

**Option A: Using the helper script (recommended)**
```bash
python view_tensorboard.py
```

**Option B: Manual launch**
```bash
tensorboard --logdir=runs
```

### 3. Open your browser

Navigate to: **http://localhost:6006**

---

## What Gets Logged

### Scalar Metrics (updated every epoch)

1. **Loss/train**: Training loss per epoch
2. **Loss/validation**: Validation loss per epoch
3. **Learning_Rate**: Current learning rate
4. **Gradient/norm**: Total gradient norm (for debugging)
5. **Best_Val_Loss**: Logged when a new best model is saved

### Histograms (updated every 10 epochs)

1. **Weights/{layer_name}**: Distribution of model weights
2. **Gradients/{layer_name}**: Distribution of gradients during backpropagation

---

## TensorBoard Features

### 1. SCALARS Tab

**View loss curves in real-time:**
- Monitor training vs validation loss
- Detect overfitting immediately (val loss increases while train loss decreases)
- Compare multiple models side-by-side
- Smooth curves with the slider

**Usage tips:**
- Toggle individual runs on/off for comparison
- Use smoothing to reduce noise
- Download data as CSV for further analysis

### 2. GRAPHS Tab

**Visualize model architecture:**
- See the computational graph
- Understand data flow through layers
- Identify bottlenecks

### 3. DISTRIBUTIONS Tab

**Analyze weight/gradient distributions:**
- Check if gradients are vanishing or exploding
- Verify proper initialization
- Monitor weight updates over time

### 4. HISTOGRAMS Tab

**3D visualization of distributions:**
- See how distributions evolve during training
- Spot dead neurons (weights/gradients not changing)
- Identify layer-specific issues

---

## Directory Structure

```
runs/
└── 20250110_143022/           # Timestamp of training run
    ├── lstm_basic/            # Individual model logs
    │   └── events.out.tfevents...
    ├── lstm_attention/
    │   └── events.out.tfevents...
    ├── gru_residual/
    │   └── events.out.tfevents...
    └── ...
```

Each model gets its own subdirectory, making it easy to compare different architectures.

---

## Comparing Multiple Models

### Compare all models from one training run:

```bash
tensorboard --logdir=runs/20250110_143022
```

All models will appear in the same dashboard.

### Compare across multiple training runs:

```bash
tensorboard --logdir=runs
```

This loads all runs. Use TensorBoard's filtering to focus on specific models.

### Compare specific models:

Create a custom directory structure:
```bash
mkdir -p comparison/lstm_models
ln -s ../../runs/20250110_143022/lstm_basic comparison/lstm_models/
ln -s ../../runs/20250110_143022/lstm_attention comparison/lstm_models/

tensorboard --logdir=comparison/lstm_models
```

---

## Advanced Usage

### Custom Port

```bash
python view_tensorboard.py --port 6007
# Or: tensorboard --logdir=runs --port 6007
```

### Remote Server Access

If training on a remote server:

```bash
# On remote server
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@remote-server
```

Then open http://localhost:6006 on your local machine.

### Filter Runs

In TensorBoard UI:
- Use regex in the filter box (bottom left)
- Example: `lstm.*` shows only LSTM models
- Example: `.*attention` shows all attention models

---

## Interpreting the Visualizations

### Good Training Signs ✅

1. **Loss curves**: Both train and val loss decreasing smoothly
2. **Small gap**: Train loss slightly lower than val loss (expected)
3. **Gradient norms**: Stable, not exploding or vanishing
4. **Weight distributions**: Evolving but not collapsing to zero

### Warning Signs ⚠️

1. **Overfitting**: Val loss increases while train loss decreases
2. **Underfitting**: Both losses plateau at high values
3. **Exploding gradients**: Gradient norm spikes dramatically
4. **Vanishing gradients**: Gradient norm approaches zero
5. **Dead neurons**: Weights/gradients not updating

### Actions to Take

**If overfitting:**
- Increase dropout rate
- Add regularization
- Reduce model complexity
- Get more training data

**If underfitting:**
- Increase model capacity (hidden_size, num_layers)
- Train for more epochs
- Reduce regularization
- Check data quality

**If gradients exploding:**
- Reduce learning rate
- Use gradient clipping (already implemented)
- Check data normalization

**If gradients vanishing:**
- Use residual connections (ResidualGRU, ResidualTCN)
- Try different activation functions
- Check initialization

---

## Keyboard Shortcuts

In TensorBoard UI:
- `d`: Toggle dark mode
- `r`: Reload data
- `←/→`: Navigate between tabs
- `Ctrl + /`: Search

---

## Exporting Data

### Download plots as images

1. Hover over any plot
2. Click the camera icon (top right)
3. Save as PNG or SVG

### Export data as CSV

1. Click "Show data download links" (below chart)
2. Download CSV for any run
3. Analyze in Excel, Python, etc.

---

## Troubleshooting

### TensorBoard not starting

```bash
# Check if tensorboard is installed
pip install tensorboard

# Check if port is in use
lsof -i :6006

# Try a different port
tensorboard --logdir=runs --port=6007
```

### No data showing up

```bash
# Verify logs exist
ls -R runs/

# Check for event files
find runs/ -name "events.out.tfevents*"

# Ensure you trained with --tensorboard flag
python train_models.py --tensorboard
```

### Old runs cluttering the view

```bash
# Archive old runs
mkdir -p runs/archive
mv runs/20250110_* runs/archive/

# Or delete old runs
rm -rf runs/20250110_*
```

---

## Best Practices

1. **Always use `--tensorboard`** during actual training runs
2. **Keep TensorBoard open** in a separate terminal to monitor progress
3. **Check gradients early** - if they're problematic, stop and adjust
4. **Compare architectures** using the same hyperparameters first
5. **Archive successful runs** for future reference
6. **Take screenshots** of important findings

---

## Integration Details

### How it works

1. **BaseModel** (`src/models/base_model.py`) has built-in TensorBoard support
2. **train_models.py** passes `tensorboard_log_dir` to each model
3. **SummaryWriter** automatically logs metrics during training
4. **All models** inherit this functionality - no code changes needed

### Adding custom metrics

If you want to log custom metrics in a specific model:

```python
# In your model's fit() method
if writer:
    writer.add_scalar('Custom/metric_name', metric_value, epoch)
```

### Logging frequency

- **Scalars**: Every epoch
- **Histograms**: Every 10 epochs (to reduce overhead)
- **Best model**: When validation loss improves

---

## Resources

- **TensorBoard Documentation**: https://www.tensorflow.org/tensorboard
- **PyTorch TensorBoard Tutorial**: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
- **Visualization Best Practices**: https://www.tensorflow.org/tensorboard/get_started

---

## Summary

TensorBoard integration provides:

✅ **Real-time monitoring** - Watch training progress live
✅ **Early problem detection** - Spot issues before wasting compute
✅ **Model comparison** - Objectively compare architectures
✅ **Debugging tools** - Understand gradient flow and weight updates
✅ **Publication-ready plots** - Export professional visualizations

**Command to remember:**
```bash
python train_models.py --tensorboard
python view_tensorboard.py
```

That's it! Happy training! 🚀
