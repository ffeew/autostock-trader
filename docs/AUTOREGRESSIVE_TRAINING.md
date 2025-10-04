# Auto-Regressive Sequential Prediction

## Overview

**This is the ONLY training mode** - all models predict multiple timesteps sequentially by feeding their own predictions back as input, similar to how Large Language Models (LLMs) generate text token-by-token.

## How It Works

### Auto-Regressive Sequential Prediction:
```python
Input:  [t0, t1, ..., t59]
Output: [t60, t61, ..., t89]  # 30 sequential predictions
```

**Process:**
1. Predict t60 from [t0...t59]
2. Predict t61 from [t0...t60] (using predicted t60)
3. Predict t62 from [t0...t61] (using predicted t60, t61)
4. Continue for 30 steps
5. Loss computed on final prediction (or all intermediate predictions)

**Benefits:**
- ✅ Model learns to use its own predictions
- ✅ Training matches inference behavior
- ✅ More robust to compounding errors
- ✅ Can generate arbitrary-length sequences
- ✅ Loss computed on ALL 30 predictions (not just final)

---

## Training Loop

### Sequential Prediction Process:

```python
for each batch:
    # Auto-regressive: predict iteratively
    predictions = []
    current_input = input_sequence

    for step in range(30):
        pred = model(current_input)
        predictions.append(pred)

        # Feed prediction back as input
        current_input = slide_window(current_input, pred)

    # Loss on ALL 30 predictions
    predictions_flat = predictions.squeeze(-1)  # (batch, 30)
    loss = MSE(predictions_flat, ground_truth_sequence)  # (batch, 30)
```

### Why This Works:

- Model sees its own predictions in the input window
- Learns to correct for its own errors
- Training matches real-world inference behavior
- Loss signal from all intermediate predictions guides learning

---

## Usage

### Training (Always Auto-Regressive):

```bash
# Train all models with auto-regressive prediction (default)
python train_models.py --tensorboard

# Train specific models (all auto-regressive)
python train_models.py --models lstm:basic gru:residual --tensorboard

# Customize hyperparameters (30-step prediction is default)
python train_models.py --epochs 100 --hidden-size 256 --tensorboard
```

### Configuration:

- **Prediction steps**: Controlled by `PREDICTION_STEPS` in `.env` (default: 30)
- **All models**: Train using auto-regressive sequential prediction
- **No flag needed**: Auto-regressive is the only mode

---

## Implementation Details

### In `BaseModel`:

```python
def forward_autoregressive(self, x, num_steps):
    """Predict num_steps into the future iteratively."""
    predictions = []
    current_x = x.clone()

    for step in range(num_steps):
        # Predict next value
        pred = self.forward(current_x)
        predictions.append(pred)

        # Update input: slide window + add prediction
        next_features = current_x[:, -1, :].clone()
        next_features[:, -1] = pred.squeeze(-1)  # Update close price

        # Slide window
        current_x = torch.cat([current_x[:, 1:, :], next_features.unsqueeze(1)], dim=1)

    return torch.stack(predictions, dim=1)
```

### Training Implementation:

```python
# All models use auto-regressive prediction
predictions = model.forward_autoregressive(batch_x, prediction_steps)
# predictions: (batch, num_steps, 1)

# Compute loss on ALL predictions
predictions_flat = predictions.squeeze(-1)  # (batch, num_steps)
loss = criterion(predictions_flat, batch_y)  # batch_y: (batch, num_steps)
```

---

## Expected Behavior

### Training Time:
- **~30-60 minutes per epoch** (30 sequential forward passes per batch)
- Significantly slower than single-step prediction
- Recommended: Use GPU for training

### Loss Values:
- Initial losses ~1-5 (normalized targets)
- Training computes MSE across all 30 predictions
- More stable convergence due to multi-step loss signal

### Model Performance:
- Better long-term prediction accuracy
- More stable when making iterative predictions
- Less sensitive to compounding errors
- Training matches real-world inference behavior

---

## Example Training Commands

### Quick Test:
```bash
# Test all models with 2 epochs
python test_models.py
```

### Train Single Model:
```bash
# Train LSTM with TensorBoard
python train_models.py \
    --models lstm:basic \
    --epochs 50 \
    --tensorboard
```

### Train Multiple Models:
```bash
# Train multiple architectures (all auto-regressive)
python train_models.py \
    --models lstm:basic lstm:bidirectional gru:residual \
    --epochs 50 \
    --tensorboard
```

### Full Production Training:
```bash
# Train all models with custom hyperparameters
python train_models.py \
    --epochs 100 \
    --hidden-size 256 \
    --batch-size 64 \
    --tensorboard
```

---

## Technical Notes

### Feature Engineering:
Currently, only the `close` price (last feature) is updated with predictions. Other features (volume, indicators, etc.) are copied from the last timestep.

**Future Enhancement:**
Could implement full feature prediction or use external forecasts for other features.

### Loss Computation:
Loss is computed on ALL 30 predictions:
- MSE between predicted sequence and ground truth sequence
- All predictions contribute to gradient updates
- More training signal than final-only loss
- Could be enhanced with weighted loss (more weight on later predictions)

### Gradient Flow:
Gradients backpropagate through all N steps, which:
- ✅ Allows model to learn from sequential errors
- ⚠️ May cause vanishing/exploding gradients for large N
- Solution: Gradient clipping (already implemented)

---

## Limitations

1. **Slower Training**: N times slower (N = number of steps)
2. **Memory Usage**: N times more memory (stores N predictions)
3. **Feature Updates**: Only updates close price, not all features
4. **Fixed Horizon**: Must specify prediction horizon upfront

---

## Future Enhancements

1. **Scheduled Sampling**: Mix ground truth and predictions during training
2. **Multi-target Loss**: Compute loss on all intermediate predictions
3. **Full Feature Prediction**: Predict all features, not just close price
4. **Variable Horizon**: Support different prediction horizons dynamically
5. **Parallel Decoding**: Batch multiple prediction steps for efficiency

---

## Summary

All models use auto-regressive sequential prediction - predicting one step at a time, using previous predictions as input, like LLMs generate text token-by-token.

**Key Points:**
- **Always enabled**: Auto-regressive is the only training mode
- **30-step default**: Controlled by `PREDICTION_STEPS` in `.env`
- **Full sequence loss**: All 30 predictions contribute to training
- **Slower but better**: Training matches real-world inference

**Training Command:**
```bash
python train_models.py --tensorboard
```

---

**Last Updated**: 2025-10-04
**Status**: ✅ Production-Ready (Only Training Mode)
