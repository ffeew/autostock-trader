# Target Normalization Fix - Applied Changes

## Problem Statement

**CRITICAL ISSUE**: Target variable (close price) was NOT normalized while features WERE normalized.

### Symptoms Observed:
- ✅ TCN models work well (val loss: 59-630)
- ❌ LSTM/GRU/Transformer models fail to converge (val loss: 1790-6755)
- ❌ Validation loss stuck or increasing
- ❌ Early stopping at 10-22 epochs
- ❌ Massive scale mismatch between inputs and outputs

### Root Cause:
- **Features**: Normalized with StandardScaler (mean=0, std=1)
- **Target**: RAW prices (mean=377, std=117, range=[181, 649])
- **Result**: Models trying to predict values in range [181-649] from normalized inputs [-3, +3]

This creates an impossible learning task for LSTM/GRU/Transformer models.

---

## Why TCN Worked But Others Didn't

### TCN (Worked):
- Conv1d layers handle scale mismatch better
- Aggressive downsampling forces robust feature learning
- Better inductive bias for this problem

### LSTM/GRU/Transformer (Failed):
- Sigmoid/tanh activations saturate with large targets
- Recurrent connections amplify scale issues
- Attention mechanisms confused by scale mismatch
- Training becomes nearly impossible

---

## Applied Fix

### Changed Files:

#### 1. `src/models/data_loader.py`

**Added target scaler (line 106)**:
```python
self.target_scaler = StandardScaler()  # Add scaler for target variable
```

**Fit and transform targets (lines 184-195)**:
```python
# Fit target scaler on training targets only
self.target_scaler.fit(train_targets.reshape(-1, 1))

# Transform all datasets
train_features = self.scaler.transform(train_features)
val_features = self.scaler.transform(val_features)
test_features = self.scaler.transform(test_features)

# Transform targets (CRITICAL: normalize target variable)
train_targets = self.target_scaler.transform(train_targets.reshape(-1, 1)).flatten()
val_targets = self.target_scaler.transform(val_targets.reshape(-1, 1)).flatten()
test_targets = self.target_scaler.transform(test_targets.reshape(-1, 1)).flatten()

logger.info("Data normalized using StandardScaler")
logger.info("Target variable normalized (mean=0, std=1)")
```

**Updated return signature (line 242)**:
```python
return train_loader, val_loader, test_loader, self.scaler, self.target_scaler
```

#### 2. `train_models.py`

**Updated to receive target_scaler (line 217)**:
```python
train_loader, val_loader, test_loader, scaler, target_scaler = data_loader.prepare_data()
```

**Added logging (line 224)**:
```python
logger.info(f"  Target normalization: ENABLED (mean=0, std=1)\n")
```

---

## Expected Results After Fix

### Loss Values:
- **Before**: Train ~100-1000, Val ~1000-7000 (scale mismatch)
- **After**: Train ~0.1-2.0, Val ~0.1-3.0 (normalized scale)

### Model Performance:
- ✅ **LSTM-basic**: Should converge (val loss < 1.0)
- ✅ **LSTM-stacked**: Should converge (val loss < 0.5)
- ✅ **LSTM-bidirectional**: Should converge (val loss < 0.5)
- ✅ **LSTM-attention**: Should improve further (val loss < 0.3)
- ✅ **GRU models**: Should converge (val loss < 2.0)
- ✅ **Transformer models**: Should converge (val loss < 2.0)
- ✅ **TCN models**: Should work even better (val loss < 0.2)

### Training Behavior:
- ✅ Validation loss **decreases** consistently
- ✅ Early stopping at 25-35 epochs (not 10-22)
- ✅ Train/val gap < 0.2 (not 1000s)
- ✅ Stable training (no explosions)

---

## Testing the Fix

### Quick Test:
```bash
source venv/bin/activate

# Train LSTM-basic (previously failed)
python train_models.py --models lstm:basic --epochs 30 --tensorboard

# In separate terminal
python view_tensorboard.py
```

### What to Look For:
1. **Initial losses**: Should be ~1.0-5.0 (not 1000s)
2. **Validation decreases**: Consistent downward trend
3. **Train/val gap**: < 0.5 throughout training
4. **Final val loss**: < 1.0 for LSTM-basic
5. **Early stopping**: Around epoch 20-30

### Full Test:
```bash
# Train all models (this will take ~12-16 hours)
python train_models.py --tensorboard
```

---

## Comparison: Before vs After

| Model | Before (Val Loss) | After (Expected) | Status |
|-------|------------------|------------------|--------|
| TCN-multiscale | 59 | < 0.5 | ✅ Works → Better |
| LSTM-attention | 408 | < 0.5 | ✅ Good → Better |
| TCN-basic | 630 | < 1.0 | ✅ Good → Better |
| LSTM-bidirectional | 673 | < 1.0 | ✅ Good → Better |
| LSTM-stacked | 1790 | < 1.0 | ⚠️ Failed → Fixed |
| GRU-residual | 2090 | < 2.0 | ⚠️ Failed → Fixed |
| TCN-residual | 2842 | < 2.0 | ⚠️ Failed → Fixed |
| GRU-basic | 3362 | < 2.0 | ❌ Failed → Fixed |
| GRU-batchnorm | 4361 | < 2.0 | ❌ Failed → Fixed |
| LSTM-basic | 4668 | < 1.0 | ❌ Failed → Fixed |
| Transformer-basic | 5686 | < 2.0 | ❌ Failed → Fixed |
| Transformer-multihead | 6155 | < 2.0 | ❌ Failed → Fixed |
| Transformer-informer | 6755 | < 2.0 | ❌ Failed → Fixed |

---

## Technical Details

### Why This Works:

1. **Matched Scales**: Both inputs and outputs now have mean=0, std=1
2. **Better Gradients**: Activation functions work in optimal range
3. **Faster Convergence**: Adam optimizer performs better with normalized targets
4. **Numerical Stability**: Prevents overflow/underflow in loss calculations

### Loss Calculation:
```python
# Before: MSE of values in range [181, 649]
loss = (pred - target)²  # pred ∈ [-3, 3], target ∈ [181, 649]
# Result: Massive loss values (1000s)

# After: MSE of normalized values
loss = (pred_norm - target_norm)²  # Both ∈ [-3, 3]
# Result: Reasonable loss values (0-5)
```

### Inverse Transform:
When making predictions in production:
```python
# Denormalize predictions
predictions_original_scale = target_scaler.inverse_transform(predictions_normalized)
```

---

## Why This Was Missed Initially

1. **TCN models worked** - masked the problem
2. **Features were normalized** - looked correct
3. **No explicit check** for target normalization
4. **Large loss values** seemed plausible for price prediction

This is a **classic machine learning mistake** - always normalize both X and y in regression tasks!

---

## Files Modified

1. ✏️ `src/models/data_loader.py` - Added target_scaler, normalization logic
2. ✏️ `train_models.py` - Updated to handle target_scaler
3. 📄 `docs/TARGET_NORMALIZATION_FIX.md` - This documentation

---

## Summary

**The fix is simple but critical**:
- Added `StandardScaler` for target variable
- Normalize targets to mean=0, std=1
- Match the scale of inputs and outputs

This single change should fix **10+ failing models** and dramatically improve training performance across all architectures.

**Status**: ✅ **IMPLEMENTED AND READY FOR TESTING**

---

**Last Updated**: 2025-10-03
**Tested**: Pending (awaiting user verification)
