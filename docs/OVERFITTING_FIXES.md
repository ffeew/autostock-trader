# LSTM Overfitting Fixes - Applied Changes

## Problem Statement

**Symptoms observed:**
- ✅ Training loss decreases normally
- ❌ Validation loss does NOT decrease (sometimes lowest at epoch 1)
- ❌ Early stopping triggers around epoch 15
- ❌ Classic overfitting: model memorizes training data without generalizing

## Root Causes Identified

### 1. **Data Shuffling for Time Series** (CRITICAL ⚠️)
**Problem**: Training DataLoader had `shuffle=True`
**Why it's bad**:
- Breaks temporal order at batch level
- Stock data has momentum/trends that shuffling destroys
- Model learns wrong patterns (future patterns before past patterns)
- Validation set maintains order → massive train/val distribution mismatch

### 2. **Missing Dropout in BasicLSTM**
**Problem**: Single-layer LSTM had `dropout=0` in PyTorch LSTM layer
**Why it's bad**:
- No regularization during training
- Model memorizes training sequences exactly
- Unable to generalize to validation data

### 3. **Learning Rate Too High**
**Problem**: Default LR = 0.001
**Why it's bad**:
- Too aggressive for financial time series
- Overshoots optimal weights
- Causes training instability

### 4. **No Learning Rate Adaptation**
**Problem**: Fixed learning rate throughout training
**Why it's bad**:
- Cannot fine-tune when close to optimal
- Misses opportunity to recover from plateaus

### 5. **Poor Weight Initialization**
**Problem**: Using PyTorch defaults
**Why it's bad**:
- May start in poor regions of loss landscape
- Slower convergence
- Higher chance of local minima

---

## Applied Fixes

### ✅ Fix 1: Disabled Batch Shuffling
**File**: `src/models/data_loader.py:205`
**Change**:
```python
# Before
shuffle=True,  # Shuffle within batches, but time order preserved in sequences

# After
shuffle=False,  # NO shuffling for time series - preserves temporal order
```
**Impact**: **CRITICAL** - Maintains temporal order essential for time series

---

### ✅ Fix 2: Added Explicit Dropout to BasicLSTM
**File**: `src/models/lstm_models.py`
**Changes**:
```python
# Added dropout layer (active regardless of num_layers)
self.dropout = nn.Dropout(dropout)

# In forward():
last_output = self.dropout(last_output)
```
**Also applied to**: BidirectionalLSTM
**Impact**: **HIGH** - Prevents memorization, forces generalization

---

### ✅ Fix 3: Lowered Default Learning Rate
**Files**:
- `src/models/base_model.py:148`
- `train_models.py:134`

**Change**:
```python
# Before
learning_rate: float = 0.001

# After
learning_rate: float = 0.0005
```
**Impact**: **HIGH** - More stable convergence, less overshooting

---

### ✅ Fix 4: Added Learning Rate Scheduler
**File**: `src/models/base_model.py:182-189`
**Change**:
```python
# Add ReduceLROnPlateau scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Reduce LR by 50% when plateau detected
    patience=5,      # Wait 5 epochs before reducing
    verbose=True,
    min_lr=1e-6
)

# Step scheduler based on validation loss
scheduler.step(val_loss)
```
**Impact**: **MEDIUM** - Adapts learning rate when stuck, improves fine-tuning

---

### ✅ Fix 5: Added Proper Weight Initialization
**File**: `src/models/lstm_models.py`
**Changes** (all LSTM models):
```python
def _init_weights(self):
    """Initialize weights using Xavier uniform initialization."""
    for name, param in self.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)  # Input-to-hidden weights
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)      # Hidden-to-hidden weights
        elif 'bias' in name:
            param.data.fill_(0)                  # Biases to zero
        elif 'fc' in name and 'weight' in name:
            nn.init.xavier_uniform_(param.data)  # Fully connected layers
```
**Impact**: **MEDIUM** - Better starting point, faster convergence

---

## Testing the Fixes

### Quick Test (Recommended)
```bash
# Train BasicLSTM for 30 epochs with TensorBoard
source venv/bin/activate
python train_models.py --models lstm:basic --epochs 30 --tensorboard

# In separate terminal, monitor progress
python view_tensorboard.py
```

### Expected Behavior After Fixes

#### ✅ **Good Signs** (what you should see):
1. **Validation loss decreases** (at least initially)
2. **Validation loss NOT lowest at epoch 1** (should improve for 10-20 epochs)
3. **Smaller train/val gap** (train loss only slightly lower than val loss)
4. **Early stopping around epoch 20-25** (not epoch 15)
5. **Learning rate reductions** (visible in logs: "Reducing learning rate...")

#### ⚠️ **Warning Signs** (if you still see these, further investigation needed):
1. Validation loss still higher at epoch 10 than epoch 1
2. Train/val gap > 0.01 and widening
3. Early stopping before epoch 15
4. No learning rate reductions

### Monitoring in TensorBoard

Open http://localhost:6006 and check:

1. **SCALARS → Loss/validation**: Should show **downward trend** initially
2. **SCALARS → Learning_Rate**: Should show **step-down pattern** (0.0005 → 0.00025 → ...)
3. **SCALARS → Gradient/norm**: Should be **stable** (0.5 - 2.0 range)

---

## Comparison: Before vs After

| Metric | Before Fixes | After Fixes (Expected) |
|--------|-------------|------------------------|
| Val loss at epoch 1 | Lowest (e.g., 0.0123) | Higher than epoch 10 |
| Val loss at epoch 10 | Higher (e.g., 0.0145) | Lower (e.g., 0.0110) |
| Val loss at epoch 20 | Same/Higher | Lowest or near-lowest |
| Early stopping epoch | ~15 | ~25 |
| Train/val gap | >0.015 | <0.005 |
| Learning rate at epoch 20 | 0.0005 (fixed) | 0.000125 (adaptive) |

---

## Additional Recommendations (If Issues Persist)

### If validation loss still doesn't decrease:

1. **Check data quality**:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/SPY_features.parquet')
   print('NaN values:', df.isnull().sum().sum())
   print('Inf values:', df.isin([float('inf'), float('-inf')]).sum().sum())
   print('Feature ranges:', df.describe())
   "
   ```

2. **Try even lower learning rate**:
   ```bash
   python train_models.py --models lstm:basic --learning-rate 0.0001 --epochs 50
   ```

3. **Increase dropout**:
   - Edit `src/models/lstm_models.py`
   - Change default dropout from 0.2 to 0.3 or 0.4

4. **Reduce model complexity**:
   ```bash
   python train_models.py --models lstm:basic --hidden-size 64 --num-layers 1
   ```

5. **Check sequence length**:
   - Current: 60 minutes
   - Try shorter: `--sequence-length 30` (faster feedback)
   - Try longer: `--sequence-length 120` (more context)

---

## Files Modified

1. ✏️ `src/models/data_loader.py` - Disabled shuffle
2. ✏️ `src/models/lstm_models.py` - Added dropout, weight init to all models
3. ✏️ `src/models/base_model.py` - Lowered LR, added scheduler
4. ✏️ `train_models.py` - Lowered default LR

---

## Summary

These fixes address the core issues causing LSTM overfitting:

1. **Temporal order preservation** (no shuffling) ← Most critical
2. **Better regularization** (explicit dropout)
3. **Slower, adaptive learning** (lower LR + scheduler)
4. **Better initialization** (Xavier/orthogonal)

The changes are **non-breaking** - existing code still works, just with better defaults.

**Next Steps**:
1. Test with: `python train_models.py --models lstm:basic --epochs 30 --tensorboard`
2. Monitor validation loss in TensorBoard
3. If still issues, try recommendations above
4. Once BasicLSTM works, test other variants

---

**Last Updated**: 2025-01-10
**Tested**: Pending (awaiting user verification)
