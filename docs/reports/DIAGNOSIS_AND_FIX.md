# StegaPoison Implementation Diagnosis & Fix

**Date:** August 8, 2026  
**Status:** 🔴 CRITICAL BUGS IDENTIFIED

---

## 🔍 Root Cause Analysis

### Critical Bug #1: Incorrect Delta Computation ⚠️

**Location:** `code/attacker/stegapoison.py` lines 89-90, 102

**Current Implementation:**
```python
# Line 89-90: Store initial params ONCE
if self.initial_item_params is None:
    self.initial_item_params = server_model.item_model.item_embedding.weight.clone().detach()

# Line 102: Always train from initial params (round 0)
local_item_params = self.initial_item_params.clone()

# Line 126: Compute delta from round 0, not current round
delta_r = theta_r - self.initial_item_params  # ❌ WRONG!
```

**The Problem:**
- `self.initial_item_params` is set once in round 1 and NEVER UPDATED
- Every subsequent round computes: Δ^(r) = Θ^(r) - Θ^(0)
- This means the attack is always computing updates relative to the **ancient round-0 model**
- As training progresses and the server model evolves, these updates become increasingly irrelevant

**Why This Causes Attack Decay:**
1. Round 200: Server model close to Θ^(0) → Attack relevant → HR@5 = 0.00610 ✓
2. Round 2000: Server model moved away → Attack still somewhat relevant → HR@5 = 0.00610 ✓
3. Round 6000: Server model far from Θ^(0) → Attack obsolete → HR@5 = 0.02335 ✗

**What The Paper Actually Says (Algorithm 1):**

> "**Step 1: Local Training**  
> Train on D_i to get Θ^(r)"

This means:
- Start with CURRENT server model Θ^(r-1)
- Train locally to get Θ^(r)
- Compute Δ^(r) = Θ^(r) - Θ^(r-1)  ← Delta from CURRENT model, not initial

**Correct Implementation Should Be:**
```python
# In prepare():
# Use CURRENT server model as baseline (updated every round)
current_item_params = server_model.item_model.item_embedding.weight.clone().detach()

# Train from current model
local_item_params = current_item_params.clone()
server_model.item_model.item_embedding.weight.data.copy_(local_item_params)

# ... training ...

# Compute delta from CURRENT model
delta_r = theta_r - current_item_params  # ✅ CORRECT!
```

---

### Critical Bug #2: No Momentum Accumulation

**Location:** Lines 75-76

**Current Implementation:**
```python
# Store per-client velocity for Step 5
self.client_velocity = {}
```

**The Problem:**
- `client_velocity` is initialized but NEVER USED
- Line 224 computes velocity fresh every time: `velocity = self.base_delta_mis + delta_i_r`
- No momentum accumulation across rounds
- The MOMENTUM parameter (0.8) is completely ignored

**Impact:**
- Without momentum, the attack doesn't build up directional consistency
- Each round's attack is independent, not building on previous rounds
- This reduces attack potency and makes it easier for defenses to filter

**Fix:**
```python
# In update(), line 224:
if client_id not in self.client_velocity:
    self.client_velocity[client_id] = torch.zeros_like(self.base_delta_mis)

# Apply momentum
momentum = getattr(self.args, 'MOMENTUM', 0.8)
self.client_velocity[client_id] = (
    momentum * self.client_velocity[client_id] + 
    (1 - momentum) * self.base_delta_mis
)

velocity = self.client_velocity[client_id] + delta_i_r
```

---

### Critical Bug #3: Statistical Invisibility Too Aggressive

**Location:** Line 246

**Current Implementation:**
```python
if delta_vel_norm > self.eta_stat * theta_r_norm:
    scale_factor = (self.eta_stat * theta_r_norm) / (delta_vel_norm + 1e-10)
    delta_scaled = scale_factor * delta_vel
```

**The Problem:**
- `eta_stat = 0.3` (INVISIBILITY_FACTOR)
- This clips the attack to 30% of model norm
- At round 6000, model norm is very large → clipping is severe
- Attack gets scaled down to almost nothing

**From Your Results:**
- Attacker gradient norm: 10.098 (constant across all rounds)
- This suggests the attack IS being clipped to a fixed magnitude
- But benign norms decrease over time (0.0004 average)
- The 23,278x ratio shows clipping isn't working as intended for stealth

**Paper's Intent:**
The statistical invisibility should keep attacks WITHIN benign distribution, not clip to a fixed norm.

**Fix:**
```python
# Instead of clipping to model norm, clip to benign gradient statistics
# Compute target based on benign gradient distribution
benign_mean_norm = compute_benign_mean_norm()  # Track this
benign_std_norm = compute_benign_std_norm()    # Track this

# Stealth factor (1.5 from paper) means "stay within 1.5 standard deviations"
stealth_factor = getattr(self.args, 'STEALTH_FACTOR', 1.5)
max_norm = benign_mean_norm + stealth_factor * benign_std_norm

if delta_vel_norm > max_norm:
    scale_factor = max_norm / (delta_vel_norm + 1e-10)
    delta_scaled = scale_factor * delta_vel
```

---

### Issue #4: Base Scale Too High

**Location:** Line 256

**Current Implementation:**
```python
attacker_item_grad = -delta_scaled.reshape(-1) * self.base_scale  # base_scale = 3.0
```

**The Problem:**
- After all the stealth mechanisms, a 3x amplification might be too aggressive
- Combined with incorrect delta computation, this amplifies the wrong signal
- May be contributing to the 23,278x gradient magnitude ratio

**Recommendation:**
- Try `base_scale = 1.0` first (no amplification)
- Let the attack mechanics work without artificial boosting
- Paper doesn't mention explicit gradient amplification factor

---

## 📋 Summary of Issues

| Bug | Severity | Impact | Status |
|-----|----------|--------|--------|
| Incorrect delta baseline (Θ^(0) vs Θ^(r-1)) | 🔴 CRITICAL | Attack decay over time | Root cause |
| No momentum accumulation | 🔴 CRITICAL | Reduced attack consistency | Major |
| Statistical invisibility miscalibrated | 🟡 HIGH | Poor stealth vs effectiveness tradeoff | Significant |
| Base scale too high | 🟡 MEDIUM | May amplify wrong signal | Minor |

---

## 🔧 Comprehensive Fix

### Fix 1: Correct Delta Computation

**File:** `code/attacker/stegapoison.py`

**Replace lines 89-127:**

```python
def prepare(self, server_model, step, *args):
    """
    Prepare attack: Execute Steps 1-4 to compute base malicious direction
    """
    # === Step 1: Local Training and Δ^(r) Computation ===
    # CRITICAL FIX: Use CURRENT server model as baseline, not initial model
    current_item_params = server_model.item_model.item_embedding.weight.clone().detach()
    
    sample_attacker = random.sample(
        self.attacker_id_list,
        k=min(self.args.ATTACKER_SAMPLE_NUM, len(self.attacker_id_list))
    )
    
    # Collect Δ^(r) from each sampled attacker
    delta_r_list = []
    
    for uid in sample_attacker:
        # Clone CURRENT model for local training (not initial model!)
        local_item_params = current_item_params.clone()
        server_model.item_model.item_embedding.weight.data.copy_(local_item_params)
        
        # Local training
        optimizer = optim.Adam(server_model.parameters(), lr=self.args.LR)
        client_data = self.attacker_user_data[uid]
        client_dataset = TrainDataset(uid, client_data)
        client_dataloader = DataLoader(
            client_dataset, shuffle=True, batch_size=self.args.BATCH_SIZE
        )
        
        for uid_batch, iid in client_dataloader:
            uid_batch = uid_batch.to(self.device, non_blocking=True)
            iid = iid.to(self.device, non_blocking=True)
            
            _, bz_loss = server_model(uid_batch, iid)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
        
        # Get trained parameters Θ^(r)
        theta_r = server_model.item_model.item_embedding.weight.clone().detach()
        
        # CRITICAL FIX: Compute Δ^(r) = Θ^(r) - Θ^(current), NOT Θ^(r) - Θ^(0)
        delta_r = theta_r - current_item_params  # ✅ Fixed!
        delta_r_list.append(delta_r)
    
    # Continue with rest of algorithm...
    # (Watermarking, LVDEP, Mirror Shift remain the same)
```

### Fix 2: Add Momentum Accumulation

**Replace lines 221-236:**

```python
with torch.no_grad():
    # === Step 5: Velocity-Based Sampling with Momentum ===
    # Initialize client velocity if first time
    if client_id not in self.client_velocity:
        self.client_velocity[client_id] = torch.zeros_like(self.base_delta_mis)
    
    # Apply momentum: v_t = α * v_{t-1} + (1-α) * Δ^(mis)
    momentum = getattr(self.args, 'MOMENTUM', 0.8)
    self.client_velocity[client_id] = (
        momentum * self.client_velocity[client_id] + 
        (1 - momentum) * self.base_delta_mis
    )
    
    # Compute velocity: v ← momentum_velocity + Δ_i^(r)
    velocity = self.client_velocity[client_id] + delta_i_r
    
    # Compute velocity norm
    v_norm = torch.norm(velocity)
    
    # Threshold-based noise injection
    if v_norm > self.tau_vel:
        noise = torch.randn_like(self.base_delta_mis) * self.eta_noise
        delta_vel = self.client_velocity[client_id] + noise
    else:
        delta_vel = self.client_velocity[client_id]
```

### Fix 3: Improved Statistical Invisibility

**Replace lines 238-252:**

```python
# === Step 6: Statistical Invisibility (Improved) ===
# Compute delta norm
delta_vel_norm = torch.norm(delta_vel)

# Use stealth factor to stay within benign distribution
# stealth_factor = 1.5 means "stay within typical benign variance"
stealth_factor = getattr(self.args, 'STEALTH_FACTOR', 1.5)

# Get model norm as reference scale
theta_r_norm = torch.norm(server_model.item_model.item_embedding.weight)

# Target: Keep update proportional to model scale but bounded by stealth
# Use eta_stat as base ratio, modulated by stealth factor
max_allowed_norm = self.eta_stat * theta_r_norm * stealth_factor

# Conditional scaling
if delta_vel_norm > max_allowed_norm:
    scale_factor = max_allowed_norm / (delta_vel_norm + 1e-10)
    delta_scaled = scale_factor * delta_vel
else:
    delta_scaled = delta_vel

# Convert from parameter update to gradient
# REDUCED base_scale from 3.0 to 1.0 to avoid over-amplification
base_scale = getattr(self.args, 'SCALE', 1.0)  # Changed from 3.0
attacker_item_grad = -delta_scaled.reshape(-1) * base_scale
```

### Fix 4: Remove Obsolete Code

**Remove lines 89-91:**
```python
# DELETE THIS:
# if self.initial_item_params is None:
#     self.initial_item_params = server_model.item_model.item_embedding.weight.clone().detach()
```

**Remove line 165:**
```python
# DELETE THIS:
# server_model.item_model.item_embedding.weight.data.copy_(self.initial_item_params)
```

---

## 🚀 Complete Fixed Version

I'll create a complete fixed version of the attacker file:

```python
# See stegapoison_fixed_v2.py (to be created)
```

---

## 📊 Expected Improvements

After applying these fixes:

### Attack Effectiveness:
- **Round 2000:** Should remain strong (HR@5 ≈ 0.00610 or better)
- **Round 6000:** Should ALSO be strong (HR@5 ≈ 0.00209, matching paper)
- **No more decay:** Attack maintains effectiveness across all rounds

### Stealth:
- **Gradient magnitude ratio:** Should decrease from 23,278x to ~100-1000x
- **Cosine similarity:** Should remain near-zero (already good)
- **Defense bypass:** Should successfully evade NormBound, Krum, etc.

### Paper Alignment:
- **HR@5 at 6000 rounds:** Target 0.00209 (paper's result)
- **nDCG@5 at 6000 rounds:** Target 0.00123 (paper's result)

---

## 🧪 Testing Plan

1. **Create fixed version:**
   ```bash
   cp code/attacker/stegapoison.py code/attacker/stegapoison_original_v1.py
   # Apply fixes to stegapoison.py
   ```

2. **Run short test (200 rounds):**
   ```bash
   python code/train.py --EXP_NAME test_fixed_ml_MF_stegapoison_FedAdam \
     --MODEL_TYPE MF --DATA ml --SEED 0 \
     --AGG_TYPE FedAdam --ATTACKER_RATIO 0.05 \
     --ATTACKER_STRAT StegaPoison --MAX_ROUND 200 \
     --LR 2e-3 --SCALE 1.0
   ```

3. **Verify no decay:**
   - Check HR@5 stays low across all checkpoints
   - Confirm gradient statistics are reasonable

4. **Full 6000-round run:**
   ```bash
   python code/train.py --EXP_NAME train6000_ml_MF_stegapoison_FedAdam_FIXED_V2 \
     --MODEL_TYPE MF --DATA ml --SEED 0 \
     --AGG_TYPE FedAdam --ATTACKER_RATIO 0.05 \
     --ATTACKER_STRAT StegaPoison --MAX_ROUND 6000 \
     --LR 2e-3 --SCALE 1.0 --SAVE_ROUND 200
   ```

5. **Compare results:**
   - HR@5 should be ≈ 0.00209 at round 6000
   - Attack should not decay
   - Stealth metrics should improve

---

## 💡 Additional Recommendations

### Hyperparameter Tuning:
After fixing the core bugs, tune these:

1. **SCALE:** Try 0.5, 1.0, 1.5, 2.0 (not 3.0)
2. **INVISIBILITY_FACTOR:** Try 0.2, 0.25, 0.3
3. **STEALTH_FACTOR:** Keep at 1.5 (paper value)
4. **MOMENTUM:** Keep at 0.8 (paper value)

### Defense Testing:
Once attack works consistently:
1. Test against all defenses (NormBound, Krum, MultiKrum, etc.)
2. Verify attack bypasses norm-based defenses
3. Validate Figure 3 claims from paper

### Multi-Seed Validation:
Run with seeds 0, 1, 2, 3, 4 and report mean ± std

---

## 📝 Summary

**The root cause of your attack decay is:**
> Computing Δ^(r) = Θ^(r) - Θ^(0) instead of Δ^(r) = Θ^(r) - Θ^(current)

**This single bug explains:**
- ✅ Why attack works at round 2000 (model still close to Θ^(0))
- ✅ Why attack fails at round 6000 (model far from Θ^(0))  
- ✅ Why your 2000-round results are closer to paper than 6000-round
- ✅ Why defenses don't help at round 6000 (attack already weak)

**After fixing this + adding momentum + calibrating invisibility:**
- Attack should maintain effectiveness across all rounds
- HR@5 should reach ≈0.00209 at round 6000 (matching paper)
- No more decay phenomenon

Would you like me to create the complete fixed version of the file?
