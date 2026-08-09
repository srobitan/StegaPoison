# StegaPoison Implementation Discrepancy Report

**Date**: 2026-08-08  
**Analysis**: Comparison of Paper Algorithm 1 vs Current Codebase

---

## Executive Summary

The current implementation achieves **significantly weaker attack performance** than reported in the paper:

| Dataset | Model | Paper HR@5 | Current HR@5 | Difference |
|---------|-------|------------|--------------|------------|
| ML-1M   | MF    | 0.00209    | 0.02335      | **11.2x weaker** |
| ML-1M   | SASRec| 0.00558    | TBD          | TBD |
| Gowalla | MF    | 0.00014    | TBD          | TBD |
| Gowalla | SASRec| 0.00769    | TBD          | TBD |

**Root Cause**: The implementation deviates from Algorithm 1 in 8 critical ways.

---

## Critical Discrepancies

### 1. ❌ Missing Step 1: Δ^(r) Computation

**Paper Algorithm 1, Step 1:**
```
Train on D_i to get Θ^(r)
Δ^(r) ← Θ^(r) - Θ^(0)
```

**Current Implementation (lines 85-96):**
```python
for uid_batch, iid in client_dataloader:
    _, bz_loss = server_model(uid_batch, iid)
    optimizer.zero_grad()
    bz_loss.backward()
    batch_sample_num = len(uid_batch)
    client_gradient_vec += grad_to_vector(server_model) * (
        batch_sample_num / client_sample_num
    )
```

**Issue**: 
- Code computes **gradients** via `grad_to_vector()`
- **Never computes Δ^(r)** as parameter difference
- Algorithm requires parameter updates, not gradients

**Fix**: Store initial parameters, perform local training, compute Δ^(r) = Θ^(r) - Θ^(0)

---

### 2. ❌ Watermarking Formula Mismatch (Step 2)

**Paper Algorithm 1, Step 2:**
```
w_j = γ * sin(2πωj)
```

**Current Implementation (lines 43-50):**
```python
a = torch.randn(self.args.EMBDIM, generator=self.watermark_gen, device=self.device) * self.watermark_scale
freq_axis = torch.arange(1, self.args.EMBDIM + 1, device=self.device, dtype=torch.float32) / self.args.EMBDIM
omega = getattr(args, 'K', 15.0)
self.watermark_pattern = a * torch.sin(2.0 * math.pi * omega * freq_axis)
```

**Issue**:
- Code uses `w_j = a_j * sin(2π * ω * f_j)` where:
  - `a_j ~ N(0, σ_w²)` is **random Gaussian noise**
  - `f_j = j/d` is **normalized frequency**
- Paper uses simple `w_j = γ * sin(2πωj)` (deterministic, j is dimension index)
- Extra random amplitude modulation not in paper

**Fix**: Use `w_j = γ * sin(2πωj)` directly without random modulation

---

### 3. ❌ LVDEP Implementation Incorrect (Step 3)

**Paper Algorithm 1, Step 3:**
```
Identify low-variance dims: L = {j | Var_j(Δ^(r)) < τ_lvdep}
For j ∉ L: Δ_j^(lvdep) = Δ_j^(mm) + ε_j
```

**Current Implementation (lines 224-238):**
```python
# Lines 225-226: Adjacent difference, not variance
shifted_grad = torch.roll(active_grad, shifts=-1, dims=0)
grad_diff = torch.abs(active_grad - shifted_grad)

# Line 229: Masking based on adjacent difference
vadp_mask = (grad_diff < self.vadp_threshold).float()
```

**Issue**:
- Uses **adjacent difference** `|g_j - g_{j+1}|` on **single client**
- Paper requires **variance across multiple clients**: `Var_j(Δ^(r))`
- Only applied to **user embeddings** (lines 214-238), not item embeddings
- The mask logic identifies smooth regions, not low-variance dimensions

**Fix**: Compute variance across all sampled attackers' Δ^(r), identify dims where Var_j < τ

---

### 4. ❌ Mirror Shift Formula Wrong (Step 4)

**Paper Algorithm 1, Step 4:**
```
d ← Δ^(lvdep)
Δ^(mis) ← Δ^(lvdep) + βd
```
Simplifies to: `Δ^(mis) = (1 + β) * Δ^(lvdep)`

**Current Implementation (lines 119-149):**
```python
attack_mode = getattr(self.args, 'ATTACK_MODE', 'hybrid')

if attack_mode == 'reverse':
    update_direction = -(1.0 + mirror_beta) * benign_update_direction
elif attack_mode == 'collapse':
    global_centroid = benign_update_direction.mean(dim=0, keepdim=True)
    update_direction = (global_centroid - benign_update_direction) * (1.0 + mirror_beta)
elif attack_mode == 'noise':
    update_direction = torch.randn_like(benign_update_direction) * self.item_grad_norm_mean
elif attack_mode == 'similarity':
    shuffled_indices = torch.randperm(benign_update_direction.size(0))
    update_direction = benign_update_direction[shuffled_indices] * -(1.0 + mirror_beta)
else: # 'hybrid' (default)
    reversal = -benign_update_direction
    centroid = benign_update_direction.mean(dim=0, keepdim=True)
    collapse = centroid - benign_update_direction
    update_direction = (0.75 * reversal + 0.25 * collapse) * (1.0 + mirror_beta)

# Lines 147-149: Extra perturbation noise
noise_variance = getattr(self.args, 'NOISE_VARIANCE', 0.35)
perturbation = torch.randn_like(update_direction) * noise_variance
update_direction += perturbation
```

**Issue**:
- Has **5 different attack modes** - none matching the paper
- Default "hybrid" mode combines reversal and collapse: `(0.75 * reversal + 0.25 * collapse) * (1 + β)`
- Adds extra perturbation noise (lines 147-149) not in algorithm
- **Completely different approach** from simple directional amplification

**Fix**: Use `Δ^(mis) = (1 + β) * Δ^(lvdep)` as per paper

---

### 5. ❌ Velocity-Based Sampling Misimplemented (Step 5)

**Paper Algorithm 1, Step 5:**
```
v ← Δ^(mis) + Δ_i^(r)
if ||v|| > τ_vel then
    Δ^(vel) ← Δ^(mis) + η·N(0, I)
else
    Δ^(vel) ← Δ^(mis)
end if
```

**Current Implementation (lines 151-158):**
```python
momentum = getattr(self.args, 'MOMENTUM', 0.8)
if not hasattr(self, 'momentum_buffer'):
    self.momentum_buffer = torch.zeros_like(update_direction)

# Update momentum with the new direction
self.momentum_buffer = momentum * self.momentum_buffer + update_direction
```

**Issue**:
- Uses **momentum buffer**: `momentum_buffer = μ * momentum_buffer + update_direction`
- **No threshold check** `||v|| > τ_vel` - always applies momentum
- **No per-client velocity** - uses global momentum buffer
- Missing the conditional noise injection based on velocity magnitude

**Fix**: Compute per-client velocity, check threshold, conditionally add noise

---

### 6. ❌ Statistical Invisibility Formula Incorrect (Step 6)

**Paper Algorithm 1, Step 6:**
```
if ||Δ^(vel)|| > η||Θ^(r)|| then
    Δ^(scaled) ← (η||Θ^(r)|| / ||Δ^(vel)||) * Δ^(vel)
else
    Δ^(scaled) ← Δ^(vel)
end if
```

**Current Implementation (lines 160-171):**
```python
# Lines 164-165: Uses gradient statistics, not model parameters
stealth_k = getattr(self.args, 'STEALTH_FACTOR', 1.5)
target_norm = self.item_grad_norm_mean + stealth_k * self.item_grad_norm_std

# Line 168: Calculates buffer norm
buffer_norm = (self.momentum_buffer ** 2).sum(dim=-1, keepdim=True).sqrt()

# Line 171: Always scales (no conditional)
self.total_item_grad = self.momentum_buffer * (target_norm / (buffer_norm + 1e-10))
```

**Issue**:
- Uses `target_norm = mean + k * std` from **gradient statistics**, not `η||Θ^(r)||`
- **Always scales to target norm** (line 171), not conditional on threshold
- Reference is gradient distribution, not model parameter norm

**Fix**: Use model parameter norm, conditional scaling as per paper

---

### 7. ❌ Missing Item Gradient LVDEP

**Paper Context**: LVDEP should apply to item embeddings primarily (the attack target)

**Current Implementation**: LVDEP (lines 214-238) **only applied to user embeddings**

**Issue**:
- User gradients get watermark + LVDEP treatment
- Item gradients get the pre-computed `total_item_grad` without LVDEP processing
- Backwards from paper's intent

**Fix**: Apply LVDEP to item embeddings, not user embeddings

---

### 8. ❌ No Step-by-Step Sequential Pipeline

**Paper Algorithm 1**: Clear 6-step sequential pipeline:
```
1. Local Training → 
2. Watermark → 
3. LVDEP → 
4. Mirror → 
5. Velocity → 
6. Invisibility
```

**Current Implementation**: 
- Components are scattered and don't follow sequential structure
- `prepare()` method computes item gradients globally
- `update()` applies different logic to user gradients
- No clear data flow through Steps 1-6

**Fix**: Restructure to follow sequential pipeline

---

## Recommended Fixes

### Priority 1: Core Algorithm Alignment

1. **Implement Δ^(r) = Θ^(r) - Θ^(0)** computation (Step 1)
2. **Fix watermarking** to `w_j = γ * sin(2πωj)` (Step 2)
3. **Fix LVDEP** to use variance across clients (Step 3)
4. **Simplify Mirror Shift** to `(1 + β) * Δ^(lvdep)` (Step 4)
5. **Implement threshold-based velocity** sampling (Step 5)
6. **Fix Statistical Invisibility** to use model parameter norm (Step 6)

### Priority 2: Target Correction

- Apply LVDEP to **item embeddings**, not user embeddings
- Ensure attack primarily targets item embedding space

### Priority 3: Restructure

- Follow sequential pipeline: 1 → 2 → 3 → 4 → 5 → 6
- Ensure each step's output feeds into the next

---

## Testing Plan

After fixes, expected results:

| Dataset | Model | Defense | Expected HR@5 | Expected nDCG@5 |
|---------|-------|---------|---------------|-----------------|
| ML-1M   | MF    | FedAdam | 0.00209       | 0.00123         |
| ML-1M   | SASRec| FedAdam | 0.00558       | 0.00339         |
| Gowalla | MF    | FedAdam | 0.00014       | 0.00010         |
| Gowalla | SASRec| FedAdam | 0.00769       | 0.00556         |

Run evaluation:
```bash
python3 eval_all.py --DATA ml gowalla --MODEL_TYPE MF SASRec --AGG_TYPE FedAdam
```

---

## Conclusion

The current implementation is an **evolved/experimental version** with added complexity (multiple attack modes, momentum) that deviates from the paper's mathematical formulation. These deviations explain the **~10x weaker attack performance** observed.

A corrected implementation following Algorithm 1 strictly should reproduce the paper's results.
