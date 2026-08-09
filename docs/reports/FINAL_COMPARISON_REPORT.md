# StegaPoison: Codebase vs Paper Results Comparison

**Date:** August 8, 2026  
**Experiment:** `train2000_ml_MF_stegapoison_FedAdam_FIXED/seed0`  
**Paper:** StegaPoison_Elsevier - Last updated 31.07.2026.pdf

---

## Executive Summary

This report compares experimental results from the StegaPoison codebase implementation against the published research paper. The comparison reveals **significant discrepancies** between expected and actual attack performance, raising questions about implementation fidelity and experimental reproducibility.

### Key Finding: ⚠️ **MAJOR PERFORMANCE DISCREPANCY DETECTED**

---

## 1. Expected Results from Paper (Table 3)

### MovieLens-1M Dataset - Matrix Factorization (MF) Model - No Defense

| Metric | Baseline (No Attack) | StegaPoison (Paper) | Degradation |
|--------|---------------------|---------------------|-------------|
| **HR@5** | 0.03549 | **0.00209** | **94.11%** |
| **nDCG@5** | 0.02226 | **0.00123** | **94.46%** |

**Paper's Claim:** StegaPoison achieves severe performance degradation (~94%) on the recommendation model.

---

## 2. Actual Results from Codebase

### MovieLens-1M MF - StegaPoison with FedAdam - Round 2000

| Metric | Our Results (Test Set) | Expected (Paper) | Discrepancy |
|--------|------------------------|------------------|-------------|
| **HR@5** | **0.00610** | 0.00209 | **2.92x HIGHER** |
| **nDCG@5** | **0.00317** | 0.00123 | **2.58x HIGHER** |
| **HR@10** | 0.00959 | N/A | - |
| **nDCG@10** | 0.00431 | N/A | - |
| **HR@20** | 0.01760 | N/A | - |
| **nDCG@20** | 0.00633 | N/A | - |

### Validation Set Results (Round 2000)

| Metric | Value |
|--------|-------|
| **HR@5** | 0.00523 |
| **nDCG@5** | 0.00268 |
| **HR@10** | 0.01028 |
| **nDCG@10** | 0.00430 |

---

## 3. Performance Trajectory Analysis

### Test Set HR@5 Progression

| Round | HR@5 | nDCG@5 |
|-------|------|--------|
| 200 | 0.00662 | 0.00358 |
| 400 | 0.00662 | 0.00357 |
| 600 | 0.00610 | 0.00311 |
| 800 | 0.00610 | 0.00322 |
| 1000 | 0.00645 | 0.00344 |
| 1200 | 0.00645 | 0.00334 |
| 1400 | 0.00645 | 0.00323 |
| 1600 | 0.00610 | 0.00312 |
| 1800 | 0.00662 | 0.00343 |
| **2000** | **0.00610** | **0.00317** |

**Observation:** Performance remains relatively stable across training rounds, fluctuating between HR@5 of 0.00610-0.00662, rather than showing continuous degradation.

---

## 4. Critical Analysis

### 4.1 Performance Discrepancy

**Finding:** Our implementation achieves **2.92x weaker attack effectiveness** than the paper reports.

**Implications:**
- Paper claims: HR@5 = 0.00209 (99.4% of users fail to find relevant items in top-5)
- Our results: HR@5 = 0.00610 (98.3% of users fail to find relevant items in top-5)
- Both show severe degradation, but paper's results are significantly more destructive

**Possible Explanations:**

1. **Different Training Rounds:** 
   - Paper experiments use 6,000 rounds (Table 2, experimental setting)
   - Our experiment stopped at 2,000 rounds
   - Attack may strengthen with continued training

2. **Baseline Difference:**
   - Missing baseline (no attack) comparison for this specific run
   - Cannot calculate actual degradation percentage
   - Need to train a clean model to establish baseline

3. **Hyperparameter Differences:**
   - Paper uses learning rate 2e-3 for ML-1M MF
   - Code default is 1e-3 (may have been overridden at runtime)
   - Other hyperparameters may differ

4. **Random Seed Effects:**
   - Paper reports mean across 5 independent runs
   - We only have results from seed0
   - Natural variance could account for some discrepancy

5. **Implementation Details:**
   - Stealth parameter interpretations
   - Attack component weights
   - Statistical invisibility calculations

### 4.2 Stealth Analysis

**Gradient Statistics from Training:**

| Metric | Mean | Observation |
|--------|------|-------------|
| Attacker Gradient Norm | 10.098301 | Extremely stable (std = 0.000000) |
| Benign Gradient Norm | 0.000434 | Variable (std = 0.000196) |
| **Norm Ratio** | **23,278x** | **Attacker >> Benign** |
| Mean Cosine Similarity | 0.000737 | Near-orthogonal (good stealth) |

**Stealth Assessment:**

✓ **Directional Concealment:** ACHIEVED
- Cosine similarity ≈ 0.0007 (near-zero)
- Attacker updates are orthogonal to benign updates
- Aligns with paper's "steganographic concealment" claim

✗ **Magnitude Concealment:** FAILED
- Attacker norms are 23,278x larger than benign norms
- This should be **easily detectable** by norm-based defenses
- Contradicts paper's claim of bypassing NormBound defense

⚠️ **Critical Question:** 
Are these norms measured before or after defense mechanisms? If post-defense, this indicates a serious stealth failure. If pre-defense, we need to verify that NormBound clipping is actually applied.

---

## 5. Comparison with Paper's Baseline Attacks (Table 3)

**Paper's Reported Results (ML-1M MF, No Defense):**

| Attack Method | HR@5 | nDCG@5 | Degradation |
|---------------|------|--------|-------------|
| No Attack | 0.03549 | 0.02226 | 0% (baseline) |
| LabelFlip | 0.03561 | 0.02238 | -0.34% (ineffective) |
| FedAttack | 0.03358 | 0.02118 | 5.38% |
| Gaussian | 0.03555 | 0.02224 | -0.17% (ineffective) |
| LIE | 0.03259 | 0.02062 | 8.17% |
| Fang | 0.03038 | 0.01897 | 14.40% |
| ClusterAttack | 0.02451 | 0.01545 | **30.94%** |
| **StegaPoison (Paper)** | **0.00209** | **0.00123** | **94.11%** 🔥 |

**Our Implementation:**
| Attack Method | HR@5 | nDCG@5 | Notes |
|---------------|------|--------|-------|
| **StegaPoison (Ours)** | **0.00610** | **0.00317** | Still severe, but 2.9x weaker |

**Analysis:**
- Even with the discrepancy, our implementation still **dramatically outperforms** all baseline attacks
- Our HR@5 of 0.00610 is still **4x more effective** than ClusterAttack (0.02451)
- The attack is working, just not as strongly as the paper reports

---

## 6. Missing Elements for Full Validation

### 6.1 Baseline Comparison ❌

**Critical Gap:** No clean baseline (no attack) results for this specific configuration.

**Required Action:**
```bash
python code/train.py --EXP_NAME baseline_ml_MF_FedAdam \
  --MODEL_TYPE MF --DATA ml --SEED 0 \
  --AGG_TYPE FedAdam --ATTACKER_RATIO 0 \
  --MAX_ROUND 2000 --LR 2e-3
```

**Expected Baseline (from paper):**
- HR@5: 0.03549
- nDCG@5: 0.02226

**Degradation Calculation:**
- Paper: (0.03549 - 0.00209) / 0.03549 = 94.11%
- Ours: (0.03549 - 0.00610) / 0.03549 = **82.81%** (if baseline matches)

### 6.2 Extended Training ❌

**Issue:** Paper uses 6,000 rounds, we only trained 2,000 rounds.

**Impact:** Attack effectiveness may increase with continued training.

**Required Action:**
```bash
# Resume training from round 2000 → 6000
python code/train.py --EXP_NAME train2000_ml_MF_stegapoison_FedAdam_FIXED \
  --MODEL_TYPE MF --DATA ml --SEED 0 \
  --AGG_TYPE FedAdam --ATTACKER_RATIO 0.05 \
  --ATTACKER_STRAT StegaPoison --MAX_ROUND 6000 --LR 2e-3
```

### 6.3 Defense Robustness Testing ❌

**Paper Claims (Figure 3):** StegaPoison remains effective even under defenses.

**Missing Experiments:**
- TrimmedMean defense
- Krum defense
- MultiKrum defense
- NormBound defense (critical given magnitude discrepancy)
- FL-WBC defense
- MultiKrum+UNION defense
- NormBound+UNION defense

**Required Actions:**
Run experiments with all AGG_TYPE values to validate Figure 3.

### 6.4 Multi-Seed Validation ❌

**Paper Method:** Reports "mean results across 5 independent random seeds"

**Current State:** Only seed0 results available

**Required Actions:**
```bash
for seed in 1 2 3 4; do
  python code/train.py --EXP_NAME train6000_ml_MF_stegapoison_FedAdam \
    --SEED $seed --MAX_ROUND 6000 --LR 2e-3 \
    --MODEL_TYPE MF --DATA ml --AGG_TYPE FedAdam \
    --ATTACKER_RATIO 0.05 --ATTACKER_STRAT StegaPoison
done
```

### 6.5 Ablation Study ❌

**Paper (Figure 6):** Tests 8 configurations to validate component synergy.

**Missing:** Systematic ablation disabling:
- Watermarking
- LVDEP (Low-Variance Dimension Perturbation)
- Mirror Shift
- Velocity-Based Sampling
- Momentum
- Statistical Invisibility

---

## 7. Hyperparameter Verification

### 7.1 Paper's Configuration (Table 1)

**ML-1M MF Hyperparameters:**
- Learning rate (η): **2e-3**
- ℓ₂ regularization (λ): **1e-5**
- L2 norm bound (NormBound): **0.1**
- StegaPoison ratio: **0.05** (5% malicious clients)
- StegaPoison stealth: **1.5**
- Embedding dimension: **64**
- Batch size: **512**
- User sample per round: **50**
- Max rounds: **6,000**

### 7.2 Code Default Configuration (train.py)

```python
parser.add_argument("--LR", type=float, default=1e-3)  # ❌ Different from paper
parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-5)  # ✓ Matches
parser.add_argument("--NORM_BOUND", type=int, default=0.1)  # ✓ Matches
parser.add_argument("--ATTACKER_RATIO", type=float, default=0)  # Set via args
parser.add_argument("--STEALTH_FACTOR", type=float, default=1.5)  # ✓ Matches
parser.add_argument("--EMBDIM", type=int, default=64)  # ✓ Matches
parser.add_argument("--BATCH_SIZE", type=int, default=512)  # ✓ Matches
parser.add_argument("--USER_SAMPLE_NUM", type=int, default=50)  # ✓ Matches
parser.add_argument("--MAX_ROUND", type=int, default=6000)  # Used 2000 in experiment
```

**Critical Finding:** Default learning rate is **1e-3** but paper uses **2e-3**. This could significantly impact convergence and attack effectiveness.

---

## 8. Detailed Training Round Analysis

### Complete Test Results Across All Checkpoints

| Round | HR@5 | nDCG@5 | HR@10 | nDCG@10 | HR@20 | nDCG@20 |
|-------|------|--------|-------|---------|-------|---------|
| 200 | 0.00662 | 0.00358 | 0.01063 | 0.00486 | 0.01917 | 0.00691 |
| 400 | 0.00662 | 0.00357 | 0.01028 | 0.00481 | 0.01882 | 0.00686 |
| 600 | 0.00610 | 0.00311 | 0.00906 | 0.00417 | 0.01691 | 0.00610 |
| 800 | 0.00610 | 0.00322 | 0.00976 | 0.00440 | 0.01830 | 0.00648 |
| 1000 | 0.00645 | 0.00344 | 0.00906 | 0.00431 | 0.01743 | 0.00638 |
| 1200 | 0.00645 | 0.00334 | 0.00976 | 0.00443 | 0.01882 | 0.00670 |
| 1400 | 0.00645 | 0.00323 | 0.00993 | 0.00437 | 0.01830 | 0.00647 |
| 1600 | 0.00610 | 0.00312 | 0.00941 | 0.00422 | 0.01743 | 0.00622 |
| 1800 | 0.00662 | 0.00343 | 0.01011 | 0.00459 | 0.01795 | 0.00656 |
| **2000** | **0.00610** | **0.00317** | **0.00959** | **0.00431** | **0.01760** | **0.00633** |

**Statistical Summary:**
- **HR@5 Range:** 0.00610 - 0.00662 (0.00052 spread)
- **HR@5 Mean:** 0.00636
- **HR@5 Std:** 0.000235
- **nDCG@5 Mean:** 0.00332

**Interpretation:**
1. Performance is **remarkably stable** after round 200
2. No clear downward or upward trend
3. Small fluctuations likely due to random sampling
4. Attack has reached a **plateau** by round 200

---

## 9. Recommendations

### 9.1 Immediate Actions (Priority 1)

1. **✅ COMPLETED:** Evaluation of round 2000 checkpoint
2. **⏳ TODO:** Train baseline (no attack) model for ground truth comparison
3. **⏳ TODO:** Continue training from round 2000 → 6000 to match paper
4. **⏳ TODO:** Verify learning rate parameter (should be 2e-3, not 1e-3)

### 9.2 Validation Actions (Priority 2)

1. **Multi-Seed Experiments:**
   - Run seeds 1, 2, 3, 4 (already have seed 0)
   - Calculate mean ± std for all metrics
   - Compare variance with paper's reported confidence

2. **Defense Robustness:**
   - Test all 7 defense mechanisms from paper
   - Reproduce Figure 3 (Defense Sensitivity Analysis)
   - Validate "bypasses norm-based defenses" claim

3. **Ablation Study:**
   - Disable attack components systematically
   - Reproduce Figure 6 (Ablation Study)
   - Validate component synergy claims

### 9.3 Investigation Actions (Priority 3)

1. **Gradient Magnitude Mystery:**
   - Investigate why attacker norms are 23,278x benign
   - Verify stealth_logs captures pre- or post-clipping values
   - Check if NormBound defense is correctly applied
   - Review Orchestra.update_one_round() implementation

2. **Performance Gap Analysis:**
   - Compare all hyperparameters against paper
   - Check for code version differences
   - Review recent commits for bug fixes
   - Contact paper authors if discrepancy persists

---

## 10. Conclusions

### 10.1 What We Know

✅ **Attack is Working:**
- HR@5 reduced to 0.00610 (severe degradation)
- Still 4x more effective than best baseline (ClusterAttack)
- Recommendation system is heavily poisoned

✅ **Directional Stealth Achieved:**
- Cosine similarity ≈ 0 (orthogonal updates)
- Matches paper's steganographic concealment claims

✅ **Stable Attack Behavior:**
- Performance plateaus quickly (by round 200)
- Consistent across 1800 additional rounds

### 10.2 What's Concerning

⚠️ **Performance Discrepancy:**
- 2.92x weaker attack than paper reports
- HR@5: 0.00610 (ours) vs 0.00209 (paper)
- Significant gap requiring investigation

❌ **Magnitude Stealth Failure:**
- Attacker gradients 23,278x larger than benign
- Contradicts "bypasses norm-based defenses" claim
- Should be easily detectable

❌ **Incomplete Validation:**
- Missing baseline comparison
- Only 2000/6000 rounds completed
- No defense robustness testing
- Single seed (need 5 for statistical validity)

### 10.3 Overall Assessment

**Status:** 🟡 **PARTIAL VALIDATION**

The StegaPoison attack implementation demonstrates **severe attack effectiveness** that dramatically outperforms baseline methods. However, **quantitative performance falls short** of the paper's reported results by approximately 3x. Several experimental gaps prevent full validation:

1. Training only 33% complete (2000/6000 rounds)
2. Missing baseline for degradation calculation
3. No defense mechanism testing
4. Single-seed results vs paper's 5-seed average
5. Potential hyperparameter mismatch (learning rate)

**Recommendation:** Continue experimental validation before drawing final conclusions. The attack clearly works, but full reproducibility requires completing the missing experimental components.

---

## 11. Next Steps Command Reference

### Train Baseline Model
```bash
cd /Users/apple/Downloads/StegaPoison/code
python train.py --EXP_NAME baseline_ml_MF_FedAdam_LR2e3 \
  --MODEL_TYPE MF --DATA ml --SEED 0 \
  --AGG_TYPE FedAdam --ATTACKER_RATIO 0 \
  --MAX_ROUND 6000 --LR 2e-3 --SAVE_ROUND 200
```

### Continue Attack Training to 6000 Rounds
```bash
cd /Users/apple/Downloads/StegaPoison/code
python train.py --EXP_NAME train6000_ml_MF_stegapoison_FedAdam_LR2e3 \
  --MODEL_TYPE MF --DATA ml --SEED 0 \
  --AGG_TYPE FedAdam --ATTACKER_RATIO 0.05 \
  --ATTACKER_STRAT StegaPoison --MAX_ROUND 6000 --LR 2e-3 --SAVE_ROUND 200
```

### Test with Defense Mechanisms
```bash
# NormBound defense
python train.py --EXP_NAME train6000_ml_MF_stegapoison_NormBound \
  --AGG_TYPE NormBound --ATTACKER_STRAT StegaPoison --MAX_ROUND 6000

# MultiKrum defense  
python train.py --EXP_NAME train6000_ml_MF_stegapoison_MultiKrum \
  --AGG_TYPE MultiKrum --ATTACKER_STRAT StegaPoison --MAX_ROUND 6000
```

---

**Report Compiled:** August 8, 2026  
**Analysis Type:** Experimental Validation & Reproducibility Assessment  
**Confidence Level:** High (based on available data)  
**Reproducibility Status:** Partial - Requires completion of validation experiments
