# StegaPoison Results Comparison Report

**Date:** August 8, 2026  
**Experiment:** `train2000_ml_MF_stegapoison_FedAdam_FIXED/seed0`  
**Paper:** StegaPoison_Elsevier - Last updated 31.07.2026.pdf

---

## 1. Expected Results from Paper (Table 3)

### MovieLens-1M Dataset - Matrix Factorization (MF) Model

#### Baseline Performance (No Attack):
- **HR@5:** 0.03549
- **nDCG@5:** 0.02226

#### StegaPoison Attack Performance (Ours):
- **HR@5:** 0.00209 (94.11% degradation)
- **nDCG@5:** 0.00123 (94.46% degradation)

---

## 2. Hyperparameters from Paper (Table 1)

**ML-1M MF Configuration:**
- Learning rate (η): **2e-3**
- ℓ₂ regularization (λ): **1e-5**
- L2 norm bound (NormBound): **0.1**
- StegaPoison ratio: **0.05** (5% malicious clients)
- StegaPoison stealth: **1.5**

---

## 3. Experimental Results Analysis

### 3.1 Stealth Metrics from Our Codebase

**Experiment Configuration:**
- Dataset: MovieLens-1M
- Model: Matrix Factorization (MF)
- Attack: StegaPoison
- Aggregation: FedAdam
- Training Rounds: 2000
- Seed: 0

**Stealth Statistics:**

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Attacker Gradient Norm | 10.098301 | 0.000000 | 10.098301 | 10.098302 |
| Benign Gradient Norm | 0.000434 | 0.000196 | 0.000112 | 0.004581 |
| Mean Cosine Similarity | 0.000737 | 0.002035 | -0.005941 | 0.007936 |

**Key Observations:**

1. **Gradient Magnitude Ratio:** 23,278x (Attacker/Benign)
   - This extremely high ratio suggests the attack is using very large gradient updates
   - Paper suggests stealth factor of 1.5, but actual implementation shows much larger norms

2. **Directional Stealth:** ✓ ACHIEVED
   - Mean cosine similarity: 0.000737 (near-zero)
   - Attacker updates are nearly orthogonal to benign updates
   - This aligns with paper's claim of steganographic concealment

3. **Statistical Invisibility:** ✗ QUESTIONABLE
   - While directional correlation is low, the magnitude difference is extreme
   - Norm-based defenses (e.g., NormBound, clipping) should easily detect this

---

## 4. Critical Findings

### 4.1 Stealth Assessment

**Paper Claims (Section 6.3):**
> "StegaPoison demonstrates significantly stronger stealth characteristics compared with existing untargeted poisoning attacks. Unlike conventional poisoning methods such as LIE, Fang, and ClusterAttack, which often introduce statistically abnormal or highly concentrated gradient patterns, StegaPoison carefully embeds adversarial perturbations within naturally smooth embedding regions..."

**Our Results:**
- ✓ **Directional Concealment:** Successfully achieved (cosine sim ≈ 0)
- ✗ **Magnitude Concealment:** Failed (23,278x larger than benign)
- ⚠ **Detection Risk:** HIGH - Norm-based defenses should detect this easily

### 4.2 Attack Performance Metrics

**Missing from Experimental Output:**
- ❌ HR@5 (Hit Rate at K=5)
- ❌ nDCG@5 (Normalized Discounted Cumulative Gain at K=5)
- ❌ Model performance evaluation results

The `stealth_logs.json` only contains gradient statistics, not the actual recommendation quality metrics needed to validate the 94.11% HR@5 degradation claim.

---

## 5. Comparison with Paper's Defense Robustness (Figure 3)

**Paper Reports (StegaPoison vs Defenses):**

| Defense Mechanism | ClusterAttack HR@5 | StegaPoison HR@5 | StegaPoison Superiority |
|-------------------|-------------------|------------------|------------------------|
| No Defense | 0.02451 | 0.00209 | 11.7x more effective |
| TrimmedMean | ~0.026 | ~0.021 | ~1.2x more effective |
| Krum | ~0.026 | ~0.017 | ~1.5x more effective |
| MultiKrum | ~0.025 | ~0.012 | ~2.1x more effective |
| NormBound | ~0.025 | ~0.002 | ~12.5x more effective |
| FL-WBC | ~0.025 | ~0.008 | ~3.1x more effective |
| MultiKrum+UNION | ~0.025 | ~0.002 | ~12.5x more effective |
| NormBound+UNION | ~0.025 | ~0.002 | ~12.5x more effective |

**Our Codebase Implementation:**
- ✓ Has NormBound defense (configured at 0.1 L2 norm bound)
- ✓ Has TrimmedMean, Krum, MultiKrum defenses
- ⚠ Has UNION-based hybrid defenses
- ❌ Missing evaluation results against these defenses

---

## 6. Discrepancies & Questions

### 6.1 Gradient Magnitude Concern

**Issue:** The attacker gradient norm is 23,278x larger than benign gradients.

**Expected from Paper:**
- Paper states stealth factor of 1.5 (Table 1)
- Statistical invisibility should keep gradients within normal distribution
- Paper claims to pass norm-based defenses like NormBound

**Actual Observation:**
- Attacker norms are consistently ~10.098
- Benign norms are ~0.0004
- This should be trivially detectable by any norm-based filter

**Possible Explanations:**
1. The stealth_logs.json measures pre-clipping norms (before NormBound applied)
2. The "stealth factor" parameter works differently than expected
3. Implementation may differ from paper's description
4. Different normalization/scaling between paper and implementation

### 6.2 Missing Performance Metrics

**Critical Gap:** No HR@5 or nDCG@5 metrics in the experimental output.

**Required Actions:**
1. Run evaluation script to measure recommendation quality
2. Compare test-set performance against Table 3 expectations
3. Validate 94.11% degradation claim

### 6.3 Defense Evaluation Missing

**Paper extensively evaluates against 7+ defense mechanisms, but:**
- No defense evaluation logs in current experiment directory
- Need to run experiments with different AGG_TYPE settings
- Should validate Figure 3 and Figure 4 claims

---

## 7. Ablation Study Components (Figure 6)

**Paper Tests 8 Configurations:**
1. **Exp-1:** Full Attack (Watermarking + LVDEP + Mirror + Velocity + Stat.Invis)
2. **Exp-2:** No Watermarking
3. **Exp-3:** No LVDEP
4. **Exp-4:** No Mirror Shift  
5. **Exp-5:** No Velocity Sampling
6. **Exp-6:** No Momentum
7. **Exp-7:** No Statistical Invisibility
8. **Full Attack:** All components enabled

**Expected Results (ML-MF):**
- Exp-1 through Exp-7: HR@5 ranges from 0.019-0.022
- Full Attack: HR@5 drops to ~0.0015

**Our Codebase:**
- ✓ Has all ablation parameters in `train.py`
- ✓ Can disable individual components
- ❌ Missing systematic ablation study results

---

## 8. Recommendations

### 8.1 Immediate Actions

1. **Run Evaluation Script:**
   ```bash
   python code/test.py --MODEL_DIR model_all/train2000_ml_MF_stegapoison_FedAdam_FIXED/seed0 \
                       --DATA ml --MODEL_TYPE MF
   ```

2. **Verify Attack Effectiveness:**
   - Measure HR@5 and nDCG@5 at round 2000
   - Compare against baseline (no attack)
   - Validate 94.11% degradation claim

3. **Investigate Gradient Magnitude:**
   - Check if stealth_logs captures pre- or post-defense norms
   - Examine Orchestra.update_one_round() implementation
   - Verify NormBound clipping is applied correctly

### 8.2 Comprehensive Validation

1. **Defense Robustness Testing:**
   - Run experiments with all AGG_TYPE values
   - Generate Figure 3 reproduction
   - Validate stealth claims against norm-based defenses

2. **Ablation Study:**
   - Systematically disable attack components
   - Reproduce Figure 6(a) and 6(b)
   - Validate synergy claims

3. **Multi-Seed Validation:**
   - Paper reports "mean results across 5 independent runs"
   - Run seeds 1, 2, 3, 4 to match paper methodology
   - Calculate mean ± std for all metrics

---

## 9. Summary

### What Matches Paper:
✓ Directional stealth (low cosine similarity)  
✓ Consistent attack gradient generation  
✓ StegaPoison parameters implemented in code  

### What's Concerning:
✗ Extreme gradient magnitude ratio (23,278x) vs paper's stealth claims  
✗ Missing HR@5/nDCG@5 performance metrics  
✗ No defense evaluation results  
✗ Single seed run (paper uses 5)  

### Next Steps:
1. Run evaluation to get HR@5 and nDCG@5 metrics
2. Investigate gradient norm measurement methodology
3. Run defense robustness experiments
4. Perform multi-seed validation

---

**Report Generated:** August 8, 2026  
**Author:** Automated Analysis System  
**Status:** Preliminary - Requires Performance Metrics
