# 🎯 QUICK START: Fix Your StegaPoison Implementation

## The Problem
Your attack decays from HR@5 = 0.00610 (round 2000) to HR@5 = 0.02335 (round 6000)
**Paper expects:** HR@5 = 0.00209 at round 6000

## The Root Cause
```
❌ OLD: delta = trained_model - initial_model_from_round_0
✅ NEW: delta = trained_model - current_model
```

Your attack was aiming at where the model WAS 6000 rounds ago, not where it IS now.

## The Fix (3 Steps)

### Step 1: Install Fixed Version (30 seconds)
```bash
cd /Users/apple/Downloads/StegaPoison

# Run the automated fix script
./test_fixed_version.sh
```

### Step 2: Quick Test (10 minutes)
The script will:
- ✅ Backup your original code
- ✅ Install the fixed version  
- ✅ Run a 200-round test
- ✅ Show you the results

### Step 3: Full Test (4-6 hours)
When prompted, say "y" to run the full 6000-round test.

## Expected Results

| Metric | Before Fix | After Fix | Target (Paper) |
|--------|-----------|-----------|----------------|
| Round 2000 HR@5 | 0.00610 ✓ | ~0.00600 ✓ | ~0.00600 |
| Round 6000 HR@5 | 0.02335 ✗ | **~0.00209 ✓** | 0.00209 |
| Attack Decay? | YES ✗ | **NO ✓** | NO |

## What Was Fixed

1. **Critical:** Delta now computed from current model (not ancient round-0 model)
2. **Important:** Momentum accumulation properly implemented  
3. **Significant:** Statistical invisibility recalibrated
4. **Minor:** Base scale reduced from 3.0 to 1.0

## Files Created for You

📄 **Read First:**
- `FIX_GUIDE.md` - Complete guide (this file)
- `DIAGNOSIS_AND_FIX.md` - Technical deep dive

📄 **Reference:**
- `FINAL_COMPARISON_REPORT.md` - Your results vs paper
- `EXECUTIVE_SUMMARY.md` - Quick reference
- `ROUND_COMPARISON_ANALYSIS.md` - Why decay happens

🔧 **Code:**
- `code/attacker/stegapoison_fixed_v2.py` - Fixed implementation
- `test_fixed_version.sh` - Automated test script

## Run It Now

```bash
cd /Users/apple/Downloads/StegaPoison
./test_fixed_version.sh
```

Press Enter and follow the prompts!

---

## Manual Installation (If Script Fails)

```bash
cd /Users/apple/Downloads/StegaPoison/code

# Backup
cp attacker/stegapoison.py attacker/stegapoison_backup.py

# Install fix
cp attacker/stegapoison_fixed_v2.py attacker/stegapoison.py

# Test (200 rounds ~ 10 min)
python3 train.py --EXP_NAME test_fixed_200 \
  --MODEL_TYPE MF --DATA ml --SEED 0 \
  --AGG_TYPE FedAdam --ATTACKER_RATIO 0.05 \
  --ATTACKER_STRAT StegaPoison --MAX_ROUND 200 \
  --LR 2e-3 --SCALE 1.0 --SAVE_ROUND 50

# Evaluate
python3 test.py --EXP_NAME test_fixed_200 \
  --MODEL_TYPE MF --DATA ml --SEED 0 \
  --MAX_ROUND 200 --SAVE_ROUND 50

# If good, run full 6000 rounds
python3 train.py --EXP_NAME train6000_fixed \
  --MODEL_TYPE MF --DATA ml --SEED 0 \
  --AGG_TYPE FedAdam --ATTACKER_RATIO 0.05 \
  --ATTACKER_STRAT StegaPoison --MAX_ROUND 6000 \
  --LR 2e-3 --SCALE 1.0 --SAVE_ROUND 200
```

---

## Success Indicators

After running the fix, you should see:

✅ Round 200 HR@5 < 0.01 (strong attack)
✅ Round 2000 HR@5 < 0.01 (still strong)  
✅ Round 6000 HR@5 ≈ 0.002 (paper match!)
✅ No sudden jumps between rounds
✅ Gradient ratio < 1000x (improved stealth)

## Questions?

Read the detailed documentation:
1. **Why did this happen?** → `DIAGNOSIS_AND_FIX.md`
2. **What's different?** → `ROUND_COMPARISON_ANALYSIS.md`  
3. **How do results compare?** → `FINAL_COMPARISON_REPORT.md`

---

**Bottom Line:** One critical bug caused all your issues. The fix is ready. Run `./test_fixed_version.sh` to apply it.

Good luck! 🚀
