#!/bin/bash
# Test Script for Fixed StegaPoison Implementation
# This script helps validate the bug fixes and compare results

echo "=============================================="
echo "StegaPoison Fix Validation Test"
echo "=============================================="
echo ""

# Configuration
DATA="ml"
MODEL="MF"
SEED=0
AGG="FedAdam"
ATTACK_RATIO=0.05
LR="2e-3"
SCALE="1.0"

# Step 1: Backup original and install fixed version
echo "Step 1: Installing fixed version..."
cd /Users/apple/Downloads/StegaPoison/code

# Backup current version if not already backed up
if [ ! -f "attacker/stegapoison_before_fix_$(date +%Y%m%d).py" ]; then
    cp attacker/stegapoison.py "attacker/stegapoison_before_fix_$(date +%Y%m%d).py"
    echo "✓ Backed up original to stegapoison_before_fix_$(date +%Y%m%d).py"
fi

# Install fixed version
cp attacker/stegapoison_fixed_v2.py attacker/stegapoison.py
echo "✓ Installed fixed version"
echo ""

# Step 2: Quick test (200 rounds)
echo "Step 2: Running quick test (200 rounds)..."
echo "This will verify the fix works without decay in early rounds."
echo ""

python3 train.py \
    --EXP_NAME test_fixed_200_ml_MF_stegapoison_FedAdam \
    --MODEL_TYPE MF \
    --DATA ml \
    --SEED 0 \
    --AGG_TYPE FedAdam \
    --ATTACKER_RATIO 0.05 \
    --ATTACKER_STRAT StegaPoison \
    --MAX_ROUND 200 \
    --SAVE_ROUND 50 \
    --LOG_ROUND 50 \
    --LR 2e-3 \
    --SCALE 1.0

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Quick test completed successfully"
    echo ""

    # Evaluate quick test
    echo "Evaluating quick test results..."
    python3 test.py \
        --EXP_NAME test_fixed_200_ml_MF_stegapoison_FedAdam \
        --MODEL_TYPE MF \
        --DATA ml \
        --SEED 0 \
        --MAX_ROUND 200 \
        --SAVE_ROUND 50

    echo ""
    echo "✓ Quick test evaluation completed"
    echo ""
else
    echo "✗ Quick test failed"
    exit 1
fi

# Step 3: Prompt user for full test
echo "=============================================="
echo "Quick test completed successfully!"
echo "=============================================="
echo ""
echo "Results should show:"
echo "  - Stable HR@5 across all checkpoints (no sudden drops)"
echo "  - HR@5 around 0.006 or lower (strong attack)"
echo "  - Stealth logs show reasonable gradient magnitudes"
echo ""
echo "Would you like to run the full 6000-round test? (y/n)"
read -p "> " run_full

if [ "$run_full" = "y" ] || [ "$run_full" = "Y" ]; then
    echo ""
    echo "Step 3: Running full test (6000 rounds)..."
    echo "This will take several hours. Progress will be logged."
    echo ""

    python3 train.py \
        --EXP_NAME train6000_ml_MF_stegapoison_FedAdam_FIXED_V2 \
        --MODEL_TYPE MF \
        --DATA ml \
        --SEED 0 \
        --AGG_TYPE FedAdam \
        --ATTACKER_RATIO 0.05 \
        --ATTACKER_STRAT StegaPoison \
        --MAX_ROUND 6000 \
        --SAVE_ROUND 200 \
        --LOG_ROUND 100 \
        --LR 2e-3 \
        --SCALE 1.0 \
        2>&1 | tee ../logs/train6000_ml_MF_stegapoison_FedAdam_FIXED_V2.txt

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Full test completed successfully"
        echo ""

        # Evaluate full test
        echo "Evaluating full test results..."
        python3 test.py \
            --EXP_NAME train6000_ml_MF_stegapoison_FedAdam_FIXED_V2 \
            --MODEL_TYPE MF \
            --DATA ml \
            --SEED 0 \
            --MAX_ROUND 6000 \
            --SAVE_ROUND 200

        echo ""
        echo "✓ Full test evaluation completed"
        echo ""
    else
        echo "✗ Full test failed"
        exit 1
    fi
else
    echo ""
    echo "Skipping full test. You can run it manually later with:"
    echo ""
    echo "  cd /Users/apple/Downloads/StegaPoison/code"
    echo "  python3 train.py --EXP_NAME train6000_ml_MF_stegapoison_FedAdam_FIXED_V2 \\"
    echo "    --MODEL_TYPE MF --DATA ml --SEED 0 --AGG_TYPE FedAdam \\"
    echo "    --ATTACKER_RATIO 0.05 --ATTACKER_STRAT StegaPoison \\"
    echo "    --MAX_ROUND 6000 --SAVE_ROUND 200 --LOG_ROUND 100 \\"
    echo "    --LR 2e-3 --SCALE 1.0"
    echo ""
fi

echo "=============================================="
echo "Test script completed"
echo "=============================================="
echo ""
echo "Expected Results (if fix works):"
echo "  - Round 200:  HR@5 ≈ 0.006, nDCG@5 ≈ 0.003"
echo "  - Round 2000: HR@5 ≈ 0.006, nDCG@5 ≈ 0.003 (no decay!)"
echo "  - Round 6000: HR@5 ≈ 0.002, nDCG@5 ≈ 0.001 (matching paper!)"
echo ""
echo "Compare with your original results:"
echo "  - Original Round 2000: HR@5 = 0.00610"
echo "  - Original Round 6000: HR@5 = 0.02335 (decay!)"
echo ""
echo "If fixed properly, Round 6000 should now show HR@5 ≈ 0.002"
echo "instead of 0.023, matching the paper's expected results."
echo ""
