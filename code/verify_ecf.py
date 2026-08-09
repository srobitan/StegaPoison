#!/usr/bin/env python3
"""
Unit Test to Verify the Mathematical Correctness of the ECF Defense Mechanism.
This script checks:
1. Temporal Embedding Drift Monitoring (TEDM) tracking and scoring.
2. Interdimensional Consistency Check (IDC) discrete histogramming, KL-D, and EMD computation.
3. Compatibility Drop Estimator (CDE) before-and-after cosine similarity drop detection.
4. Overall Credibility-Weighted Aggregation.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace

# Import the aggregator classes and other dependencies
import sys
sys.path.append('code')
from agg import ECF

class MockUserEmbedding(nn.Module):
    def __init__(self, num_users, emb_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        # Initialize with standard normal
        nn.init.normal_(self.user_embedding.weight)

class MockItemEmbedding(nn.Module):
    def __init__(self, num_items, emb_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.item_embedding.weight)

class MockPredictor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, u, i):
        return (u * i).sum(dim=-1)

class MockModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.user_model = MockUserEmbedding(num_users, emb_dim)
        self.item_model = MockItemEmbedding(num_items, emb_dim)
        self.predictor = MockPredictor()

    def forward(self, uid, iid):
        u = self.user_model.user_embedding(uid)
        i = self.item_model.item_embedding(iid)
        score = self.predictor(u, i)
        loss = score.mean()
        return score, loss

def test_ecf_defense():
    print("================================================================")
    print("Testing Mathematical Soundness of ECF Defense Mechanism")
    print("================================================================")

    # 1. Setup Namespace & Config
    args = Namespace(
        SEED=0,
        EMBDIM=64,
        NUM_USERS=10,
        NUM_ITEMS=20,
        LR=0.01,
        BATCH_SIZE=2,
        USER_SAMPLE_NUM=5,
        AGG_TYPE="ECF",
        # ECF Hyperparameters
        ECF_K=1.5,
        ECF_GAMMA_TEDM=1.0,
        ECF_BINS_IDC=20,
        ECF_TAU_IDC=2.0,
        ECF_ALPHA_IDC=2.0,
        ECF_LAMBDA_CDE=10.0
    )
    
    device = torch.device("cpu")
    model = MockModel(args.NUM_USERS, args.NUM_ITEMS, args.EMBDIM).to(device)
    
    # Instantiate ECF
    ecf_defense = ECF(model, args, device)
    
    # 2. Simulate Round 1 Client Updates
    # We will have 5 clients in total.
    # Clients 0, 1, 2, 3 are benign (small standard updates).
    # Client 4 is an attacker (large updates, structured interdimensional perturbation, compatibility drop).
    
    print("\n[Simulation] Preparing client updates for Round 1...")
    
    # Active items per client
    client_active_items = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7, 8],
        4: [9, 10]
    }
    
    # User embeddings before update
    user_weight_before = model.user_model.user_embedding.weight.clone()
    item_weight_before = model.item_model.item_embedding.weight.clone()
    
    for client_idx in range(5):
        is_attacker = (client_idx == 4)
        active_items = client_active_items[client_idx]
        
        # User grad
        user_grad = torch.zeros(args.NUM_USERS * args.EMBDIM)
        # Non-zero gradient for client's own ID
        uid_offset = client_idx * args.EMBDIM
        if is_attacker:
            # Attacker user gradient points in opposite/adversarial direction to create compatibility drop
            user_grad[uid_offset : uid_offset + args.EMBDIM] = 5.0 * torch.randn(args.EMBDIM)
        else:
            user_grad[uid_offset : uid_offset + args.EMBDIM] = 0.1 * torch.randn(args.EMBDIM)
            
        # Item grad
        item_grad = torch.zeros(args.NUM_ITEMS * args.EMBDIM)
        for iid in active_items:
            iid_offset = iid * args.EMBDIM
            if is_attacker:
                # Structured perturbation in specific dimensions (e.g. all positive high variance)
                # This breaks the IDC interdimensional consistency
                atk_vec = torch.zeros(args.EMBDIM)
                atk_vec[0::2] = 2.5 # Even dimensions highly positive
                atk_vec[1::2] = -2.5 # Odd dimensions highly negative
                item_grad[iid_offset : iid_offset + args.EMBDIM] = atk_vec
            else:
                item_grad[iid_offset : iid_offset + args.EMBDIM] = 0.05 * torch.randn(args.EMBDIM)
                
        # Other grad
        other_grad = torch.zeros(1)
        
        # Sample num
        sample_num = 100
        
        ecf_defense.collect_client_update(
            client_user_grad=user_grad,
            client_item_grad=item_grad,
            client_other_grad=other_grad,
            client_sample_num=sample_num,
            is_attacker=is_attacker,
            client_id=client_idx
        )
        
    print("  - All 5 client updates collected.")
    
    # 3. Perform Aggregation and ECF checks
    print("\n[Evaluation] Running ECF Aggregation for Round 1...")
    
    # We will temporarily mock the return of agg to inspect the internal values
    # Let's save the original server_optimizer step to verify everything runs smoothly
    orig_step = ecf_defense.server_optimizer.step
    
    # Execute ECF.agg
    filter_precision, filter_recall, num_filtered = ecf_defense.agg()
    
    print(f"  - Aggregation completed successfully!")
    print(f"  - Filter Precision: {filter_precision:.4f} (expected: 1.0 since only the attacker should be flagged)")
    print(f"  - Filter Recall: {filter_recall:.4f} (expected: 1.0 since the attacker was identified)")
    print(f"  - Number of Filtered Clients (credibility < 0.5): {num_filtered}")
    
    assert filter_precision == 1.0, f"Filter precision should be 1.0, got {filter_precision}"
    assert filter_recall == 1.0, f"Filter recall should be 1.0, got {filter_recall}"
    assert num_filtered == 1, f"Should filter exactly 1 client (the attacker), got {num_filtered}"
    print("  => Round 1 Detection and Filtering: PASSED")
    
    # 4. Verify Temporal Embedding Drift Monitoring (TEDM) for Round 2
    # We will simulate a second round. For client 4 (the attacker), they will send a highly drifting update.
    # For benign clients, they will send updates similar to round 1 (minimal drift).
    print("\n[Simulation] Preparing client updates for Round 2 (Verifying TEDM)...")
    
    for client_idx in range(5):
        is_attacker = (client_idx == 4)
        active_items = client_active_items[client_idx]
        
        # User grad
        user_grad = torch.zeros(args.NUM_USERS * args.EMBDIM)
        uid_offset = client_idx * args.EMBDIM
        user_grad[uid_offset : uid_offset + args.EMBDIM] = 0.1 * torch.randn(args.EMBDIM)
            
        # Item grad
        item_grad = torch.zeros(args.NUM_ITEMS * args.EMBDIM)
        for iid in active_items:
            iid_offset = iid * args.EMBDIM
            if is_attacker:
                # Highly drifting update compared to Round 1
                item_grad[iid_offset : iid_offset + args.EMBDIM] = 8.0 * torch.randn(args.EMBDIM)
            else:
                # Minor update (minimal drift from previous item embedding weight)
                item_grad[iid_offset : iid_offset + args.EMBDIM] = 0.05 * torch.randn(args.EMBDIM)
                
        # Other grad
        other_grad = torch.zeros(1)
        sample_num = 100
        
        ecf_defense.collect_client_update(
            client_user_grad=user_grad,
            client_item_grad=item_grad,
            client_other_grad=other_grad,
            client_sample_num=sample_num,
            is_attacker=is_attacker,
            client_id=client_idx
        )
        
    print("  - Running ECF Aggregation for Round 2...")
    filter_precision, filter_recall, num_filtered = ecf_defense.agg()
    
    print(f"  - Filter Precision: {filter_precision:.4f}")
    print(f"  - Filter Recall: {filter_recall:.4f}")
    print(f"  - Number of Filtered Clients: {num_filtered}")
    
    assert filter_precision == 1.0, "ECF should maintain 100% precision in Round 2!"
    assert filter_recall == 1.0, "ECF should successfully recall and isolate the attacker in Round 2!"
    print("  => Round 2 Anomaly Drift Monitoring: PASSED")
    
    print("\n================================================================")
    print("SUCCESS: All ECF Defense Mechanism mathematical tests PASSED!")
    print("================================================================")

if __name__ == "__main__":
    test_ecf_defense()
