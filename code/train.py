# %%
import sys
import numpy as np

import os
import random
import math
import pickle
import torch
import argparse
from tqdm import tqdm

from orchestra import Orchestra
from model import get_model

# assert "1.7.1" in torch.__version__


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP_NAME", type=str, default=None)
    parser.add_argument("--MODEL_TYPE", type=str, default="MF")
    parser.add_argument("--DATA", type=str, default="ml", choices=["ml", "gowalla"])
    parser.add_argument("--CACHE_DIR", type=str, default="../CacheData")
    parser.add_argument("--DROPOUT", type=float, default=0.2)
    parser.add_argument("--EMBDIM", type=int, default=64)
    parser.add_argument("--LR", type=float, default=1e-3)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-5)
    parser.add_argument("--BATCH_SIZE", type=int, default=512)
    parser.add_argument(
        "--AGG_TYPE",
        type=str,
        choices=["FedAdam", "TrimmedMean", "Krum", "MultiKrum", "NormBound", "FLWBC", "MultiKrumUNION", "NormBoundUNION"],
        default="FedAdam",
    )
    parser.add_argument("--ATTACKER_RATIO", type=float, default=0)
    parser.add_argument(
        "--ATTACKER_STRAT",
        type=str,
        default=None,
        choices=["StegaPoison"],
    )
    parser.add_argument(
        "--ATTACK_MODE",
        type=str,
        default="hybrid",
        choices=["collapse", "reverse", "noise", "similarity", "hybrid"],
        help="Attack mode for StegaPoison"
    )
    parser.add_argument("--USER_SAMPLE_NUM", type=int, default=50)
    parser.add_argument("--ATTACKER_SAMPLE_NUM", type=int, default=50)
    parser.add_argument("--NORM_BOUND", type=int, default=0.1)
    parser.add_argument("--MAX_ROUND", type=int, default=6000)
    parser.add_argument("--SAVE_ROUND", type=int, default=200)
    parser.add_argument("--LOG_ROUND", type=int, default=100)

    # === ALGORITHM HYPERPARAMETERS ===
    parser.add_argument("--SEED", type=int, default=0)
    parser.add_argument("--SCALE", type=float, default=3)
    parser.add_argument("--ALPHA", type=float, default=1)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--AGG_SAMPLE_NUM", type=float, default=500)
    parser.add_argument("--GAP_SAMPLE", type=int, default=50)
    # SASRec specific parameters
    parser.add_argument("--MAX_SEQ_LEN", type=int, default=50)
    parser.add_argument("--NUM_HEADS", type=int, default=2)
    parser.add_argument("--NUM_LAYERS", type=int, default=2)
    
    # StegaPoison steganographic parameters - MAXIMIZED
    parser.add_argument("--VADP_THRESHOLD", type=float, default=0.5, help="Variance-Adaptive Dimension Perturbation threshold (higher = more aggressive)")
    parser.add_argument("--MIRROR_RATIO", type=float, default=0.95, help="Ratio of dimensions to mirror (higher = more disruption)")
    parser.add_argument("--NOISE_VARIANCE", type=float, default=0.35, help="Statistical noise level (higher = stronger)")
    parser.add_argument("--COHERENCE_WEIGHT", type=float, default=20.0, help="Embedding coherence weight (higher = more disruption)")
    parser.add_argument("--INVISIBILITY_FACTOR", type=float, default=0.3, help="Statistical invisibility factor (lower = stronger attack)")
    parser.add_argument("--WATERMARK_SCALE", type=float, default=0.05, help="Scale for watermark injection")
    parser.add_argument("--VADP_SCALE", type=float, default=1.0, help="Scale for VADP noise")
    parser.add_argument("--MOMENTUM", type=float, default=0.8, help="Momentum for velocity-based sampling (0 = disabled)")
    parser.add_argument("--STEALTH_FACTOR", type=float, default=1.5, help="Stealth clipping standard deviation margin (higher = relaxed constraint)")
    parser.add_argument("--DEVICE", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use for training (auto/cpu/cuda/mps)")

    args = parser.parse_args()
    args.MODEL_DIR = f"../model_all/{args.EXP_NAME}/seed{args.SEED}"
    args.GAP_CACHE = "../CacheData/gap_cache.pkl"
    args.ATTACKER_PER_ROUND = math.ceil(args.USER_SAMPLE_NUM * args.ATTACKER_RATIO)
    if args.DATA == "ml":
        args.DATA_PATH = "../Data/ml-1m/ratings.dat"
    else:
        args.DATA_PATH = "../Data/gowalla_10core.tsv"
    return args


args = parse_args()
os.makedirs(args.MODEL_DIR, exist_ok=True)
os.makedirs(args.CACHE_DIR, exist_ok=True)

# %%
with open(args.DATA_PATH, "r", encoding="utf-8") as f:
    ratings = f.readlines()

print("Number of ratings:", len(ratings))
user_data = {}
uid_remap, iid_remap = {}, {}
for line in tqdm(ratings):
    if args.DATA == "ml":
        uid, iid, rate, timestamp = line.strip("\n").split("::")
        timestamp = int(timestamp)
    else:
        uid, iid, timestamp = line.strip("\n").split("\t")
        timestamp = int(timestamp)

    if uid not in uid_remap:
        uid_remap[uid] = len(uid_remap)
    if iid not in iid_remap:
        iid_remap[iid] = len(iid_remap)
    uid = uid_remap[uid]
    iid = iid_remap[iid]

    if uid not in user_data:
        user_data[uid] = [(iid, timestamp)]
    else:
        user_data[uid].append((iid, timestamp))

args.NUM_USERS = len(uid_remap)
args.NUM_ITEMS = len(iid_remap)
print("Number of users:", args.NUM_USERS)
print("Number of items:", args.NUM_ITEMS)

for uid in user_data:
    user_data[uid] = sorted(user_data[uid], key=lambda x: x[-1])

# %%
full_attacker_list_path = os.path.join(
    args.CACHE_DIR, f"{args.DATA}-full-attacker-list.pkl"
)

if os.path.exists(full_attacker_list_path):
    with open(full_attacker_list_path, "rb") as f:
        full_attacker_list = pickle.load(f)
    print("Loading full attacker list from", full_attacker_list_path)
else:
    full_attacker_list = random.sample(
        range(args.NUM_USERS), k=int(0.05 * args.NUM_USERS)
    )
    with open(full_attacker_list_path, "wb") as f:
        pickle.dump(full_attacker_list, f)
    print("Dumping full attacker list to", full_attacker_list_path)

attacker_list_path = os.path.join(
    args.CACHE_DIR, f"{args.DATA}-attacker-list-{args.ATTACKER_RATIO}.pkl"
)
if os.path.exists(attacker_list_path):
    with open(attacker_list_path, "rb") as f:
        attacker_id_list = pickle.load(f)
    print("Loading attacker list from", attacker_list_path)
else:
    attacker_id_list = random.sample(
        full_attacker_list, k=int(args.ATTACKER_RATIO * args.NUM_USERS)
    )
    with open(attacker_list_path, "wb") as f:
        pickle.dump(attacker_id_list, f)
    print("Dumping attacker list to", attacker_list_path)

cache_path = os.path.join(args.CACHE_DIR, f"{args.DATA}-seed{args.SEED}.pkl")
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
        train_user_data = cache_data["train_user_data"]
        val_user_data = cache_data["val_user_data"]
        test_user_data = cache_data["test_user_data"]
    print("Loading cache data from", cache_path)
else:
    train_user_data = {}
    val_user_data = {"uid": [], "label": [], "mask": []}
    test_user_data = {"uid": [], "label": [], "mask": []}
    iid_set = set(range(args.NUM_ITEMS))

    for uid in tqdm(user_data):
        pos_iid = [x[0] for x in user_data[uid]]
        train_pos_iid = pos_iid[:-2]
        val_pos_iid = pos_iid[-2]
        test_pos_iid = pos_iid[-1]

        candidate_iid_list = list(iid_set - set(pos_iid))

        # train
        for iid in train_pos_iid:
            neg_iid = random.choice(candidate_iid_list)
            if uid not in train_user_data:
                train_user_data[uid] = [[iid, neg_iid]]  # pos, neg
            else:
                train_user_data[uid].append([iid, neg_iid])

        train_user_data[uid] = np.array(train_user_data[uid])

        if uid not in full_attacker_list:
            # val
            val_user_data["uid"].append(uid)
            val_user_data["label"].append(val_pos_iid)
            label_mask = np.zeros(args.NUM_ITEMS, dtype=bool)
            label_mask[pos_iid[:-2]] = True
            val_user_data["mask"].append(label_mask)
            # test
            test_user_data["uid"].append(uid)
            test_user_data["label"].append(test_pos_iid)
            label_mask = np.zeros(args.NUM_ITEMS, dtype=bool)
            label_mask[pos_iid[:-1]] = True
            test_user_data["mask"].append(label_mask)

    val_user_data["uid"] = np.array(val_user_data["uid"])
    val_user_data["label"] = np.array(val_user_data["label"])
    val_user_data["mask"] = np.array(val_user_data["mask"])
    test_user_data["uid"] = np.array(test_user_data["uid"])
    test_user_data["label"] = np.array(test_user_data["label"])
    test_user_data["mask"] = np.array(test_user_data["mask"])

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "train_user_data": train_user_data,
                "val_user_data": val_user_data,
                "test_user_data": test_user_data,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("Dumping cache data to", cache_path)

if args.ATTACKER_STRAT is None:
    attacker_id_list = []

print(f"{len(attacker_id_list)} attackers in total")
print(attacker_id_list)
print("Number of training data:", sum(len(train_user_data[x]) for x in train_user_data))
print("Number of validation data:", len(val_user_data["label"]))
print("Number of test data:", len(test_user_data["label"]))

# %%
random.seed(args.SEED)
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
torch.cuda.manual_seed_all(args.SEED)

if args.DEVICE == "cpu":
    device = torch.device("cpu")
elif args.DEVICE == "cuda" or (args.DEVICE == "auto" and torch.cuda.is_available()):
    device = torch.device("cuda")
elif args.DEVICE == "mps" or (args.DEVICE == "auto" and torch.backends.mps.is_available()):
    device = torch.device("mps")
else:
    device = torch.device("cpu")
server_model = get_model(args).to(device)
print(server_model)

# Check for existing checkpoints to resume training
start_round = 0
checkpoint_files = []
if os.path.exists(args.MODEL_DIR):
    checkpoint_files = [f for f in os.listdir(args.MODEL_DIR) if f.startswith("round-") and f.endswith(".pt")]

if checkpoint_files:
    # Extract round numbers and find the maximum one
    round_numbers = [int(f.split("-")[1].split(".")[0]) for f in checkpoint_files]
    max_round = max(round_numbers)
    if max_round < args.MAX_ROUND:
        ckpt_path = os.path.join(args.MODEL_DIR, f"round-{max_round}.pt")
        print(f"Resuming training from checkpoint: {ckpt_path}")
        server_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        start_round = max_round

# %%
orch = Orchestra(server_model, train_user_data, attacker_id_list, args, device)

# Stealth logging
stealth_logs = {
    'round': [],
    'attacker_grad_norm': [],
    'benign_grad_norm': [],
    'mean_cosine_sim': []
}

if start_round > 0:
    # Load existing stealth logs if they exist
    stealth_log_path = os.path.join(args.MODEL_DIR, "stealth_logs.json")
    if os.path.exists(stealth_log_path):
        try:
            import json
            with open(stealth_log_path, 'r') as f:
                existing_logs = json.load(f)
            # Keep logs up to start_round (since rounds are 1-indexed, keep up to start_round elements)
            stealth_logs = {
                'round': existing_logs['round'][:start_round],
                'attacker_grad_norm': existing_logs['attacker_grad_norm'][:start_round],
                'benign_grad_norm': existing_logs['benign_grad_norm'][:start_round],
                'mean_cosine_sim': existing_logs['mean_cosine_sim'][:start_round]
            }
            print(f"Loaded existing stealth logs for {start_round} rounds")
        except Exception as e:
            print(f"Warning: could not load existing stealth logs: {e}")

# %%
for i in tqdm(range(start_round, args.MAX_ROUND)):
    (
        round_loss,
        round_acc,
        round_attacker,
        attacker_grad_norm,
        benign_grad_norm,
        mean_cosine_sim,
        filter_stat,
        attacker_loss,
    ) = orch.update_one_round(i)
    
    # Record stealth stats
    stealth_logs['round'].append(i + 1)
    stealth_logs['attacker_grad_norm'].append(attacker_grad_norm.item() if attacker_grad_norm is not None else None)
    stealth_logs['benign_grad_norm'].append(benign_grad_norm.item() if benign_grad_norm is not None else None)
    stealth_logs['mean_cosine_sim'].append(mean_cosine_sim.item() if mean_cosine_sim is not None else None)

    # Clear MPS cache to prevent Out-Of-Memory errors on Apple Silicon Mac
    if device.type == "mps":
        torch.mps.empty_cache()

    if (i + 1) % args.LOG_ROUND == 0:
        print(
            "Round: {}, train_loss: {:.5f}, acc: {:.5f}".format(
                i + 1, round_loss, round_acc
            )
        )

    if (i + 1) % args.SAVE_ROUND == 0:
        ckpt_path = os.path.join(args.MODEL_DIR, f"round-{i + 1}.pt")
        torch.save(orch.agg.server_model.state_dict(), ckpt_path)
        print(f"Model saved to {ckpt_path}")

# Save stealth logs
stealth_log_path = os.path.join(args.MODEL_DIR, "stealth_logs.json")
import json
with open(stealth_log_path, 'w') as f:
    json.dump(stealth_logs, f, indent=4)
print(f"Stealth logs saved to {stealth_log_path}")
