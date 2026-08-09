#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Centralized Training Runner for StegaPoison Defenses")
    parser.add_argument("--DATA", type=str, nargs="+", default=["ml", "gowalla"], choices=["ml", "gowalla"])
    parser.add_argument("--MODEL_TYPE", type=str, nargs="+", default=["MF", "SASRec"], choices=["MF", "SASRec"])
    parser.add_argument(
        "--AGG_TYPE", 
        type=str, 
        nargs="+", 
        default=["FedAdam", "TrimmedMean", "Krum", "MultiKrum", "NormBound", "FLWBC", "MultiKrumUNION", "NormBoundUNION"],
        choices=["FedAdam", "TrimmedMean", "Krum", "MultiKrum", "NormBound", "FLWBC", "MultiKrumUNION", "NormBoundUNION"]
    )
    parser.add_argument("--MAX_ROUND", type=int, default=6000)
    parser.add_argument("--SAVE_ROUND", type=int, default=200)
    parser.add_argument("--LOG_ROUND", type=int, default=100)
    parser.add_argument("--ATTACKER_RATIO", type=float, default=0.05)
    parser.add_argument("--ATTACKER_STRAT", type=str, default="StegaPoison")
    parser.add_argument("--DEVICE", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Paths setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_all_dir = os.path.abspath(os.path.join(base_dir, "..", "model_all"))
    logs_dir = os.path.abspath(os.path.join(base_dir, "..", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_all_dir, exist_ok=True)

    print("================================================================")
    # Pass device selection through to child processes without importing torch here.
    # Importing torch in the parent process initializes a CUDA context that competes
    # with the child subprocess for GPU memory on limited-VRAM cards (e.g. GTX 1050 Ti 4GB).
    device = args.DEVICE  # "auto" is passed to child; child's train.py handles detection
    print(f"Starting Training Suite Runner on Device: {device.upper()}")
    print("================================================================")
    print(f"Datasets:   {args.DATA}")
    print(f"Models:     {args.MODEL_TYPE}")
    print(f"Defenses:   {args.AGG_TYPE}")
    print(f"Max Rounds: {args.MAX_ROUND} (save every {args.SAVE_ROUND}, log every {args.LOG_ROUND})")
    print("================================================================")

    # Grid search expansion
    run_configs = []
    for data in args.DATA:
        for model in args.MODEL_TYPE:
            for agg in args.AGG_TYPE:
                run_configs.append((data, model, agg))

    completed_runs = 0
    skipped_runs = 0
    failed_runs = []

    for idx, (data, model, agg) in enumerate(run_configs):
        exp_name = f"train6000_{data}_{model}_stegapoison_{agg}"
        if args.MAX_ROUND != 6000:
            exp_name = f"train{args.MAX_ROUND}_{data}_{model}_stegapoison_{agg}"

        ckpt_dir = os.path.join(model_all_dir, exp_name, "seed0")
        target_ckpt = os.path.join(ckpt_dir, f"round-{args.MAX_ROUND}.pt")
        
        print(f"\n[{idx+1}/{len(run_configs)}] Checking: {exp_name}...")
        
        # Check if already trained to the max round
        if os.path.exists(target_ckpt):
            print(f"  => [SKIP] Already completed! Checkpoint found at {target_ckpt}")
            skipped_runs += 1
            continue

        print(f"  => [START] Launching training run...")
        log_file_path = os.path.join(logs_dir, f"{exp_name}.txt")
        print(f"  => [LOGS] Streaming output to: {log_file_path}")
        
        # Build CLI command
        cmd = [
            sys.executable,
            "train.py",
            "--EXP_NAME", exp_name,
            "--MODEL_TYPE", model,
            "--DATA", data,
            "--AGG_TYPE", agg,
            "--MAX_ROUND", str(args.MAX_ROUND),
            "--SAVE_ROUND", str(args.SAVE_ROUND),
            "--LOG_ROUND", str(args.LOG_ROUND),
            "--ATTACKER_RATIO", str(args.ATTACKER_RATIO),
            "--ATTACKER_STRAT", args.ATTACKER_STRAT,
            "--DEVICE", device,
            "--SEED", "0"
        ]
        
        # Run process and stream to log file
        try:
            with open(log_file_path, "a", encoding="utf-8") as log_f:
                process = subprocess.Popen(
                    cmd,
                    cwd=base_dir,
                    stdout=log_f,
                    stderr=subprocess.STDOUT
                )
                print("  => [RUNNING] Process started. Waiting for completion...")
                process.wait()
                
            if process.returncode == 0:
                print(f"  => [SUCCESS] Completed training for {exp_name}")
                completed_runs += 1
            else:
                print(f"  => [ERROR] Non-zero return code {process.returncode} for {exp_name}")
                failed_runs.append(exp_name)
        except Exception as e:
            print(f"  => [EXCEPTION] Failed to run {exp_name}: {str(e)}")
            failed_runs.append(exp_name)

    print("\n================================================================")
    print("Suite Run Execution Summary:")
    print(f"  - Total Configs: {len(run_configs)}")
    print(f"  - Skipped (Already Completed): {skipped_runs}")
    print(f"  - Successfully Trained: {completed_runs}")
    print(f"  - Failed Runs: {len(failed_runs)}")
    if len(failed_runs) > 0:
        print(f"    Failed details: {failed_runs}")
    print("================================================================")

if __name__ == "__main__":
    main()
