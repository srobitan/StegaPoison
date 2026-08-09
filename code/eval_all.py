#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import re
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Centralized Evaluation and Metrics Aggregator for StegaPoison Defenses")
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
    parser.add_argument("--SEED", type=int, default=0)
    parser.add_argument("--FORCE_EVAL", action="store_true", help="Force re-running test.py even if cached")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Paths setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_all_dir = os.path.abspath(os.path.join(base_dir, "..", "model_all"))
    results_dir = os.path.abspath(os.path.join(base_dir, "results"))
    os.makedirs(results_dir, exist_ok=True)
    
    cache_file = os.path.join(results_dir, "eval_cache.json")
    eval_cache = {}
    if os.path.exists(cache_file) and not args.FORCE_EVAL:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                eval_cache = json.load(f)
        except Exception:
            pass

    print("================================================================")
    print("Starting Metrics Aggregator and Evaluator Suite")
    print("================================================================")
    print(f"Datasets:   {args.DATA}")
    print(f"Models:     {args.MODEL_TYPE}")
    print(f"Defenses:   {args.AGG_TYPE}")
    print(f"Max Round:  {args.MAX_ROUND}")
    print("================================================================")

    # Grid search expansion
    run_configs = []
    for data in args.DATA:
        for model in args.MODEL_TYPE:
            for agg in args.AGG_TYPE:
                run_configs.append((data, model, agg))

    results = []

    for idx, (data, model, agg) in enumerate(run_configs):
        exp_name = f"train6000_{data}_{model}_stegapoison_{agg}"
        if args.MAX_ROUND != 6000:
            exp_name = f"train{args.MAX_ROUND}_{data}_{model}_stegapoison_{agg}"

        ckpt_dir = os.path.join(model_all_dir, exp_name, "seed0")
        target_ckpt = os.path.join(ckpt_dir, f"round-{args.MAX_ROUND}.pt")
        
        print(f"\n[{idx+1}/{len(run_configs)}] Checking: {exp_name}...")
        
        # Check if the checkpoint exists
        if not os.path.exists(target_ckpt):
            print(f"  => [WARN] Checkpoint not found at {target_ckpt}. Skipping evaluation.")
            continue

        # Check if already cached
        cache_key = f"{data}_{model}_{agg}_{args.MAX_ROUND}"
        if cache_key in eval_cache and not args.FORCE_EVAL:
            print("  => [CACHE] Found cached evaluation metrics!")
            results.append(eval_cache[cache_key])
            continue

        print(f"  => [RUN] Launching test.py to evaluate {target_ckpt}...")
        
        # Build test.py command
        cmd = [
            sys.executable,
            "test.py",
            "--EXP_NAME", exp_name,
            "--MODEL_TYPE", model,
            "--DATA", data,
            "--MAX_ROUND", str(args.MAX_ROUND),
            "--SAVE_ROUND", str(args.MAX_ROUND),
            "--SEED", str(args.SEED)
        ]
        
        try:
            # Run test.py and capture stdout
            process = subprocess.run(
                cmd,
                cwd=base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
            
            output = process.stdout
            err_output = process.stderr
            
            if process.returncode != 0:
                print(f"  => [ERROR] test.py exited with non-zero code {process.returncode}")
                print(f"     Error output:\n{err_output}")
                continue
                
            # Parse metrics using regex for the specific target round
            # [Val] Round: 6000, HR@5: 0.01377, nDCG@5: 0.00833, HR@10: 0.02283, nDCG@10: 0.01121, HR@20: 0.03817, nDCG@20: 0.01509
            val_pattern = rf"\[Val\] Round: {args.MAX_ROUND}, HR@5: ([\d\.]+), nDCG@5: ([\d\.]+), HR@10: ([\d\.]+), nDCG@10: ([\d\.]+), HR@20: ([\d\.]+), nDCG@20: ([\d\.]+)"
            test_pattern = rf"\[Test\] Round: {args.MAX_ROUND}, HR@5: ([\d\.]+), nDCG@5: ([\d\.]+), HR@10: ([\d\.]+), nDCG@10: ([\d\.]+), HR@20: ([\d\.]+), nDCG@20: ([\d\.]+)"
            
            val_match = re.search(val_pattern, output)
            test_match = re.search(test_pattern, output)
            
            if val_match and test_match:
                metrics = {
                    "dataset": data,
                    "model": model,
                    "defense": agg,
                    "max_round": args.MAX_ROUND,
                    "val": {
                        "HR5": float(val_match.group(1)),
                        "nDCG5": float(val_match.group(2)),
                        "HR10": float(val_match.group(3)),
                        "nDCG10": float(val_match.group(4)),
                        "HR20": float(val_match.group(5)),
                        "nDCG20": float(val_match.group(6)),
                    },
                    "test": {
                        "HR5": float(test_match.group(1)),
                        "nDCG5": float(test_match.group(2)),
                        "HR10": float(test_match.group(3)),
                        "nDCG10": float(test_match.group(4)),
                        "HR20": float(test_match.group(5)),
                        "nDCG20": float(test_match.group(6)),
                    }
                }
                
                eval_cache[cache_key] = metrics
                # Write back cache immediately
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(eval_cache, f, indent=2)
                    
                print(f"  => [SUCCESS] Metrics successfully extracted: Test HR@5={metrics['test']['HR5']:.5f}, nDCG@5={metrics['test']['nDCG5']:.5f}")
                results.append(metrics)
            else:
                print("  => [ERROR] Failed to parse metrics from test.py output. Output structure mismatch.")
                # Print last 10 lines of output to help debug
                lines = output.splitlines()
                last_lines = "\n".join(lines[-10:])
                print(f"  Last 10 lines of test.py stdout:\n{last_lines}")
                
        except Exception as e:
            print(f"  => [EXCEPTION] Failed to run test.py: {str(e)}")

    if len(results) == 0:
        print("\n================================================================")
        print("No evaluation results were found or extracted. Unified table skipped.")
        print("================================================================")
        return

    # Generate Markdown Table Report
    md_output_path = os.path.join(results_dir, f"StegaPoison_Defense_Evaluation_{args.MAX_ROUND}.md")
    
    with open(md_output_path, "w", encoding="utf-8") as md_f:
        md_f.write(f"# 🛡️ StegaPoison: Defense Mechanism Evaluation Report (Round {args.MAX_ROUND})\n\n")
        md_f.write("This report presents a unified, side-by-side performance evaluation of federated recommendation defense strategies under the **StegaPoison** attack.\n\n")
        
        # Group results by (dataset, model)
        groups = {}
        for r in results:
            g_key = (r["dataset"], r["model"])
            if g_key not in groups:
                groups[g_key] = []
            groups[g_key].append(r)
            
        for (dataset, model), group in sorted(groups.items()):
            ds_name = "MovieLens-1M" if dataset == "ml" else "Gowalla"
            md_f.write(f"## 📊 {ds_name} - {model} Recommender\n\n")
            md_f.write("| Defense Strategy | Val HR@5 | Val nDCG@5 | Val HR@10 | Val nDCG@10 | Test HR@5 | Test nDCG@5 | Test HR@10 | Test nDCG@10 |\n")
            md_f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            # Sort group: FedAdam first, then alphabetically
            def sort_key(item):
                def_name = item["defense"]
                if def_name == "FedAdam":
                    return "0_FedAdam"
                else:
                    return f"1_{def_name}"
                    
            sorted_group = sorted(group, key=sort_key)
            
            for item in sorted_group:
                def_name = item["defense"]
                v = item["val"]
                t = item["test"]
                
                name_str = f"`FedAdam` (No Defense)" if def_name == "FedAdam" else f"`{def_name}`"
                md_f.write(f"| {name_str} | {v['HR5']:.5f} | {v['nDCG5']:.5f} | {v['HR10']:.5f} | {v['nDCG10']:.5f} | {t['HR5']:.5f} | {t['nDCG5']:.5f} | {t['HR10']:.5f} | {t['nDCG10']:.5f} |\n")
            md_f.write("\n")
            
        print("\n================================================================")
        print(f"Successfully generated unified markdown report at:")
        print(f"  {md_output_path}")
        print("================================================================")

if __name__ == "__main__":
    main()
