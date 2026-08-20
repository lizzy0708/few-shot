import argparse
import subprocess
import re
import numpy as np


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    match = re.search(r"AUROC:\s*([0-9.]+)", result.stdout)

    if match is None:
        raise RuntimeError("AUROC not found in output.")

    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "decomposition", "mixing"])

    parser.add_argument("--support_domain", type=str, required=True)
    parser.add_argument("--query_domain", type=str, required=True)
    parser.add_argument("--shot", type=int, required=True)

    parser.add_argument("--ckpt", type=str, default="mask_decomposition.pth")
    parser.add_argument("--mix_domains", type=str, nargs="+", default=None)
    parser.add_argument("--mix_num", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mix_mode", type=str, default="centered",
                        choices=["add", "centered", "plus_minus"])

    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    args = parser.parse_args()

    aurocs = []

    for seed in args.seeds:
        if args.method == "baseline":
            cmd = [
                "python", "scripts/eval/run_baseline_fsad.py",
                "--support_domain", args.support_domain,
                "--query_domain", args.query_domain,
                "--shot", str(args.shot),
                "--seed", str(seed),
            ]

        elif args.method == "decomposition":
            cmd = [
                "python", "scripts/eval/run_fewshot.py",
                "--support_domain", args.support_domain,
                "--query_domain", args.query_domain,
                "--shot", str(args.shot),
                "--ckpt", args.ckpt,
                "--seed", str(seed),
            ]

        elif args.method == "mixing":
            if args.mix_domains is None:
                raise ValueError("--mix_domains is required for mixing method.")

            cmd = [
                "python", "scripts/eval/run_mixing_fsad.py",
                "--support_domain", args.support_domain,
                "--query_domain", args.query_domain,
                "--mix_domains",
            ]

            cmd.extend(args.mix_domains)

            cmd.extend([
                "--shot", str(args.shot),
                "--mix_num", str(args.mix_num),
                "--alpha", str(args.alpha),
                "--mix_mode", args.mix_mode,
                "--ckpt", args.ckpt,
                "--seed", str(seed),
            ])

        auroc = run_command(cmd)
        aurocs.append(auroc)

    aurocs = np.array(aurocs)

    print("\n" + "#" * 80)
    print("Final Result")
    print("#" * 80)
    print("Method :", args.method)
    print("Support:", args.support_domain)
    print("Query  :", args.query_domain)
    print("Shot   :", args.shot)
    print("Seeds  :", args.seeds)
    print("AUROCs :", aurocs.tolist())
    print(f"Mean   : {aurocs.mean():.4f}")
    print(f"Std    : {aurocs.std():.4f}")
    print("#" * 80)


if __name__ == "__main__":
    main()