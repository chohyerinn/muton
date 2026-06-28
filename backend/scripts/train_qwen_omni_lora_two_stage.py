import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_qwen_omni_lora.py"


def run_step(name: str, command: list[str], dry_run: bool) -> None:
    pretty = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n[{name}]")
    print(pretty)
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage LoRA fine-tuning wrapper for Qwen2.5-Omni Thinker.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--stage_a_train_jsonl", type=str, default="out/qwen_omni_meld_train.jsonl")
    parser.add_argument("--stage_a_val_jsonl", type=str, default="out/qwen_omni_meld_dev.jsonl")
    parser.add_argument("--stage_a_output_dir", type=str, default="out/qwen_omni_lora/meld_stage")
    parser.add_argument("--stage_a_epochs", type=int, default=2)
    parser.add_argument("--stage_a_batch_size", type=int, default=1)
    parser.add_argument("--stage_a_grad_accum", type=int, default=4)
    parser.add_argument("--stage_a_lr", type=float, default=2e-4)

    parser.add_argument("--stage_b_train_jsonl", type=str, default="out/qwen_omni_ko.jsonl")
    parser.add_argument("--stage_b_val_jsonl", type=str, default="")
    parser.add_argument("--stage_b_output_dir", type=str, default="out/qwen_omni_lora/ko_stage")
    parser.add_argument("--stage_b_epochs", type=int, default=4)
    parser.add_argument("--stage_b_batch_size", type=int, default=1)
    parser.add_argument("--stage_b_grad_accum", type=int, default=4)
    parser.add_argument("--stage_b_lr", type=float, default=1e-4)
    args = parser.parse_args()

    common = [
        args.python,
        str(TRAIN_SCRIPT),
        "--model_name",
        args.model_name,
        "--torch_dtype",
        args.torch_dtype,
    ]
    if args.load_in_4bit:
        common.append("--load_in_4bit")
    if args.gradient_checkpointing:
        common.append("--gradient_checkpointing")

    stage_a = common + [
        "--train_jsonl",
        args.stage_a_train_jsonl,
        "--val_jsonl",
        args.stage_a_val_jsonl,
        "--output_dir",
        args.stage_a_output_dir,
        "--epochs",
        str(args.stage_a_epochs),
        "--batch_size",
        str(args.stage_a_batch_size),
        "--grad_accum",
        str(args.stage_a_grad_accum),
        "--learning_rate",
        str(args.stage_a_lr),
    ]

    stage_b = common + [
        "--train_jsonl",
        args.stage_b_train_jsonl,
        "--output_dir",
        args.stage_b_output_dir,
        "--resume_from_adapter",
        args.stage_a_output_dir,
        "--epochs",
        str(args.stage_b_epochs),
        "--batch_size",
        str(args.stage_b_batch_size),
        "--grad_accum",
        str(args.stage_b_grad_accum),
        "--learning_rate",
        str(args.stage_b_lr),
    ]
    if args.stage_b_val_jsonl:
        stage_b.extend(["--val_jsonl", args.stage_b_val_jsonl])

    run_step("stage_a_qwen_omni_lora", stage_a, args.dry_run)
    run_step("stage_b_qwen_omni_lora", stage_b, args.dry_run)


if __name__ == "__main__":
    main()
