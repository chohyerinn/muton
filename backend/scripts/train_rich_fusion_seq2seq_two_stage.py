import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_rich_fusion_seq2seq.py"


def run_step(name: str, command: list[str], dry_run: bool) -> None:
    pretty = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n[{name}]")
    print(pretty)
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the richer multimodal fusion-to-text training in two stages.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--decoder_model", type=str, default="google/mt5-small")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--stage_a_pre_ckpt", type=str, default="out/fusion_ko_final/final.pt")
    parser.add_argument("--stage_a_train_pt", type=str, default="out/meld_train_rich.pt")
    parser.add_argument("--stage_a_val_pt", type=str, default="out/meld_dev_rich.pt")
    parser.add_argument("--stage_a_out", type=str, default="out/fusion_seq2seq_rich/meld_best.pt")
    parser.add_argument("--stage_a_epochs", type=int, default=5)
    parser.add_argument("--stage_a_bs", type=int, default=2)
    parser.add_argument("--stage_a_lr_fusion", type=float, default=3e-4)
    parser.add_argument("--stage_a_lr_decoder", type=float, default=1e-4)
    parser.add_argument("--stage_a_max_target_len", type=int, default=48)
    parser.add_argument("--stage_a_max_new_tokens", type=int, default=24)
    parser.add_argument("--stage_a_num_beams", type=int, default=4)
    parser.add_argument("--stage_a_max_memory_tokens", type=int, default=320)

    parser.add_argument("--stage_b_train_pt", type=str, default="out/fusion_dataset_rich.pt")
    parser.add_argument("--stage_b_out", type=str, default="out/fusion_seq2seq_rich/ko_best.pt")
    parser.add_argument("--stage_b_epochs", type=int, default=12)
    parser.add_argument("--stage_b_bs", type=int, default=2)
    parser.add_argument("--stage_b_lr_fusion", type=float, default=1e-4)
    parser.add_argument("--stage_b_lr_decoder", type=float, default=5e-5)
    parser.add_argument("--stage_b_val_ratio", type=float, default=0.2)
    parser.add_argument("--stage_b_max_target_len", type=int, default=64)
    parser.add_argument("--stage_b_max_new_tokens", type=int, default=32)
    parser.add_argument("--stage_b_num_beams", type=int, default=4)
    parser.add_argument("--stage_b_unfreeze_last_nlayers", type=int, default=1)
    parser.add_argument("--stage_b_freeze_decoder", action="store_true")
    parser.add_argument("--stage_b_max_memory_tokens", type=int, default=320)
    args = parser.parse_args()

    common_prefix = [args.python, str(TRAIN_SCRIPT)]

    stage_a = common_prefix + [
        "--pre_ckpt",
        args.stage_a_pre_ckpt,
        "--train_pt",
        args.stage_a_train_pt,
        "--val_pt",
        args.stage_a_val_pt,
        "--emotion_space",
        "meld",
        "--decoder_model",
        args.decoder_model,
        "--epochs",
        str(args.stage_a_epochs),
        "--bs",
        str(args.stage_a_bs),
        "--lr_fusion",
        str(args.stage_a_lr_fusion),
        "--lr_decoder",
        str(args.stage_a_lr_decoder),
        "--max_target_len",
        str(args.stage_a_max_target_len),
        "--max_new_tokens",
        str(args.stage_a_max_new_tokens),
        "--num_beams",
        str(args.stage_a_num_beams),
        "--max_memory_tokens",
        str(args.stage_a_max_memory_tokens),
        "--out_pt",
        args.stage_a_out,
    ]

    stage_b = common_prefix + [
        "--pre_ckpt",
        args.stage_a_out,
        "--train_pt",
        args.stage_b_train_pt,
        "--emotion_space",
        "ko",
        "--decoder_model",
        args.decoder_model,
        "--epochs",
        str(args.stage_b_epochs),
        "--bs",
        str(args.stage_b_bs),
        "--lr_fusion",
        str(args.stage_b_lr_fusion),
        "--lr_decoder",
        str(args.stage_b_lr_decoder),
        "--val_ratio",
        str(args.stage_b_val_ratio),
        "--max_target_len",
        str(args.stage_b_max_target_len),
        "--max_new_tokens",
        str(args.stage_b_max_new_tokens),
        "--num_beams",
        str(args.stage_b_num_beams),
        "--max_memory_tokens",
        str(args.stage_b_max_memory_tokens),
        "--freeze_fusion_backbone",
        "--unfreeze_last_nlayers",
        str(args.stage_b_unfreeze_last_nlayers),
        "--out_pt",
        args.stage_b_out,
    ]
    if args.stage_b_freeze_decoder:
        stage_b.append("--freeze_decoder")

    run_step("stage_a_meld_rich_alignment", stage_a, args.dry_run)
    run_step("stage_b_korean_rich_finetune", stage_b, args.dry_run)


if __name__ == "__main__":
    main()
