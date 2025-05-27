import hydra
import torch
import os

from gidd.checkpoints import load_checkpoint
from gidd.utils import score_sudoku

@hydra.main(config_path="../configs", config_name="score_sudoku", version_base="1.1")
def main(args):
    run_dir = args.path
    ckpt_path = os.path.join(hydra.utils.to_absolute_path(run_dir), "checkpoints/latest")

    _, _, tokenizer, _ = load_checkpoint(ckpt_path, device="cpu")

    samples_dir = os.path.join(hydra.utils.to_absolute_path(run_dir), "samples")
    for difficulty in os.listdir(samples_dir):
        difficulty_dir = os.path.join(samples_dir, difficulty)
        if not os.path.isdir(difficulty_dir):
            continue

        samples_pre_correction_path = os.path.join(difficulty_dir, "evaluation_samples_pre_correction.pt")
        samples_post_correction_path = os.path.join(difficulty_dir, "evaluation_samples_post_correction.pt")
        diffusion_mask_path = os.path.join(difficulty_dir, "evaluation_diffusion_mask.pt")
        solutions_path = os.path.join(difficulty_dir, "evaluation_solutions.pt")

        if not all(os.path.exists(p) for p in [samples_pre_correction_path, diffusion_mask_path, solutions_path]):
            print(f"Skipping {difficulty_dir}, missing files.")
            continue

        samples_pre_correction = torch.load(samples_pre_correction_path, weights_only=True)
        # samples_post_correction = torch.load(samples_post_correction_path, weights_only=True)
        diffusion_mask = torch.load(diffusion_mask_path, weights_only=True)
        solutions = torch.load(solutions_path, weights_only=True)

        metrics_pre_correction = score_sudoku(samples_pre_correction, diffusion_mask, solutions, tokenizer)
        # metrics_post_correction = score_sudoku(samples_post_correction, diffusion_mask, solutions, tokenizer)
        print(f"\nPre-correction metrics for {difficulty}:")
        for key, value in metrics_pre_correction.items():
            print(f"{key}: {value:.4f}")
        # print(f"\nPost-correction metrics for {difficulty}:")
        # for key, value in metrics_post_correction.items():
        #     print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()