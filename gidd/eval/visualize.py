import torch
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np

def sample_history_to_str(history, solution):
    num_steps, seq_len = history.shape
    history_str = f"\"num_steps={num_steps} seq_len={seq_len} history="
    history_str += " ".join([" ".join([str(token.item()) for token in step]) for step in history])
    history_str += " "
    history_str += " ".join([str(token.item()) for token in solution])
    history_str += "\""
    return history_str

def history_to_str(history, solution):
    shape = history.shape
    if len(shape) == 2:
        return sample_history_to_str(history, solution)
    elif len(shape) == 3:
        raise NotImplementedError("Only history of a single sample is supported, history shape must be (num_steps, seq_len)")

def history_str_to_tensor(history_str):
    history_str_split = history_str.strip('"').split(" ")
    num_steps = history_str_split[0]
    seq_len = history_str_split[1]
    history = " ".join(history_str_split[2:])
    num_steps = int(num_steps.split("=")[1])
    seq_len = int(seq_len.split("=")[1])
    history = history.split("=")[1].split(" ")
    history = [int(token) for token in history]
    history_tensor = torch.tensor(history).view(num_steps + 1, seq_len)
    solution_tensor = history_tensor[-1, :].clone()
    history_tensor = history_tensor[:-1, :]
    return history_tensor, solution_tensor
    
def visualize_history(history, solution, mask_token_id):
    def print_grid(step, solution, mask_token_id):
        for i in range(9):
            row_str = ""
            for j in range(9):
                idx = i * 9 + j
                token = step[idx].item()
                if token == mask_token_id:
                    row_str += ". "
                else:
                    correct = token == solution[idx].item()
                    color = "\033[92m" if correct else "\033[91m"
                    row_str += f"{color}{token}\033[0m "
            print(row_str)
    print("Solution:")
    print_grid(solution, solution, mask_token_id)
    print("\nSampling steps:")
    for step_idx, step in enumerate(history):
        print(f"\nStep {step_idx}:")
        print_grid(step, solution, mask_token_id)


def visualize_history_as_video(history, mask_token_id, output_path="outputs/evaluate_all/sampling_process.gif"):
    history = history[:-1, :]  # exclude final solution
    solution = history[-1, :]  # optionally used for highlighting correctness

    fig, ax = plt.subplots()
    ax.axis('off')

    def update(frame):
        ax.clear()
        ax.set_title(f"Step {frame}", fontsize=16)
        step = history[frame]
        grid = step.reshape(9, 9)
        sol_grid = solution.reshape(9, 9)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(np.zeros((9, 9)), cmap='gray_r', vmin=0, vmax=9)  # background
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5  # Line width: bold every 3 steps
            ax.axhline(i - 0.5, color='black', lw=lw)
            ax.axvline(i - 0.5, color='black', lw=lw)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(8.5, -0.5)

        for (i, j), token in np.ndenumerate(grid):
            if token == mask_token_id:
                text = "."
                color = "gray"
            else:
                correct = token == sol_grid[i, j].item()
                color = "green" if correct else "red"
                text = str(token.item())
            ax.text(j, i, text, ha='center', va='center', fontsize=14, color=color)

    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False)
    ani.save(output_path, writer='pillow', fps=3)
    print(f"Saved animation to {output_path}")

def visualize_history_as_video_multi(histories, mask_token_id, output_path="outputs/evaluate_all/sampling_process_multi.gif"):
    num_samples = len(histories)
    checkpoints = [h['checkpoint'] for h in histories]
    strategies = [h['strategy'] for h in histories]
    params = [h['params'] for h in histories]
    solutions = [h['solution'] for h in histories]
    histories = [h['history'] for h in histories]

    max_steps = max(h.shape[0] for h in histories)
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))

    if num_samples == 1:
        axes = [axes]

    for ax in axes:
        ax.axis('off')

    grid_size = int((histories[0].shape[1]) ** 0.5)
    box_size = int(grid_size ** 0.5)

    fixed_positions = [(h[0].reshape(grid_size, grid_size) != mask_token_id) for h in histories]

    def update(frame):
        for i, ax in enumerate(axes):
            ax.clear()
            ax.set_title(f"{strategies[i]} - Step {frame}", fontsize=10)
            if frame >= histories[i].shape[0]:
                step = histories[i][-1]
            else:
                step = histories[i][frame]

            grid = step.reshape(grid_size, grid_size)
            sol_grid = solutions[i].reshape(grid_size, grid_size)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray_r', vmin=0, vmax=9)

            for j in range(grid_size + 1):
                lw = 2 if j % box_size == 0 else 0.5
                ax.axhline(j - 0.5, color='black', lw=lw)
                ax.axvline(j - 0.5, color='black', lw=lw)
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(grid_size - 0.5, -0.5)

            for (r, c), token in np.ndenumerate(grid):
                if token == mask_token_id:
                    text = "."
                    color = "gray"
                else:
                    text = str(token.item())
                    if fixed_positions[i][r, c]:
                        color = "black"
                    else:
                        correct = token == sol_grid[r, c].item()
                        color = "green" if correct else "red"
                ax.text(c, r, text, ha='center', va='center', fontsize=12, color=color)

    ani = animation.FuncAnimation(fig, update, frames=max_steps, repeat=False)
    ani.save(output_path, writer='pillow', fps=1)
    print(f"Saved multi-sample animation to {output_path}")

def visualize_final_grid_with_update_gradient_multi(histories, mask_token_id, save_path):
    num_samples = len(histories)
    grid_size = int(histories[0]['history'].shape[1] ** 0.5)
    box_size = int(grid_size ** 0.5)
    cmap = plt.get_cmap('coolwarm')

    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6), constrained_layout=True)
    if num_samples == 1:
        axes = [axes]

    for ax, history_entry in zip(axes, histories):
        history = history_entry['history']
        solution = history_entry['solution']
        first_step = history[0]
        final_step = history[-1]
        num_steps, seq_len = history.shape

        update_steps = [[] for _ in range(seq_len)]
        for step_idx in range(1, num_steps):
            changed = (history[step_idx] != history[step_idx - 1])
            for i in torch.where(changed)[0]:
                update_steps[i.item()].append(step_idx)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_aspect('equal')

        for j in range(grid_size + 1):
            lw = 2 if j % box_size == 0 else 0.5
            ax.axhline(j - 0.5, color='black', lw=lw)
            ax.axvline(j - 0.5, color='black', lw=lw)

        for idx in range(seq_len):
            r, c = divmod(idx, grid_size)
            updates = update_steps[idx]
            num_updates = len(updates)
            init_token = first_step[idx].item()
            final_token = final_step[idx].item()
            target_token = solution[idx].item()

            if num_updates == 0:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, color='gray', alpha=0.2))
            else:
                for j, update_step in enumerate(updates):
                    color = cmap(update_step / (num_steps - 1))
                    width = 1.0 / num_updates
                    ax.add_patch(
                        patches.Rectangle(
                            (c - 0.5 + j * width, r - 0.5),
                            width, 1,
                            color=color
                        )
                    )

            # Digit content
            if final_token == mask_token_id:
                text = "."
                digit_color = 'lightgray'
            else:
                text = str(final_token)
                if init_token != mask_token_id:
                    digit_color = 'black'
                else:
                    digit_color = 'black' if final_token == target_token else 'red'

            ax.text(c, r, text, ha='center', va='center', fontsize=12, color=digit_color)

            # Step number if exactly one update
            if num_updates == 1:
                step_str = str(updates[0])
                ax.text(c - 0.4, r - 0.35, step_str, ha='left', va='top',
                        fontsize=6, color='black', fontweight='normal')

        ax.set_title(f"{history_entry['strategy']}", fontsize=12)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar.set_label("Update Time (Early → Late)", fontsize=12)

    plt.savefig(save_path)
    plt.close()
    print(f"Saved image to {save_path}")



def visualize_sampling_process(csv_file, mask_token_id, rows_to_visualize, visualization_type='print'):
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f, fieldnames=[
            'accuracy', 'correctly_filled_cells', 'not_fully_unmasked',
            'checkpoint', 'strategy', 'params', 'history'
        ])
        next(reader)
        histories = []
        for row in reader:
            history_str = row['history']
            history, solution = history_str_to_tensor(history_str)
            histories.append({
                "history": history,
                "solution": solution,
                "checkpoint": row['checkpoint'],
                "strategy": row['strategy'],
                "params": row['params'],
            })
        selected_histories = [histories[i] for i in rows_to_visualize]
        if visualization_type == 'print':
            for i, history in zip(rows_to_visualize, selected_histories):
                print(f"Sample {i}:")
                visualize_history(history['history'], history['solution'], mask_token_id)
                print("\n" + "-" * 50 + "\n")
        elif visualization_type == 'video':
            visualize_history_as_video_multi(selected_histories, mask_token_id)
        elif visualization_type == 'image':
            filename = "outputs/evaluate_all/final_grid_updates_multiple.png"
            visualize_final_grid_with_update_gradient_multi(selected_histories, mask_token_id, filename)


def main(args):
    visualize_sampling_process(args.csv_file, args.mask_token_id, args.rows, args.visualization_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file with evaluation results')
    parser.add_argument('--mask_token_id', type=int, default=0, help='Token ID of the mask token')
    parser.add_argument('--rows', type=int, nargs='+', default=[0],
                        help='Indices of rows to visualize from the CSV file')
    parser.add_argument('--visualization_type', type=str, choices=['print', 'video', 'image'], default='print',
                        help='Type of visualization to perform: "print" for console output, "video" for animation')
    args = parser.parse_args()
    main(args)