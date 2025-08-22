import argparse
import csv
import statistics
from collections import defaultdict

# read a csv file passed as an argument
# header will be accuracy,correctly_filled_cells,not_fully_unmasked,checkpoint,strategy,params
# example line:
# 0.8095,0.9680,0.0000,2025-06-18/12-51-55/checkpoints/latest/,gidd_emulate_mdlm_adaptive_score_select_update,"score_mask_position=MDM_max select_position=top_k_gumbel unmask_token=MDM_max k=1 gumbel_noise_coefficient=0 dataset=easy num_samples=6400 num_denoising_steps=81 batch_size=64 min_p=0 compile_torch=0 seed=1"
# the seed parameter will always be present
# TODO: aggregate accuravy and correctly_filled_cells over the seeds (mean and std) and write the results to <csv_file_name>_aggregated.csv
def parse_params(params_str):
    params = {}
    for item in params_str.split():
        if '=' in item:
            key, value = item.split('=', 1)
            params[key] = value
    return params

def main(args):
    csv_file = args.csv_file
    results = defaultdict(lambda: {'accuracy': [], 'correctly_filled_cells': []})

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f, fieldnames=[
            'accuracy', 'correctly_filled_cells', 'not_fully_unmasked',
            'checkpoint', 'strategy', 'params', 'denoised_fraction', 'mask_fraction', 'uniform_fraction', 'history'
        ])
        reader.__next__()  # Skip header row
        for row in reader:
            params = parse_params(row['params'])
            # Remove seed from params to aggregate over seeds
            params_no_seed = {k: v for k, v in params.items() if k != 'seed'}
            key = (row['strategy'], row['checkpoint'], tuple(sorted(params_no_seed.items()))) # TODO: sort according to custom order to make analysis easier
            results[key]['accuracy'].append(float(row['accuracy']))
            results[key]['correctly_filled_cells'].append(float(row['correctly_filled_cells']))

    output_file = csv_file.replace('.csv', '_aggregated.csv')
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow([ 
            'accuracy_mean', 'accuracy_std', 
            'correctly_filled_cells_mean', 'correctly_filled_cells_std',
            'strategy', 'params',
        ])
        for (strategy, checkpoint, params_tuple), metrics in results.items():
            params_str = " ".join(f"{k}={v}" for k, v in params_tuple)
            acc_mean = statistics.mean(metrics['accuracy'])
            acc_std = statistics.stdev(metrics['accuracy']) if len(metrics['accuracy']) > 1 else 0.0
            cfc_mean = statistics.mean(metrics['correctly_filled_cells'])
            cfc_std = statistics.stdev(metrics['correctly_filled_cells']) if len(metrics['correctly_filled_cells']) > 1 else 0.0
            writer.writerow([
                f"{acc_mean:.4f}", f"{acc_std:.4f}", 
                f"{cfc_mean:.4f}", f"{cfc_std:.4f}",
                strategy, checkpoint, params_str,
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file with evaluation results')
    args = parser.parse_args()
    main(args)