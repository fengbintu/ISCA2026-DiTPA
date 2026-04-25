## Documention for `outputs/`

This directory contains intermediate outputs produced by software evaluation runs. Software evaluation outputs are organized under Hydra time-stamped folders. A typical layout includes:

- `outputs/<date>/<time>/intermediate_data/logs/`: evaluation logs (e.g., baseline_eval_record.log, ditpa_eval_record.log)

- `outputs/<date>/<time>/intermediate_data/videos/`: visualization videos (if enabled in the code)

- `outputs/<date>/<time>/intermediate_data/trajectories/`: trajectories of all tasks

- `outputs/<date>/<time>/intermediate_data/software_res/`: exported software results (e.g., *_software_results.xlsx, and sample_task.video)

In addition, the pipeline writes “latest software result path” markers under `outputs/` for hardware evaluation. Each file contains the absolute path to the corresponding exported .xlsx.

- `outputs/latest_baseline_output_path.txt`
- `outputs/latest_ditpa_output_path.txt`

For the quick evaluation, we saved example software results in `outputs/example_output/`

- `baseline_software_results.xlsx`
- `ditpa_software_results.xlsx`