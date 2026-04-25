## Documention for `scripts/`

This directory contains the implementation and evaluation codes for DiTPA (software and hardware).

#### Software Implementation and Evaluation
- Entry point: [ditpa_software_evaluation.py](ditpa_software_evaluation.py)
- Workflow: Loads models and datasets, runs task-wise inference, updates environment state in the LIBERO simulator, and saves results to the `outputs/` folder.
- Control loop: The full VLA model run is handled by [close_loop_eval.py](close_loop_eval.py).
- Components:
    - Vision Language Model (VLM): implemented under the `film_efficientnet/` and `vision_tokenizers/` directories.
    - Action planner: [action_planner_modeling.py](action_planner_modeling.py). Underlying Transformer backbone is [modeling_llama.py](modeling_llama.py).
- Runtime flags: [modeling_llama.py](modeling_llama.py) contains runtime parameters. Set `action_skip_flag`, `iter_skip_flag`, and `modality_skip_flag` to switch between baseline and DiTPA modes.

#### Hardware Implementation and Evaluation
- Entry point: [ditpa_hardware_evaluation.py](ditpa_hardware_evaluation.py)
- Workflow: Reads software outputs from `outputs/`, simulates the hardware execution pipeline, calculates computation latency and off-chip memory access, and reports performance metrics such as action frequency, task execution time, and energy efficiency.
