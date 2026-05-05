# DiTPA: A DiT-based Action Planner Accelerator Exploiting Action-Denoising-Multimodality Redundancy for Embodied Artificial Intelligence

[**Project Overview**](#1-project-overview) | [**Quick Evaluation**](#2-quick-evaluation) | [**Environment Installation**](#3-environment-installation) | [**DiTPA Software Evaluation**](#4-ditpa-software-evaluation) | [**DiTPA Hardware Evaluation**](#5-ditpa-hardware-evaluation)

---

## 1. Project Overview

This project reproduces and evaluates the DiTPA architecture with multi-level redundancy exploitation, aiming to provide a reference for related research. To make evaluation convenient and reproducible, we provide one-click scripts for environment setup, software evaluation, and hardware evaluation. The evaluation outputs include key results such as success rate, action frequency, task execution time, energy efficiency, and visualization artifacts.

### 1.1 Quick evaluation

To evaluate DiTPA more quickly, we provide a quick evaluation script ([quick_evaluation.sh](quick_evaluation.sh)) with minimal hardware requirements and environment setting effort.

### 1.2 Full evaluation

A complete evaluation consists of three steps:
- Environment setup ([environment_setup.sh](environment_setup.sh))
- DiTPA software evaluation ([ditpa_software_evaluation.sh](ditpa_software_evaluation.sh))
- DiTPA hardware evaluation ([ditpa_hardware_evaluation.sh](ditpa_hardware_evaluation.sh))

Each step has a corresponding one-click script. Full evaluation script ([full_evaluation.sh](full_evaluation.sh)) includes the above three steps and can execute them automatically in sequence.

In our typical setup, we run experiments on an NVIDIA A40 GPU with CUDA 12.1, Python 3.10, and PyTorch 2.2. The total disk usage for the running environment and project is about 28 GB. A full evaluation typically takes around 3 days (depending on hardware configuration).

### 1.3 Prerequisites

Before running DiTPA, make sure your system satisfies the following requirements:

- OS: Linux (recommended: Ubuntu 22.04)
- Python: 3.10
- Conda: Anaconda (required for environment setup)

Optional (for full evaluation)
- GPU: NVIDIA GPU (recommended: A40)
- CUDA: 12.1-compatible environment

### 1.4 Repository structure

- root directory: evaluation entry scripts and top-level utilities
- `scripts/`: DiTPA software implementation and the hardware simulator
- `config/`: configuration files for DiTPA evaluations
- `data_process/`: dataset, checkpoint, and simulation-environment processing utilities
- `checkpoint/`: model checkpoints
- `dataset/`: evaluation datasets
- `outputs/`: intermediate outputs produced by software evaluation runs
- `results/`: final evaluation results
- `utils/`: helper utilities (format conversion, Excel export, logging, etc.)


## 2. Quick Evaluation

Quick evaluation contains three steps: repository preparation, conda environment setup, and DiTPA hardware evaluation. Since DiTPA software evaluation requires a full simulator stack and complete datasets, quick evaluation directly uses the example software results in `outputs/example_output/` for hardware evaluation. The detailed operations are implemented in [quick_evaluation.sh](quick_evaluation.sh).

### 2.1 One-command run

Clone the repository.
```bash
git clone "https://github.com/j9h5f2m8k/DiTPA.git" "DiTPA"
cd DiTPA
```

Then, run the command.
```bash
./quick_evaluation.sh
```

### 2.2 Step-by-step operations
#### Step 1: Prepare the repository
It clones the repository (or skips cloning if `./DiTPA/.git` already exists), then enters the repo directory.

```bash
REPO_URL="https://github.com/j9h5f2m8k/DiTPA.git"
REPO_DIR="DiTPA"

if [ -d ".git" ]; then
    echo "Repo already exists at ./${REPO_DIR}, skipping clone."
else
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}" || { echo "Failed to enter directory"; exit 1; }
fi
```

#### Step 2: Create and activate a minimal conda environment

It creates a conda environment named ditpa (Python 3.10) if it does not exist, activates it, then installs minimal Python packages needed for hardware evaluation.

```bash
ENV_NAME="ditpa"
PYTHON_VERSION="3.10"

source "$(conda info --base)/etc/profile.d/conda.sh"

CREATED_ENV=0
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env ${ENV_NAME} already exists, skipping create."
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    CREATED_ENV=1
fi

conda activate "${ENV_NAME}"

if [ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]; then
    echo "Failed to activate ${ENV_NAME} environment!"
    exit 1
fi
echo "Current environment: ${CONDA_DEFAULT_ENV}"

if [ "${CREATED_ENV}" -eq 1 ]; then
    echo "===== Installing minimal packages for hardware evaluation... ====="
    conda install -n "${ENV_NAME}" -y numpy pandas matplotlib openpyxl
fi
```

#### Step 3: Run DiTPA hardware evaluation

It runs [scripts/ditpa_hardware_evaluation.py](scripts/ditpa_hardware_evaluation.py) using example outputs, and output results in [results/ditpa_evaluation_results.txt](results/ditpa_evaluation_results.txt).

```bash
python scripts/ditpa_hardware_evaluation.py \
    --baseline_sw_res_path ./outputs/example_output/baseline_software_results.xlsx \
    --ditpa_sw_res_path ./outputs/example_output/ditpa_software_results.xlsx
```


## 3. Environment Installation

It installs the complete simulation environment required for the full evaluation, including:

- Repository preparation
- Dependency installation (PyTorch + simulator stack)
- CUDA toolkit / nvcc installation
- Dataset and model checkpoint download
- Simulator configuration (MuJoCo backend + LIBERO config)

All operations below are implemented in [environment_setup.sh](environment_setup.sh).

### 3.1 One-command run

```bash
./environment_setup.sh
```

### 3.2 Step-by-step operations
#### Step 1: Prepare the repository
It clones the repository (or skips cloning if `./DiTPA/.git` already exists), then enters the repo directory.

```bash
REPO_URL="https://github.com/j9h5f2m8k/DiTPA.git"
REPO_DIR="DiTPA"

if [ -d ".git" ]; then
    echo "Repo already exists at ./${REPO_DIR}, skipping clone."
else
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}" || { echo "Failed to enter directory"; exit 1; }
fi
```

#### Step 2: Create and activate a conda environment
It creates a conda environment named ditpa (Python 3.10) if it does not exist, then activates it.

```bash
ENV_NAME="ditpa"
PYTHON_VERSION="3.10"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env ${ENV_NAME} already exists, skipping create."
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

if [ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]; then
    echo "Failed to activate ${ENV_NAME} environment!"
    exit 1
fi
echo "Current environment: ${CONDA_DEFAULT_ENV}"
```

#### Step 3: Install PyTorch (CUDA 12.1)
It installs the PyTorch 2.2 from the CUDA 12.1 wheel index. Since installing PyTorch3D requires a dependency on PyTorch, it is necessary to install PyTorch in advance.

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
```

#### Step 4: Install other dependencies (MuJoCo / robosuite / LIBERO, etc.)
It installs dependencies from requirements.txt, then moves the LIBERO source code into the root directory.

```bash
pip install -r requirements.txt
cp -r src/libero/libero/ ./
```

#### Step 5: Install CUDA Toolkit and nvcc
It installs cuda-nvcc=12.1 and CUDA toolkit via conda.

```bash
conda install -c nvidia cuda-nvcc=12.1 -y
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit -y
```

#### Step 6: Set up environment variables
It sets and appends the following variables to `~/.bashrc` for PyTorch3D installation.

```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH

cat >> ~/.bashrc << EOF
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:\$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:\$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:\$LIBRARY_PATH
EOF
```

#### Step 7: Install PyTorch3D
It installs PyTorch3D from a pinned Git commit.

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@0a59450f0ebbe12d9a8db3de937814932517633b" \
    --no-build-isolation --no-cache-dir
```

#### Step 8: Download dataset and checkpoint
It downloads the LIBERO dataset into `./dataset/` using HuggingFace CLI and downloads the Dita model checkpoint into `./checkpoint`.

```bash
huggingface-cli download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir ./dataset \
  --local-dir-use-symlinks False

huggingface-cli download j9h5f2m8k/DiTPA-checkpoints \
  --local-dir ./checkpoint \
  --local-dir-use-symlinks False
```

#### Step 9: Configure simulator and patch source files

##### Step 9.1 Set MuJoCo rendering backend to osmesa

```bash
export MUJOCO_GL=osmesa
cat >> ~/.bashrc << EOF
export MUJOCO_GL=osmesa
EOF
```

##### Step 9.2 Patch HuggingFace Transformers LLaMA modeling file
The script locates the installed `transformers.models.llama.modeling_llama` path and overwrites it with `scripts/modeling_llama.py`.

```bash
DST="$(python -c "import importlib.util; print(importlib.util.find_spec('transformers.models.llama.modeling_llama').origin)")"
cp "scripts/modeling_llama.py" "$DST"
```

##### Step 9.3 Configure LIBERO paths
It creates `.libero/config.yaml` and points LIBERO to the correct repo-local paths.

```bash
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero"
mkdir -p "${LIBERO_CONFIG_PATH}"

cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${REPO_ROOT}/libero/libero
bddl_files: ${REPO_ROOT}/libero/libero/bddl_files
init_states: ${REPO_ROOT}/libero/libero/init_files
datasets: ${REPO_ROOT}/dataset
assets: ${REPO_ROOT}/libero/libero/assets
EOF
```


## 4. DiTPA Software Evaluation

After installing the complete simulation environment, run DiTPA software evaluation first. This step executes closed-loop evaluation in LIBERO environments and produces logs, exported software results (e.g., `.xlsx`), and task visualization videos. The entry script is [ditpa_software_evaluation.sh](ditpa_software_evaluation.sh), which runs [scripts/ditpa_software_evaluation.py](scripts/ditpa_software_evaluation.py).

### 4.1 One-command run

```bash
./ditpa_software_evaluation.sh
```

### 4.2 Step-by-step operations
#### Step 1: Activate the conda environment

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ditpa

if [ "$CONDA_DEFAULT_ENV" != "ditpa" ]; then
    echo "Failed to activate ditpa environment!"
    exit 1
fi
echo "Current environment: $CONDA_DEFAULT_ENV"
```

#### Step 2: Set config path and clear distributed variables

```bash
# determine libero config path
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero"

# not use distributed mode
unset SLURM_PROCID SLURM_NTASKS SLURM_STEP_NODELIST SLURM_NTASKS_PER_NODE
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
```

#### Step 3: Run baseline software evaluation

```bash
export TEST_LABEL="baseline"
python scripts/ditpa_software_evaluation.py
```

#### Step 4: Run DiTPA software evaluation

```bash
export ACTION_SKIP_FLAG=true
export ITER_SKIP_FLAG=true
export MODALITY_SKIP_FLAG=true
export TEST_LABEL="ditpa"
python scripts/ditpa_software_evaluation.py
```

### 4.3 Outputs
Software evaluation outputs are organized under Hydra time-stamped folders inside `outputs/`. A typical layout includes:

- `outputs/<date>/<time>/intermediate_data/logs/`: evaluation logs (e.g., baseline_eval_record.log, ditpa_eval_record.log)

- `outputs/<date>/<time>/intermediate_data/videos/`: visualization videos (if enabled in the code)

- `outputs/<date>/<time>/intermediate_data/trajectories/`: trajectories of all tasks

- `outputs/<date>/<time>/intermediate_data/software_res/`: exported software results (e.g., *_software_results.xlsx, and sample_task.video)

In addition, the pipeline writes “latest software result path” markers under `outputs/` for hardware evaluation. Each file contains the absolute path to the corresponding exported .xlsx.

- `outputs/latest_baseline_output_path.txt`
- `outputs/latest_ditpa_output_path.txt`


## 5. DiTPA Hardware Evaluation

After software evaluation, run DiTPA hardware evaluation to simulate hardware behavior and report derived metrics (e.g., performance results). The entry script is [ditpa_hardware_evaluation.sh](ditpa_hardware_evaluation.sh), which runs
[scripts/ditpa_hardware_evaluation.py](scripts/ditpa_hardware_evaluation.py).

### 5.1 One-command run

```bash
./ditpa_hardware_evaluation.sh
```
### 5.2 Step-by-step operations
#### Step 1: Activate the conda environment

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ditpa

if [ "$CONDA_DEFAULT_ENV" != "ditpa" ]; then
    echo "Failed to activate ditpa environment!"
    exit 1
fi
echo "Current environment: $CONDA_DEFAULT_ENV"
```

#### Step 2: Read the latest software evaluation outputs

```bash
BASELINE_TXT="./outputs/latest_baseline_output_path.txt"
DITPA_TXT="./outputs/latest_ditpa_output_path.txt"

[ -f "$BASELINE_TXT" ] || { echo "Missing $BASELINE_TXT"; exit 1; }
[ -f "$DITPA_TXT" ] || { echo "Missing $DITPA_TXT"; exit 1; }

baseline_sw_res_path="$(head -n 1 "$BASELINE_TXT" | tr -d '\r\n')"
ditpa_sw_res_path="$(head -n 1 "$DITPA_TXT" | tr -d '\r\n')"

[ -f "$baseline_sw_res_path" ] || { echo "Baseline xlsx not found: $baseline_sw_res_path"; exit 1; }
[ -f "$ditpa_sw_res_path" ] || { echo "DiTPA xlsx not found: $ditpa_sw_res_path"; exit 1; }
```

#### Step 3: Run DiTPA hardware evaluation

```bash
python scripts/ditpa_hardware_evaluation.py \
  --baseline_sw_res_path "$baseline_sw_res_path" \
  --ditpa_sw_res_path "$ditpa_sw_res_path"
```

### 5.3 Outputs

Hardware evaluation writes results to:

- `results/ditpa_evaluation_results.txt`

**Notes**: 

(1) If the latest_*.txt files do not exist, it usually means software evaluation has not finished successfully. You can also run the Python script directly and manually pass `--baseline_sw_res_path` and `--ditpa_sw_res_path` if needed. 

(2) Due to the random initialization of the LIBERO simulation environment and the diffusion workflow adopted by the DiT action planner, the task execution exhibits randomness, resulting in slight fluctuations in each evaluation result. 

## Citation:
If you find this repository or the paper helpful, we would appreciate it if you could cite our work:
```
@inproceedings{DiTPA2026,
  author    = {Xin Zhao and Longke Yan and Jiancong Li and Yongkun Wu and Fengbin Tu},
  title     = {DiTPA: A DiT-based Action Planner Accelerator Exploiting Action-Denoising-Multimodality Redundancy for Embodied Artificial Intelligence},
  booktitle = {53rd Annual International Symposium on Computer Architecture (ISCA)},
  year      = {2026},
  address   = {Raleigh, USA},
  month     = {June},  
}
```