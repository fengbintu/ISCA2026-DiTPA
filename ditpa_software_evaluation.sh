echo "========================================"
echo "   DiTPA Software Evaluation Script"
echo "========================================"

# ==================== 1. activate DiTPA environmrnt ====================
echo "===== 1. Activating DiTPA environmrnt... ====="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ditpa

if [ "$CONDA_DEFAULT_ENV" != "ditpa" ]; then
    echo "❌ Failed to activate ditpa environment!"
    exit 1
fi
echo "Current environment: $CONDA_DEFAULT_ENV"

# ==================== 2. set config path and clear distributed variables ====================
echo "===== 2. Setting config path and clearing distributed variables... ====="

# determine libero config path
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/.libero"

# not use distributed mode
unset SLURM_PROCID SLURM_NTASKS SLURM_STEP_NODELIST SLURM_NTASKS_PER_NODE
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT

# ==================== 3. run baseline software evaluation ====================
echo "===== 3. Running baseline software evaluation... ====="

export TEST_LABEL="baseline"
python scripts/ditpa_software_evaluation.py

# ==================== 4. run DiTPA software evaluation ====================
echo "===== 4. Running DiTPA software evaluation... ====="

# config for DiTPA evaluation
export ACTION_SKIP_FLAG=true
export ITER_SKIP_FLAG=true
export MODALITY_SKIP_FLAG=true
export TEST_LABEL="ditpa"
python scripts/ditpa_software_evaluation.py

# ==================== 5. complete ====================
echo ""
echo "============================================"
echo "✅ DiTPA evaluation completed successfully!"
echo "============================================"