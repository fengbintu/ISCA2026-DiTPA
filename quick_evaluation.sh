echo "========================================"
echo "   Quick Evaluation Script"
echo "========================================"

# ==================== 1. prepare the repository ====================
echo "===== 1. Preparing DiTPA repository... ====="

REPO_URL="https://github.com/fengbintu/ISCA2026-DiTPA.git"
REPO_DIR="DiTPA"

if [ -d ".git" ]; then
    echo "Repo already exists at ./${REPO_DIR}, skipping clone."
else
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}" || { echo "Failed to enter directory"; exit 1; }
fi

# ==================== 2. create conda environmrnt ====================
ENV_NAME="ditpa"
PYTHON_VERSION="3.10"

echo "===== 2. Preparing conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})... ====="

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
    echo "❌ Failed to activate ${ENV_NAME} environment!"
    exit 1
fi
echo "Current environment: ${CONDA_DEFAULT_ENV}"

if [ "${CREATED_ENV}" -eq 1 ]; then
    echo "===== Installing minimal packages for hardware evaluation... ====="
    conda install -n "${ENV_NAME}" -y numpy pandas matplotlib openpyxl
fi

# ==================== 3. run DiTPA hardware evaluation ====================
echo "===== 3. Running DiTPA hardware evaluation... ====="

# start evaluation
python scripts/ditpa_hardware_evaluation.py \
    --baseline_sw_res_path ./outputs/example_output/baseline_software_results.xlsx \
    --ditpa_sw_res_path ./outputs/example_output/ditpa_software_results.xlsx

# ==================== 4. complete ====================
echo ""
echo "============================================"
echo "Results saved to ./results/ditpa_evaluation_results.txt"
echo "✅ DiTPA evaluation completed successfully!"
echo "============================================"