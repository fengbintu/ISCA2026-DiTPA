echo "========================================"
echo "   DiTPA Environment Installation Script"
echo "========================================"

# ==================== 1. prepare the repository ====================
echo "===== 1. Preparing DiTPA repository... ====="

REPO_URL="https://github.com/j9h5f2m8k/DiTPA.git"
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

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env ${ENV_NAME} already exists, skipping create."
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

if [ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]; then
    echo "❌ Failed to activate ${ENV_NAME} environment!"
    exit 1
fi
echo "Current environment: ${CONDA_DEFAULT_ENV}"

# ==================== 3. install PyTorch (CUDA 12.1) ====================
echo "===== 3. Installing PyTorch with CUDA 12.1... ====="
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# ==================== 4. instal other dependencies: mujoco, robosuite, libero simulation platform, etc. ====================
echo "===== 4. Installing dependencies from requirements.txt... ====="
pip install -r requirements.txt
cp -r src/libero/libero/ ./

# ==================== 5. install CUDA Toolkit and nvcc ====================
echo "===== 5. Installing CUDA Toolkit and nvcc... ====="
conda install -c nvidia cuda-nvcc=12.1 -y
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit -y

# ==================== 6. set up environment variables ====================
echo "===== 6. Setting up environment variables... ====="
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

# ==================== 7. install PyTorch3D ====================
echo "===== 7. Installing PyTorch3D... ====="
pip install "git+https://github.com/facebookresearch/pytorch3d.git@0a59450f0ebbe12d9a8db3de937814932517633b" \
    --no-build-isolation --no-cache-dir

# ==================== 8. download dataset and checkpoint ====================
echo "===== 8. Downloading dataset and checkpoint... ====="
huggingface-cli download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir ./dataset \
  --local-dir-use-symlinks False

huggingface-cli download j9h5f2m8k/DiTPA-checkpoints \
  --local-dir ./checkpoint \
  --local-dir-use-symlinks False

# ==================== 9. configure simulation platform and patch source files ====================
# needed when error occurs: AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
echo "===== 9.1 Setting MuJoCo rendering backend to osmesa... ====="
export MUJOCO_GL=osmesa

cat >> ~/.bashrc << EOF
export MUJOCO_GL=osmesa
EOF

echo "===== 9.2 Patching source files... ====="
DST="$(python -c "import importlib.util; print(importlib.util.find_spec('transformers.models.llama.modeling_llama').origin)")"
cp "scripts/modeling_llama.py" "$DST"

echo "===== 9.3 Configuring simulation platform LIBERO paths... ====="
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

# ==================== 10. complete ====================
echo ""
echo "============================================"
echo "✅ Environment installation completed successfully!"
echo "============================================"
