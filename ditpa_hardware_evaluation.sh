echo "========================================"
echo "   DiTPA Hardware Evaluation Script"
echo "========================================"

# ==================== 1. activate DiTPA environmrnt ====================
echo "===== 1. Activating DiTPA environmrnt... ====="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ditpa

if [ "$CONDA_DEFAULT_ENV" != "ditpa" ]; then
    echo "❌ Failed to activate DiTPA environment!"
    exit 1
fi
echo "Current environment: $CONDA_DEFAULT_ENV"

# ==================== 2. read the latest software evaluation outputs ====================
echo "===== 2. Reading latest software evaluation outputs... ====="

# find latest software evaluation results
BASELINE_TXT="./outputs/latest_baseline_output_path.txt"
DITPA_TXT="./outputs/latest_ditpa_output_path.txt"

[ -f "$BASELINE_TXT" ] || { echo "❌ Missing $BASELINE_TXT"; exit 1; }
[ -f "$DITPA_TXT" ] || { echo "❌ Missing $DITPA_TXT"; exit 1; }

baseline_sw_res_path="$(head -n 1 "$BASELINE_TXT" | tr -d '\r\n')"
ditpa_sw_res_path="$(head -n 1 "$DITPA_TXT" | tr -d '\r\n')"

[ -f "$baseline_sw_res_path" ] || { echo "❌ Baseline xlsx not found: $baseline_sw_res_path"; exit 1; }
[ -f "$ditpa_sw_res_path" ] || { echo "❌ DiTPA xlsx not found: $ditpa_sw_res_path"; exit 1; }

# ==================== 3. run DiTPA hardware evaluation ====================
echo "===== 3. Running DiTPA hardware evaluation... ====="
# start evaluation
python scripts/ditpa_hardware_evaluation.py \
  --baseline_sw_res_path "$baseline_sw_res_path" \
  --ditpa_sw_res_path "$ditpa_sw_res_path"

# ==================== 4. complete ====================
echo ""
echo "============================================"
echo "Results saved to ./results/ditpa_evaluation_results.txt"
echo "✅ DiTPA evaluation completed successfully!"
echo "============================================"