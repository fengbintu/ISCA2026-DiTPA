#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "   Full Evaluation Script"
echo "========================================"

echo "===== (1/3) Setup environment (environment_setup.sh) ====="
bash ./environment_setup.sh

echo "===== (2/3) Software evaluation (ditpa_software_evaluation.sh) ====="
bash ./ditpa_software_evaluation.sh

echo "===== (3/3) Hardware evaluation (ditpa_hardware_evaluation.sh) ====="
bash ./ditpa_hardware_evaluation.sh

echo ""
echo "============================================"
echo "✅ Full evaluation completed successfully!"
echo "============================================"