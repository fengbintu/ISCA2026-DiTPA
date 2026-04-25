## Documention for `dataset/`

This directory contains LIEBRO robotic datasets for DiTPA evaluation.

#### Command for dataset downloading: 
```bash
echo "===== Downloading dataset and checkpoint... ====="
huggingface-cli download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir ./dataset \
  --local-dir-use-symlinks False
```