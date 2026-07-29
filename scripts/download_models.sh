#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-core}"
command -v hf >/dev/null || {
  echo "Missing 'hf'. Install requirements.txt first." >&2
  exit 1
}

mkdir -p models

download_core() {
  hf download black-forest-labs/FLUX.1-dev --local-dir models/FLUX.1-dev
  hf download black-forest-labs/FLUX.1-Kontext-dev --local-dir models/FLUX.1-Kontext-dev
  hf download black-forest-labs/FLUX.2-klein-9B --local-dir models/klein9b
}

download_scores() {
  hf download facebook/dinov2-large --local-dir models/dinov2
  hf download openai/clip-vit-large-patch14 --local-dir models/clip-vit-large-patch14
}

download_baselines() {
  hf download XLabs-AI/flux-ip-adapter --local-dir models/flux-ip-adapter
  hf download bytedance-research/USO --include 'uso_flux_v1.0/*' --local-dir models/USO
  hf download InstantX/CSGO csgo_4_32.bin --local-dir models/CSGO
  hf download Gaojunyao/StyleShot --local-dir models/StyleShot
  hf download Gaojunyao/StyleShot_lineart --local-dir models/StyleShot_lineart
  hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir models/stable-diffusion-v1-5
  hf download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
}

case "$mode" in
  core) download_core ;;
  scores) download_scores ;;
  baselines) download_baselines ;;
  all) download_core; download_scores; download_baselines ;;
  *) echo "Usage: $0 [core|scores|baselines|all]" >&2; exit 2 ;;
esac

echo "Model download complete (mode=$mode)."
