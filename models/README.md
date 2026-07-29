# Model files

Use `bash scripts/download_models.sh MODE` from the repository root.

```text
FLUX.1-dev/                       # core T2I
FLUX.1-Kontext-dev/               # core I2I
klein9b/                          # paper backbone
dinov2/                           # score mode
clip-vit-large-patch14/           # score/IP-Adapter mode
flux-ip-adapter/                  # baseline mode
USO/uso_flux_v1.0/                # baseline mode
CSGO/csgo_4_32.bin                # baseline mode
StyleShot/                        # baseline mode
StyleShot_lineart/                # baseline mode
stable-diffusion-v1-5/            # StyleShot base
Qwen-Image-Edit-2511/             # Qwen baseline
```

Large model files are intentionally ignored by Git. CSD weights live in the
separate sibling repository `../../CSD_Score/models/`.
