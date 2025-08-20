# Compute Where It Counts (CWIC)


## Installation
```sh
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc

pixi shell

wandb login
# gcloud auth application-default login --no-launch-browser

python scripts/train_cwic.py
```