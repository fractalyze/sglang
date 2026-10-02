#!/bin/bash
# One-time env build on build-server-3: venv + editable SGLang from src-gate.
# On build-server-2 the venv is W2's (same base commit); only the extra deps are added.
set -euxo pipefail
source "$(dirname "$0")/env.sh"
cd $G4
[ -d $G4_VENV ] || uv venv -p 3.12 $G4_VENV
source $G4_VENV/bin/activate
uv pip install -e "src-gate/python"
uv pip install absl-py datasets huggingface_hub
python -c "import torch, sgl_kernel, flashinfer; print('torch', torch.__version__, torch.version.cuda, 'sgl_kernel', sgl_kernel.__version__, 'flashinfer', flashinfer.__version__)"
echo SETUP-DONE
