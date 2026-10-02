export G=/data/jooman/gemma4nv
export HG=/home/jooman/gemma4nv
export HF_HOME=$G/hf
export TMPDIR=$HG/tmp
export UV_CACHE_DIR=$HG/cache/uv
export PIP_CACHE_DIR=$HG/cache/pip
export TRITON_CACHE_DIR=$HG/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$HG/cache/inductor
export FLASHINFER_WORKSPACE_BASE=$HG/cache/flashinfer
export XDG_CACHE_HOME=$HG/cache/xdg
export CUDA_CACHE_PATH=$HG/cache/nv
export PATH=$HOME/.local/bin:$PATH
[ -f $HG/venv/bin/activate ] && source $HG/venv/bin/activate
export SGLANG_BUILD_RUST_EXTS=none
