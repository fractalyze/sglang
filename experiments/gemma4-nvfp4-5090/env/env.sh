# Source on a gemma4nv host before any gate command.
# bs3: the root disk is full, so every cache/tmp dir lives under /data.
# bs2: /data is nearly full, so the venv and caches are W2's under /home and
#      the model is W2's HF-cache snapshot; only small study files go to /data.
export G4=/data/jooman/gemma4nv
case "$(hostname)" in
  build-server-2)
    export G4_HOME=/home/jooman/gemma4nv
    export G4_MODEL_DIR=$G4/hf/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/a19cfe00be84568a6867111c9a68c9c44fdcffe6
    ;;
  *)
    export G4_HOME=$G4
    export G4_MODEL_DIR=$G4/models/Gemma-4-26B-A4B-NVFP4
    ;;
esac
export G4_VENV=$G4_HOME/venv
export TMPDIR=$G4_HOME/tmp UV_CACHE_DIR=$G4_HOME/cache/uv PIP_CACHE_DIR=$G4_HOME/cache/pip
export HF_HOME=$G4/hf XDG_CACHE_HOME=$G4_HOME/cache/xdg
export TRITON_CACHE_DIR=$G4_HOME/cache/triton TORCHINDUCTOR_CACHE_DIR=$G4_HOME/cache/inductor
export FLASHINFER_WORKSPACE_BASE=$G4_HOME/cache/flashinfer CUDA_CACHE_PATH=$G4_HOME/cache/nv
export TVM_FFI_CACHE_DIR=$G4_HOME/cache/tvm-ffi CUTE_DSL_CACHE_DIR=$G4_HOME/cache/cute-dsl
export SGLANG_CACHE_DIR=$G4_HOME/cache/sglang SGLANG_BUILD_RUST_EXTS=none
# Host-safety protocol: JIT builds never fan out to nproc+2 parallel nvcc jobs
# (that OOM-killed both hosts on 2026-10-02). FlashInfer and tvm_ffi read MAX_JOBS.
export MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME=/usr/local/cuda-13 PATH=/usr/local/cuda-13/bin:$HOME/.local/bin:$PATH
mkdir -p "$TMPDIR"
if [ -f "$G4_VENV/bin/activate" ]; then source "$G4_VENV/bin/activate"; fi
