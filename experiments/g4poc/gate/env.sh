# Source on a g4poc host before any gate/workload command.
# Reuses the gemma4nv venv and JIT caches (same SGLang base, already built under
# the host-safety caps); study files live under $G4POC. The host lock stays
# gemma4nv's: one engine per host, whichever study launches it.
# bs3: the root disk is full, so every cache/tmp dir lives under /data.
export G4=/data/jooman/gemma4nv
export G4POC=/data/jooman/g4poc
case "$(hostname)" in
  build-server-2) export G4_HOME=/home/jooman/gemma4nv ;;
  *) export G4_HOME=$G4 ;;
esac
export G4_VENV=$G4_HOME/venv
export G4POC_HOST_LOCK=$G4/host.lock G4POC_GPU_LOCK=$G4/gpu.lock
# The study's FP8 checkpoint (text-only dir; see BASELINE-FP8.md section 2).
export G4POC_MODEL_DIR=${G4POC_MODEL_DIR:-$G4POC/models/gemma-4-26B-A4B-it-fp8ch/text}
# bs2's /data is nearly full: run records go to /home there.
case "$(hostname)" in
  build-server-2) export G4POC_RUNS_DIR=${G4POC_RUNS_DIR:-/home/jooman/g4poc/runs} ;;
esac
export TMPDIR=$G4_HOME/tmp UV_CACHE_DIR=$G4_HOME/cache/uv PIP_CACHE_DIR=$G4_HOME/cache/pip
export HF_HOME=$G4/hf XDG_CACHE_HOME=$G4_HOME/cache/xdg
export TRITON_CACHE_DIR=$G4_HOME/cache/triton TORCHINDUCTOR_CACHE_DIR=$G4_HOME/cache/inductor
export FLASHINFER_WORKSPACE_BASE=$G4_HOME/cache/flashinfer CUDA_CACHE_PATH=$G4_HOME/cache/nv
export TVM_FFI_CACHE_DIR=$G4_HOME/cache/tvm-ffi CUTE_DSL_CACHE_DIR=$G4_HOME/cache/cute-dsl
export SGLANG_CACHE_DIR=$G4_HOME/cache/sglang SGLANG_BUILD_RUST_EXTS=none
# Host-safety protocol: JIT builds never fan out to nproc+2 parallel nvcc jobs.
export MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME=/usr/local/cuda-13 PATH=/usr/local/cuda-13/bin:$HOME/.local/bin:$PATH
mkdir -p "$TMPDIR" "$G4POC"
if [ -f "$G4_VENV/bin/activate" ]; then source "$G4_VENV/bin/activate"; fi
