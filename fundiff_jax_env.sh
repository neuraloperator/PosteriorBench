# Source this file before running FunDiff JAX jobs from this workspace.
_fundiff_env_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export POSTERIORBENCH_ROOT="${POSTERIORBENCH_ROOT:-$_fundiff_env_root}"
export POSTERIORBENCH_FUNDIFF_VENV="${POSTERIORBENCH_FUNDIFF_VENV:-$POSTERIORBENCH_ROOT/.venv-fundiff-jax}"
export PYTHONPATH="$POSTERIORBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$POSTERIORBENCH_FUNDIFF_VENV/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

_fundiff_jax_nvidia=""
for _fundiff_jax_candidate in "$POSTERIORBENCH_FUNDIFF_VENV"/lib/python*/site-packages/nvidia; do
  if [[ -d "$_fundiff_jax_candidate" ]]; then
    _fundiff_jax_nvidia="$_fundiff_jax_candidate"
    break
  fi
done
if [[ -n "$_fundiff_jax_nvidia" ]]; then
  export LD_LIBRARY_PATH="$_fundiff_jax_nvidia/cublas/lib:$_fundiff_jax_nvidia/cuda_cupti/lib:$_fundiff_jax_nvidia/cuda_nvrtc/lib:$_fundiff_jax_nvidia/cuda_runtime/lib:$_fundiff_jax_nvidia/cudnn/lib:$_fundiff_jax_nvidia/cufft/lib:$_fundiff_jax_nvidia/cusolver/lib:$_fundiff_jax_nvidia/cusparse/lib:$_fundiff_jax_nvidia/nccl/lib:$_fundiff_jax_nvidia/nvjitlink/lib:$_fundiff_jax_nvidia/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _fundiff_env_root
unset _fundiff_jax_nvidia
unset _fundiff_jax_candidate
