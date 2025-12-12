#!/bin/bash
# Because installing packages is a pain in MareNostrum 5, this script makes it
# easier.
# It will load Python 3.11, create a virtual environment, and install all the
# required packages for RoMAE in this virtualenv.

# All of these packages are stored in the following directories
package_root_folder=/gpfs/projects/ehpc10/rmae/uzivanov/software_m
PREFIX_DIR=/gpfs/projects/ehpc10/rmae/uzivanov/python_prefix

module load cuda/12.6 mkl/2024.0 intel nvidia-hpc-sdk/25.1 hdf5/1.14.4.2-nvidia-nvhpcx cudnn/9.6.0-cuda12 tensorrt/10.0.0-cuda12 gcc/13.2.0-nvidia-hpc-sdk

export PYTHON_BIN="$PREFIX_DIR/bin/python3"

$PYTHON_BIN -m venv --system-site-packages .venv
source .venv/bin/activate

pip install $package_root_folder/filelock-3.18.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/typing_extensions-4.13.2-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/setuptools-78.1.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/mpmath-1.3.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/sympy-1.13.1-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/networkx-3.4.2-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/jinja2-3.1.6-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/fsspec-2025.3.2-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/numpy-2.2.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install $package_root_folder/nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/triton-3.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/pillow-11.2.1-cp311-cp311-manylinux_2_28_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/torch-2.6.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/torchaudio-2.6.0+cu126-cp311-cp311-linux_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/torchvision-0.21.0+cu126-cp311-cp311-linux_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/wheel-0.45.1-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/packaging-24.2-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/PyYAML-6.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/charset_normalizer-3.4.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/idna-3.10-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/urllib3-2.4.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/certifi-2025.1.31-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/requests-2.32.3-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/tqdm-4.67.1-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/six-1.17.0-py2.py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/pluggy-1.5.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/protobuf-5.29.4-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/huggingface_hub --no-index --no-build-isolation
pip install $package_root_folder/psutil-release --no-index --no-build-isolation
pip install $package_root_folder/accelerate --no-index --no-build-isolation
pip install $package_root_folder/flit_core-3.12.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/click --no-index --no-build-isolation
pip install $package_root_folder/python-pathspec --no-index --no-build-isolation
pip install $package_root_folder/trove-classifiers --no-index --no-build-isolation
pip install $package_root_folder/hatch/backend --no-index --no-build-isolation
pip install $package_root_folder/docker_pycreds-0.4.0-py2.py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/smmap-5.0.2-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/gitdb-4.0.12-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/GitPython --no-index --no-build-isolation
pip install $package_root_folder/platformdirs-4.3.7-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/annotated_types-0.7.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/pydantic_core-2.33.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/typing_inspection-0.4.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/pydantic-2.11.3-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/sentry_sdk-2.25.1-py2.py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/setproctitle-1.3.5-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/wandb-0.19.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation
pip install $package_root_folder/python_dotenv-1.1.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/pydantic_settings-2.8.1-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/nvidia_ml_py-12.570.86-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/pynvml-12.0.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/h5py-3.13.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-build-isolation

# Next steps:
# You're gonna wanna actually install RoMAE, probably with the following command
# from the project directory:
# pip install -e . --no-index --no-build-isolation

# You are also gonna need to install your experiment, if you have additional
# requirements for your experiment good luck with that!

echo "Done creating virtual environment in .venv!"
echo "To use the virtualenv, first load the correct modules:"
echo "module load mkl/2024.0 intel impi hdf5/1.14.1-2-gcc python/3.11.5-gcc nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 gcc/11.4.0 nccl/2.19.4 pytorch/2.4.0"
echo "Then activate it:"
echo "source .venv/bin/activate"
echo "Once this is done, you can install RoMAE by moving to the directory with the package in it and running:"
echo "pip install -e . --no-index --no-build-isolation"
echo "Good luck with your research!"
