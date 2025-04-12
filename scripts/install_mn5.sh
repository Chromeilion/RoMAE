#!/bin/bash
# Because installing packages is a pain in MareNostrum 5, this script makes it
# easier.
# It will load Python 3.11, create a virtual environment, and install all the
# required packages for RoMA in this virtualenv.

# All of these packages are stored in the following directory
package_root_folder=/gpfs/projects/ehpc10/rmae/uzivanov/software

# Some packages are stored as source files, while others are prebuilt wheels.
# I would store everything as source, but for some packages it's not possible,
# and for others the wheel is just way more convenient.

module load mkl/2024.0 intel impi hdf5/1.14.1-2-gcc python/3.11.5-gcc nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 gcc/11.4.0 nccl/2.19.4 pytorch/2.4.0
python -m venv --system-site-packages .venv
source .venv/bin/activate

pip install $package_root_folder/safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install $package_root_folder/huggingface_hub --no-index --no-build-isolation
pip install $package_root_folder/psutil-release --no-index --no-build-isolation
pip install $package_root_folder/accelerate --no-index --no-build-isolation
pip install $package_root_folder/flit_core-3.12.0-py3-none-any.whl --no-index --no-build-isolation
pip install $package_root_folder/click --no-index --no-build-isolation
#cd $package_root_folder/packaging
#pip install .
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
pip install $package_root_folder/filelock-3.18.0-py3-none-any.whl --no-index --no-build-isolation

# Next steps:
# You're gonna wanna actually install RoMA, probably with the following command
# from the project directory:
# pip install -e . --no-index --no-build-isolation

# You are also gonna need to install your experiment, if you have additional
# requirements for your experiment good luck with that!

echo "Done creating virtual environment in .venv!"
echo "To use the virtualenv, first load the correct modules:"
echo "module load mkl/2024.0 intel impi hdf5/1.14.1-2-gcc python/3.11.5-gcc nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 gcc/11.4.0 nccl/2.19.4 pytorch/2.4.0"
echo "Then activate it:"
echo "source .venv/bin/activate"
echo "Once this is done, you can install RoMA by moving to the directory with the package in it and running:"
echo "pip install -e . --no-index --no-build-isolation"
echo "Good luck with your research!"
