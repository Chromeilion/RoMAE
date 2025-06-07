module load cuda/12.6 mkl/2024.0 intel nvidia-hpc-sdk/25.1 hdf5/1.14.4.2-nvidia-nvhpcx cudnn/9.6.0-cuda12 tensorrt/10.0.0-cuda12 gcc/13.2.0-nvidia-hpc-sdk

PYTHON_LOC=/gpfs/projects/ehpc10/rmae/uzivanov/Python-3.11.11.tgz
PYTHON_VERSION=3.11.11
PREFIX_DIR=/gpfs/projects/ehpc10/rmae/uzivanov/python_prefix

mkdir $PREFIX_DIR

tar -xzf $PYTHON_LOC
cd Python-"$PYTHON_VERSION"/ || exit
make clean
./configure --enable-optimizations CC="gcc -pthread" CXX="g++ -pthread" "--prefix=$PREFIX_DIR"
make -j "$(nproc --all)"
make install
cd ..

# Remove leftover files
rm -r ./Python-"$PYTHON_VERSION"/