git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose/
git submodule update --init --recursive --remote
mkdir build/
cd build/
sudo cmake-gui ..

-> Click "Configure" and select "Unix Makefile"
-> Check "BUILD_PYTHON"
-> (optionnal) check "DOWNLOAD_BODY_COCO_MODEL" and "DOWNLOAD_BODY_MPI_MODEL"
-> Click "Generate"

sudo make -j6

sudo gedit /etc/environment
-> add this line : LD_LIBRARY_PATH="/usr/lib/python3/site-packages"

sudo cp python/openpose/pyopenpose.cpython*.so /usr/lib/python3/dist-packages/
sudo ln -s /usr/lib/python3/dist-packages/pyopenpose.cpython*.so /usr/lib/python3/dist-packages/pyopenpose

