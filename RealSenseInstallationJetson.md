git clone https://github.com/IntelRealSense/librealsense.git

mkdir librealsense/build
cd librealsense/build/
sudo cmake-gui ..

-> Click "Configure", select "Unix Makefile"
-> Check BUILD_PYTHON_BINDINGS
-> Click "Configure' again
-> Change PYTHON_EXECUTABLE to  /usr/bin/python3.6m
-> Change PYTHON_INSTALL_DIR to /usr/lib/python3/dist-packages/pyrealsense2
-> If there are "X11" errors, remove examples in the "BUILD" group
-> Click "Generate"

sudo make -j6
sudo make install

export PYTHONPATH=$PYTHONPATH:/usr/lib

cp wrappers/python/pyrealsense2* path/to/project/
