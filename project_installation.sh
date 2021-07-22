#!/bin/bash

sudo apt-get install git -y

cd ~/Documents
# git clone https://github.com/kemar74/mesure_activite_humaine.git

sudo apt-get install python3 -y
sudo apt-get install python3-pip -y
sudo python3 -m pip install virtualenv -y
cd mesure_activite_humaine
python3 -m virtualenv virtual_env
source virtual_env/bin/activate
cd ~/Documents

sudo apt-get install cmake -y
sudo apt-get install cmake-gui -y

sudo apt-get install libusb-1.0-0-dev -y
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev xorg-dev -y

git clone https://github.com/IntelRealSense/librealsense.git
mkdir librealsense/build && cd librealsense/build

echo '-> Click "Configure", select "Unix Makefile"'
echo '-> Check BUILD_PYTHON_BINDINGS'
echo '-> Click "Configure" again'
echo '-> Change PYTHON_EXECUTABLE to  /usr/bin/python3.6m'
echo '-> Change PYTHON_INSTALL_DIR to /usr/lib/python3/dist-packages/pyrealsense2'
echo '-> If there are "X11" errors, remove examples in the "BUILD" group'
echo '-> Click "Generate"'

sudo cmake-gui ..


sudo make -j6
sudo make install

export PYTHONPATH=$PYTHONPATH:/usr/lib

sudo cp wrappers/python/pyrealsense2* ~/Documents/mesure_activite_humaine/

echo '-----------------------'
echo 'Installation of Cubemos'
echo 'Download Cubemos from here:'
echo 'https://framos-my.sharepoint.com/:f:/p/c_scheubel/Em4eCGd-pctJnxkyG_pxRIgBcueMBoF5EuWtCYwGFEldsA?e=2XZZZB'
echo 'When done, press <ENTER>'

read -s -n 1 key

cd ~/Documents
unzip ~/Downloads/cubemos_SkeletonTracking.zip -d .
unzip -x cubemos_SkeletonTracking/cubemos_Skeleton_Tracking_v3.0.zip
sudo apt-get install ./cubemos_Skeleton_Tracking_v3.0/cubemos-SkeletonTracking_3.0.*.deb

exho '-------------------------'
echo 'Create a free-licence for Cubemos with a temporary email address:'
echo 'https://temp-mail.org/'
echo 'Access Cubemos website and "Try for free" with the temp address: https://www.intelrealsense.com/skeleton-tracking/'
echo 'Check the licence key in the temp email inbox'
echo 'When done, press <ENTER>'

read -s -n 1 key

bash /opt/cubemos/skeleton_tracking/scripts/post_installation.sh
source /etc/profile.d/cubemos.sh

pip install -r requirements.txt
pip install --find-links="$CUBEMOS_SKEL_SDK/wrappers/python" --no-index cubemos.skeletontracking

echo 'If there are still some errors, open this file (gedit project_installation.sh) and try the last few commands by hand'

