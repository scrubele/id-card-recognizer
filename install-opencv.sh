OPENCV_VERSION='4.5.2' # Version to be installed

# 1. KEEP UBUNTU OR DEBIAN UP TO DATE

sudo apt-get -y update
sudo apt-get -y upgrade
# sudo apt-get -y dist-upgrade  # Uncomment to handle changing dependencies with new vers. of pack.
# sudo apt-get -y autoremove    # Uncomment to remove packages that are now no longer needed

# 2. INSTALL THE DEPENDENCIES

# Build tools:
sudo apt-get install -y build-essential cmake

# GUI (if you want GTK, change 'qt5-default' to 'libgtkglext1-dev' and remove '-DWITH_QT=ON'):
sudo apt-get install -y qt5-default libvtk6-dev

# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev \
  libopenexr-dev libgdal-dev

# Video I/O:
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev \
  libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
  libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev

# Parallelism and linear algebra libraries:
sudo apt-get install -y libtbb-dev libeigen3-dev

# Python:
sudo apt-get install -y python-dev python-tk pylint python-numpy \
  python3-dev python3-tk pylint3 python3-numpy flake8

# Java:
sudo apt-get install -y ant default-jdk

# Documentation and other:
sudo apt-get install -y doxygen unzip wget

# 3. DOWNLOAD THE LIBRARY

wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
mv opencv-${OPENCV_VERSION} opencv

wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

cd OpenCV &
mkdir build &
cd build

# 4. INSTALL THE LIBRARY

cmake -DFORCE_GTK=ON -DBUILD_PROTOBUF=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..

make -j4
make install
ldconfig
