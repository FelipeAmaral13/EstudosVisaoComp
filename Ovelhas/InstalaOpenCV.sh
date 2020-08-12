#!/bin/bash

echo ""
echo "-----------------------------------------------"
echo "  Atualizacao do Sistema (update e upgrade )   "
echo "-----------------------------------------------"
echo ""

sudo apt-get update
sudo apt-get upgrade -y

echo ""
echo "-----------------------------------------------"
echo "Instalacao de pacotes necessarios para o OpenCV"
echo "-----------------------------------------------"
echo ""

sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk2.0-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y python2.7-dev python3-dev


echo ""
echo "-----------------------------------------------"
echo "Remocao de pacotes Desnecessarios para o OpenCV"
echo "-----------------------------------------------"
echo ""

sudo apt autoremove

echo ""
echo "----------------------------------------"
echo "Download do codigo-fonte do OpenCV 3.1.0"
echo "----------------------------------------"
echo ""

cd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip opencv.zip

wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip

echo ""
echo "------------------------------------------------------------------------"
echo "Instalacao do numpy "
echo "(pacote do Python para operacoes com arrays e matrizes multidimensionais"
echo "------------------------------------------------------------------------"
echo ""

pip install numpy

echo ""
echo "--------------------------"
echo "Compilacao do OpenCV 3.1.0"
echo "--------------------------"
echo ""

cd ~/opencv-3.1.0/
mkdir build
cd build
cmake -G 'Unix Makefiles' \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -DCMAKE_BUILD_TYPE=Release  \
  -DBUILD_EXAMPLES=ON \
  -DINSTALL_C_EXAMPLES=ON \
  -DINSTALL_PYTHON_EXAMPLES=ON  \
  -DBUILD_NEW_PYTHON_SUPPORT=ON \
  -DWITH_FFMPEG=ON  \
  -DWITH_GSTREAMER=OFF  \
  -DWITH_GTK=ON \
  -DWITH_JASPER=ON  \
  -DWITH_JPEG=ON  \
  -DWITH_PNG=ON \
  -DWITH_TIFF=ON  \
  -DWITH_OPENEXR=ON \
  -DWITH_PVAPI=ON \
  -DWITH_UNICAP=OFF \
  -DWITH_EIGEN=ON \
  -DWITH_XINE=OFF \
  -DBUILD_TESTS=OFF \
  -DCMAKE_SKIP_RPATH=ON \
  -DWITH_CUDA=OFF \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DENABLE_SSE=ON -DENABLE_SSE2=ON -DENABLE_SSE3=OFF \
  -DWITH_OPENGL=ON -DWITH_TBB=ON -DWITH_1394=ON -DWITH_V4L=ON


make

echo ""
echo "--------------------------"
echo "Instalacao do OpenCV 3.1.0"
echo "--------------------------"
echo ""

sudo make install

echo ""
echo "--------------------------------------------"
echo "cria links e cache para bibliotecas"
echo "recentemente adicionadas (no caso, o OpenCV)"
echo "--------------------------------------------"
echo ""

sudo ldconfig
