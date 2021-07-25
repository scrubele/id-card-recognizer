TESSERACT_VERSION='4.1.1' # Version to be installed

wget https://github.com/tesseract-ocr/tesseract/archive/${TESSERACT_VERSION}.zip
unzip ${TESSERACT_VERSION}.zip
cd tesseract-${TESSERACT_VERSION}
./autogen.sh
./configure
make -j4
sudo make install
sudo ldconfig
make training -j4
sudo make training-install

wget https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata
sudo mv -v eng.traineddata /usr/local/share/tessdata/