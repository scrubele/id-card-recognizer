git clone https://github.com/duckie/difflib
cd difflib
mkdir build
cd build
cmake ../
make
ldconfig
./test/difflib-test