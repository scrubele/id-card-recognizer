# http://chokkan.org/software/simstring/api/

wget chokkan.org/software/dist/simstring-1.0.tar.gz
tar xzf simstring-1.0.tar.gz
cd simstring-1.0
./configure
sudo awk '/<iostream>/ { print; print "#include <unistd.h>"; next }1' include/simstring/memory_mapped_file.h >test.tmp &&
  mv -f test.tmp include/simstring/memory_mapped_file.h
make -j4
make install
