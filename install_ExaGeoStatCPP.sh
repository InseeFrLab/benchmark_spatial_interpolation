cd /home/onyxia/work/

# Install libtool
wget http://ftpmirror.gnu.org/libtool/libtool-2.4.7.tar.gz
tar -xvzf libtool-2.4.7.tar.gz
cd libtool-2.4.7
./configure --prefix=$HOME/local
make
make install

export PATH=$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH

export LIBTOOL=$HOME/local/bin/libtool
export LIBTOOLIZE=$HOME/local/bin/libtoolize

export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH

# Clone the ExaGeoStatCPP repo
cd /home/onyxia/work/
git clone https://github.com/ecrc/ExaGeoStatCPP.git 

# Run in R
# install.packages(c("Rcpp", "assertthat"))

# Install ExaGeoStatCPP
cd ExaGeoStatCPP
R CMD INSTALL . --configure-args="-r"



