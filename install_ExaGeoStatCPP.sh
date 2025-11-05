cd /home/onyxia/work/

# Install libtool
sudo apt install libtool libtool-bin

# Clone the ExaGeoStatCPP repo
cd /home/onyxia/work/
git clone https://github.com/ecrc/ExaGeoStatCPP.git 

# Run in R
# install.packages(c("Rcpp", "assertthat"))

# Install hwloc
# sudo apt update
# sudo apt install hwloc libhwloc-dev

# Install ExaGeoStatCPP
cd ExaGeoStatCPP
R CMD INSTALL . --configure-args="-r"



