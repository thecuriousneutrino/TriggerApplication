#!/bin/bash

cd /root/HyperK/WCSim
git pull
make clean 
make rootcint 
make 

cd -

source ./GetToolDAQ.sh

./main configfiles/Dummy/ToolChainConfig


