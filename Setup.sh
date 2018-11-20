#!/bin/bash

#Application path location of applicaiton

ToolDAQapp=`pwd`

source /root/HyperK/root/bin/thisroot.sh

export LD_LIBRARY_PATH=${ToolDAQapp}/lib:${ToolDAQapp}/ToolDAQ/zeromq-4.0.7/lib:${ToolDAQapp}/ToolDAQ/boost_1_66_0/install/lib:$WCSIMDIR:$LD_LIBRARY_PATH
