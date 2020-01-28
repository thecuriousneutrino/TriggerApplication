#!/bin/bash

#Application path location of applicaiton

ToolDAQapp=`pwd`

source /root/HyperK/root/bin/thisroot.sh

if [ -z "$WCSIMDIR" ]; then
    echo "Setup WCSim (i.e. set \$WCSIMDIR) before setting up TriggerApplication";
    echo "Also make sure ROOT is setup (requirement of setting up WCSim)";
    echo "Also make sure BONSAI is setup (i.e. set \$BONSAIDIR)";
    echo "And then run this script again!"
    return;
fi

export LD_LIBRARY_PATH=${ToolDAQapp}/lib:${ToolDAQapp}/ToolDAQ/zeromq-4.0.7/lib:${ToolDAQapp}/ToolDAQ/boost_1_66_0/install/lib:$WCSIMDIR:$BONSAIDIR:$LD_LIBRARY_PATH
