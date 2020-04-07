#!/bin/bash

#Application path location of applicaiton

ToolDAQapp=$(readlink -f $(dirname $BASH_SOURCE))

if [ "$TrigGERAppinDOCKer" = "indubitably" ]; then
    echo "You're running in docker. Setting up ROOT/WCSim/Geant4"
    echo "(If you're not, why have you set \$TrigGERAppinDOCKer to \"indubitably\"?!)"
    source $WCSIMDIR/../env-WCSim.sh

fi

echo "" > $ToolDAQapp/Build.h

if [ -z "$WCSIMDIR" ]; then
    echo "Setup WCSim (i.e. set \$WCSIMDIR) before setting up TriggerApplication";
    echo "Also make sure ROOT is setup (requirement of setting up WCSim)";
    echo "And then run this script again!"
    return;
fi

export LD_LIBRARY_PATH=${ToolDAQapp}/lib:${ToolDAQapp}/ToolDAQ/zeromq-4.0.7/lib:${ToolDAQapp}/ToolDAQ/boost_1_66_0/install/lib:$WCSIMDIR:$LD_LIBRARY_PATH

if [ -z "$BONSAIDIR" ]; then
    echo "Running without BONSAI";
else
    echo "#define BONSAIEXISTS" >> $ToolDAQapp/Build.h
    export LD_LIBRARY_PATH=$BONSAIDIR:$LD_LIBRARY_PATH
fi

if [ -z "$EBONSAIDIR" ]; then
    echo "Running without energetic BONSAI";
else
    echo "#define EBONSAIEXISTS" >> Build.h
    export LD_LIBRARY_PATH=$EBONSAIDIR:$LD_LIBRARY_PATH
fi


