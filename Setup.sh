#!/bin/bash

#Application path location of applicaiton
ToolDAQapp=$(readlink -f $(dirname $BASH_SOURCE))

#Setup for when running in docker
if [ "$TrigGERAppinDOCKer" = "indubitably" ]; then
    echo "You're running in docker. Setting up ROOT/WCSim/Geant4"
    echo "(If you're not, why have you set \$TrigGERAppinDOCKer to \"indubitably\"?!)"
    #Setup WCSIM
    source $WCSIMDIR/../env-WCSim.sh
    #Setup BONSAI, if it's there
    if [ -d $HYPERKDIR/hk-BONSAI ]; then
        export BONSAIDIR=$HYPERKDIR/hk-BONSAI
	export PATH=$BONSAIDIR/rootbonsai:$PATH
    fi
    #Setup FLOWER, if it's there
    if [ -d $HYPERKDIR/flower ]; then
        export FLOWERDIR=$HYPERKDIR/FLOWER
	export PATH=$FLOWERDIR/rootflower:$PATH
    fi
fi

#Check WCSim is setup
if [ -z "$WCSIMDIR" ]; then
    echo "Setup WCSim (i.e. set \$WCSIMDIR) before setting up TriggerApplication";
    echo "Also make sure ROOT is setup (requirement of setting up WCSim)";
    echo "And then run this script again!"
    return;
fi

#Add the standard things to the library path
export LD_LIBRARY_PATH=${ToolDAQapp}/lib:${ToolDAQapp}/ToolDAQ/zeromq-4.0.7/lib:${ToolDAQapp}/ToolDAQ/boost_1_66_0/install/lib:$WCSIMDIR:$LD_LIBRARY_PATH

#File to turn on/off optional packages in the TriggerApplication build
echo "" > $ToolDAQapp/Build.h

#Check if we've got BONSAI
if [ -z "$BONSAIDIR" ]; then
    echo "Running without BONSAI";
else
    echo "#define BONSAIEXISTS" >> $ToolDAQapp/Build.h
    export LD_LIBRARY_PATH=$BONSAIDIR:$LD_LIBRARY_PATH
fi

#Check if we've got FLOWER
if [ -z "$FLOWERDIR" ]; then
    echo "Running without FLOWER";
else
    echo "#define FLOWEREXISTS" >> Build.h
    export LD_LIBRARY_PATH=$FLOWERDIR:$LD_LIBRARY_PATH
fi
