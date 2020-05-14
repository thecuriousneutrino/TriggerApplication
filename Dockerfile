### Created by Dr. Benjamin Richards (b.richards@qmul.ac.uk)

### Download base image from repo
FROM hkdaq/triggerapplication:base

### Run the following commands as super user (root):
USER root

#Setup HK prerequisites & get latest version of WCSim
WORKDIR $WCSIMDIR
RUN source ../env-WCSim.sh; git pull; make clean; make rootcint; make;
ENV ROOT_INCLUDE_PATH $WCSIMDIR/include

#Get TriggerApplication
WORKDIR $HYPERKDIR
RUN git clone https://github.com/HKDAQ/TriggerApplication.git;
ENV TRIGGERAPPDIR $HYPERKDIR/TriggerApplication
WORKDIR $TRIGGERAPPDIR
RUN ln -s $HYPERKDIR/ToolDAQ/ ToolDAQ

# Compile TriggerApp
ENV TrigGERAppinDOCKer indubitably
RUN source ./Setup.sh; make clean; make update; make;

### Open terminal
ENTRYPOINT source $TRIGGERAPPDIR/Setup.sh && /bin/bash
