### Created by Dr. Benjamin Richards (b.richards@qmul.ac.uk)

### Download base image from repo
FROM hkdaq/triggerapplication:base

### Run the following commands as super user (root):
USER root

#Setup HK prerequisites & get latest version of WCSim
WORKDIR $WCSIMDIR
RUN source ../env-WCSim.sh; git pull; make clean; make rootcint; make;

#Get TriggerApplication
WORKDIR /
RUN git clone https://github.com/HKDAQ/TriggerApplication.git;
WORKDIR /TriggerApplication/
RUN ln -s /TriggerApplicationPreReq ToolDAQ

# Compile TriggerApp
ENV TrigGERAppinDOCKer indubitably
RUN source ./Setup.sh; make clean; make update; make;

### Open terminal
ENTRYPOINT source /TriggerApplication/Setup.sh && /bin/bash
