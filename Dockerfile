### Created by Dr. Benjamin Richards (b.richards@qmul.ac.uk)

### Download base image from repo
FROM hkdaq/triggerapplication:base

### Run the following commands as super user (root):
USER root

Run cd /root/HyperK/ ; source /root/HyperK/env-WCSim.sh ; cd WCSim ; git pull ;make clean; make rootcint; make; cd /TriggerApplication; source Setup.sh; make update; make; echo "source /root/HyperK/env-WCSim.sh" >>  /TriggerApplciaiton/Setup.sh 

### Open terminal
CMD ["/bin/bash"]
