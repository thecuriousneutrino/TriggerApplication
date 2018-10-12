### Created by Dr. Benjamin Richards (b.richards@qmul.ac.uk)

### Download base image from repo
FROM hkdaq/triggerapplication:base

### Run the following commands as super user (root):
USER root

Run cd TriggerApplication; make update; make;

### Open terminal
CMD ["/bin/bash"]
