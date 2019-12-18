# TriggerApplication tools

The tools available are

## Data input

### Simulation/digit input
* WCSimReader
  * Read in simulation data from WCSim files
* WCSimASCIReader
  * Read in simulation data from ASCII files

### Reconstruction input
* ReconRandomiser
  * Produce randomised reconstruction distributions
* ReconDataIn
  * Read in reconstruction data from a `TTree`

## Data output

### Trigger output
* DataOut
  * Write out trigger (+ full simulation) data in WCSim file format
* TriggerOutput
  * Write out trigger data in text format

### Reconstruction output
* ReconDataOut
  * Write out reconstructed event data in a `TTree`

## Triggers

* nhits
  * Trigger on more than x-digits in a y ns sliding-window
* test_vertices
  * Trigger on more than x-digits in a y ns sliding-window, using time-of-flight subtracted times, on a fixed grid of vertices

## Reconstruction

* BONSAI
  * Reconstruct low-energy events with hk-BONSAI
* (test_vertices)
  * TODO: add option (or similar tool) to test_vertices

## SuperNova Filters

* ReconFilter
  * Filter out reconstructed events that don't pass cuts

## SuperNova Triggers

* dimfit
  * Find the number of dimensions that a positional vertex distribution corresponds to
  * Algorithm inherited from SK

## Miscellaneous

* CUDA
  * Some CUDA-GPU related code for on-GPU triggers
* template
  * Skeleton used by `newTool.sh`
* DummyTool
  * Dummy that prints out differnt messages to the console depending on the debug level
