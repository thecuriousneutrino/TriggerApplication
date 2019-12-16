# TriggerApplication

The tools available are

## Data input

* WCSimReader
  * Read in data from WCSim files
* WCSimASCIReader
  * Read in data from ASCII files
* ReconRandomiser
  * Produce randomised vertex distributions

## Data output

* DataOut
  * Write out data in WCSim file format
* TriggerOutput
  * Write out data in text format

## Triggers

* nhits
  * Trigger on more than x-digits in a y ns sliding-window
* test_vertices
  * Trigger on more than x-digits in a y ns sliding-window, using time-of-flight subtracted times, on a fixed grid of vertices

## Reconstruction

* BONSAI
  * Reconstruct low-energy events with hk-BONSAI
* (test_vertices)

## SuperNova Triggers

* dimfit
  * Find the number of dimensions a positional vertex distribution corresponds to
  * Algorithm inherited from SK

## Miscellaneous

* CUDA
  * Some CUDA-GPU related code for on-GPU triggers
* template
  * Skeleton used by `newTool.sh`
* DummyTool
  * Dummy that prints out differnt messages to the console depending on the debug level
