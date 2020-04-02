# TriggerApplication tools

This is a list of tools with a brief description

For more, see the README of each tool

Table of Contents
=================

   * [TriggerApplication tools](#triggerapplication-tools)
      * [Data input](#data-input)
         * [Simulation/digit input](#simulationdigit-input)
         * [Reconstruction input](#reconstruction-input)
      * [Data output](#data-output)
         * [Trigger output](#trigger-output)
         * [Reconstruction output](#reconstruction-output)
      * [Data resets](#data-resets)
         * [Reconstruction reset](#reconstruction-reset)
      * [Triggers](#triggers)
      * [Reconstruction](#reconstruction)
      * [SuperNova Filters](#supernova-filters)
      * [SuperNova Triggers](#supernova-triggers)
      * [Miscellaneous](#miscellaneous)

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

## Data resets

### Reconstruction reset
* ReconReset
  * `Reset()` all instances of `ReconInfo` in the data model

## Triggers

* nhits
  * Trigger on more than x-digits in a y ns sliding-window
* test_vertices
  * Trigger on more than x-digits in a y ns sliding-window, using time-of-flight subtracted times, on a fixed grid of vertices

## Reconstruction

* BONSAI
  * Reconstruct low-energy events with hk-BONSAI
  * Requires a trigger
  * Returns reconstructed vertex position and direction
* EnergeticBONSAI
  * Reconstruct low-energy events with energetic-BONSAI
  * Requires a reconstructed vertex
  * Returns reconstructed energy
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
