# ReconFilter

Filter out reconstructed events that don't pass cuts

## Data

* Read in a collection of reconstruction results from a `ReconInfo` object
  * From `RecoInfo`, or from a map
* For every event that passes criteria (fiducial volume, reconstruction tool, etc.) 
  * Write out into new `RecoInfo` object

## Configuration

### The parameters saying what to filter

```
reconstruction_algorithm ALGSTRING
min_recon_likelihood LIKELIHOOD
min_recon_time_likelihood LIKELIHOOD
max_r_pos DISTANCE
max_z_pos DISTANCE
```

* `reconstruction_algorithm` Use the results of which reconstruction algorithm?
* `min_recon_likelihood` Events with a likelihood smaller than this will not be included in dimfit calculations
* `min_recon_time_likelihood` Events with a time-fit likelihood smaller than this will not be included in dimfit calculations
* `max_r_pos` Events with larger reconstructed `r` (in cm) than this will not be included in dimfit calculations
* `max_z_pos` Events with larger reconstructed `z` (in cm) than this will not be included in dimfit calculations
  * Note that this is a detector half-height because the co-ordinate origin is at the detector centre

### Input/output

```
input_filter_name NAME
output_filter_name NAME
```

* `input_filter_name` Name of the `ReconInfo` object to take from the map `RecoInfoMap`
  * If ALL takes the all reconstructed events from the global object `RecoInfo`
* `output_filter_name` Name of the `ReconInfo` object to save in the map `RecoInfoMap`
  * Cannot be ALL
  * Cannot be the same as `input_filter_name`
* `outfilename` File path to output file

### Misc

```
verbose LEVEL
```

* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)