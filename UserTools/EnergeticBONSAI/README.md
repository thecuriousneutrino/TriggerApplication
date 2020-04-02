# EnergeticBONSAI

EnergeticBONSAI will take a reconstructed vertex (and hits from the associated trigger) from the DataModel and run the energetic-BONSAI energy reconstruction algorithm

## Data

* Performs setup of energetic-BONSAI reading in from the DataModel
  * Geometry from `WCSimGeomTree`
  * PMT dark rate from `IDPMTDarkRate`
  * Number of PMTs from `IDNPMTs`
* For each reconstructed vertex that pass a filter (see `ReconFilter`)
  * Read in the vertex from the `ReconInfo`
  * Get the hits in the trigger from `IDWCSimEvent_Triggered`
  * Run the energetic-BONSAI external program to get the reconstructed energy


## Configuration

### Parameters that specify which reconstructed events to use
```
input_filter_name NAME
nhitsmin NUM
nhitsmax NUM
```

* `input_filter_name` Name of the `ReconInfo` object to take from the map `RecoInfoMap`
  * If ALL takes the all reconstructed events from the global object `RecoInfo`
* `nhitsmin` When the number of hits in the trigger is below this number, energetic-BONSAI will not be called
* `nhitsmax` When the number of hits in the trigger is above this number, energetic-BONSAI will not be called

### energetic-BONSAI running parameters
```
n_working_pmts NUM
detector_name  NAME
overwrite_nearest_neighbours BOOL
ebonsai_verbose NUM
```
* `n_working_pmts` Use this to mask PMTs. If `n_working_pmts` more than `number of PMTs in simulated geometry`, this is set to `number of PMTs in simulated geometry`. Default is `number of PMTs in simulated geometry`
* `detector_name` Used by energetic-BONSAI to set default values of e.g. nearest neighbour distance
* `overwrite_nearest_neighbours` `false`: read in the nearest neighbour file from `$EBONSAIDIR/data/`. `true`: overwrite this file with a recalculation. Takes ~5 minutes to calculate for ~40k PMTs
* `ebonsai_verbose` Verbosity level for energetic-BONSAI internally. Runs from 0 (low verbosity) to 9 (high verbosity)

### Misc
```
verbose LEVEL
```
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)

