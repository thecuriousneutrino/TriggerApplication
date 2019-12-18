# ReconDataOut

Write out a new file with a `TTree` storing reconstruction information

## Data

* Sets up a `TTree` called `reconTree` which is filled with the result of every reconstruction taken from the data member `RecoInfo` (or `RecoInfoMap`)
    	* `EventNum` 
      	* `TriggerNum`
	* `NDigits`
	* `Reconstructer` an enumeration of the tool that reconstructed (e.g. kReconBONSAI)
	  * TODO it is currently cast to an int. Need to setup a root linkdef to allow the enum to be used directly in the tree
	* `Time`
	* `Vertex[3]` x,y,z
	* `GoodnessOfFit`
	* `GoodnessOfTimeFit`
	* `HasDirection` boolean saying whether the reconstruction tool provides direction
	* `DirectionEuler[3]` theta (zenith), phi (azimuth), alpha
	* `CherenkovCone[2]` cos(Cherenkov angle), ellipticity
	* `DirectionLikelihood`

* For more information on the meaning of the output, see for example the [hk-BONSAI documentation](https://github.com/hyperk/hk-BONSAI)

## Configuration

```
input_filter_name NAME
outfilename /path/to/file
verbose LEVEL
```

* `input_filter_name` Name of the `ReconInfo` object to take from the map `RecoInfoMap`
  * If ALL takes the all reconstructed events from the global object `RecoInfo`
* `outfilename` File path to output file
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
