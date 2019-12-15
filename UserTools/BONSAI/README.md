# BONSAI

Call hk-BONSAI for every trigger, to reconstruct the event vertex position and direction.
hk-BONSAI is a low-energy reconstruction algorithm/software, therefore should only be run on low-energy events.
Write out a new file with a `TTree` storing this information.

## Data

* Sets up a `TTree` called `vertexInfo` which is filled every positive trigger. #MOVE TO NEW TOOL
  	For more information on the meaning of the output, see the [hk-BONSAI documentation](https://github.com/hyperk/hk-BONSAI)
    	* `EventNum` 
      	* `TriggerNum`
	* `NDigits`
	* `Vertex[4]` x,y,z,t
	* `DirectionEuler[3]` theta (zenith), phi (azimuth), alpha
	* `CherenkovCone[2]` cos(Cherenkov angle), ellipticity
	* `DirectionLikelihood`
	* `GoodnessOfFit`
	* `GoodnessOfTimeFit`
* For every trigger, gets the digit information from the data member `IDWCSimEvent_Triggered` (filled by the DataOut tool)
	* If the number of digits is in range [`nhitmin`, `nhitmax`] (inclusive), calls BONSAI with the digit information
	* Fills the `RecoInfo` in the data model with the BONSAI result

## Configuration

```
verbose LEVEL
nhitsmin UINT
nhitsmax UINT
```
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
* `nhitsmin` If the number of digits in a trigger is less than this, don't run BONSAI
* `nhitsmax` If the number of digits in a trigger is more than this, don't run BONSAI