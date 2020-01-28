# BONSAI

Call hk-BONSAI for every trigger, to reconstruct the event vertex position and direction.
hk-BONSAI is a low-energy reconstruction algorithm/software, therefore should only be run on low-energy events.

## Data

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