# Data Model
*************************

Data Model Class can be defined how ever the User requires. A Store is provided which ineficently maps variables to string lkeys via conversion to stringstream and can be used for debuging or other useful vairables.

A TTree map with getter and setter functions is provided and can be uncommented if required.

## Members

### Filled by `WCSimReader`, used by all tools

* `bool HasOD` - Does the geometry contain an OD?

### Filled by `WCSimReader`, used by trigger tools

Digit information
* `std::vector<SubSample> IDSamples` - Each member of this vector has a vector for each of digit tubeID, charge, and time. For the inner detector (ID).
* `std::vector<SubSample> ODSamples` - Same for outer detector (OD)

Geometry information
* `std::vector<PMTInfo> IDGeom` - Each member of this vector is an object with a information for a single PMT - tubeID, x, y, z. For the ID
* `std::vector<PMTInfo> ODGeom` - Same for the OD
* `int IDNPMTs` - Number of ID PMTs. Deprecated
* `int ODNPMTs` - Same for the OD. Deprecated
* `double IDPMTDarkRate` - The average dark noise rate for the ID PMTs
* `double ODPMTDarkRate` - Same for the OD

### Filled by `WCSimReader`, used by `DataOut`

Pass through information
* `TChain * WCSimGeomTree` - The `wcsimGeoT` chain
* `TChain * WCSimOptionsTree` - The `wcsimRootOptionsT` chain
* `TChain * WCSimEventTree` - The `wcsimT` chain. Deprecated
* `TObjArray * CurrentWCSimFiles` - The WCSim file(s) that the current event comes from
* `std::vector<int> CurrentWCSimEventNums` - The WCSim event number(s) that the current event comes from
* `WCSimRootEvent * WCSimEventID` - The current WCSim event. For the ID
* `WCSimRootEvent * WCSimEventOD` - Same for the OD
* `bool IsMC` - Is this Monte Carlo, rather than data?

### Filled by trigger tools, used by `DataOut`

Trigger results
* `TriggerInfo IDTriggers` - Containts vectors of trigger type, readout window start/end times, the time the trigger fired, and any additional information this trigger provides (in a `vector<float>`). For the ID
* `TriggerInfo ODTriggers` - Same for the OD
* `bool triggeroutput` - Has a trigger fired in this event? Deprecated
