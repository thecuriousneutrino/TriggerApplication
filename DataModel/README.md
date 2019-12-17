# Data Model
*************************

Data Model Class can be defined how ever the User requires. A Store is provided which ineficently maps variables to string lkeys via conversion to stringstream and can be used for debuging or other useful vairables.

A TTree map with getter and setter functions is provided and can be uncommented if required.

## TriggerApplication data members

The variables in this DataModel used by TriggerApplication tools are

Digit Information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `std::vector<SubSample>` | IDSamples           | Store charge, time, PMT ID of every ID digit | nhits | WCSimReader, WCSimASCIReader |
|  `std::vector<SubSample>` | ODSamples           | Store charge, time, PMT ID of every OD digit | nhits | WCSimReader |
|  `WCSimRootEvent *`       | IDWCSimEvent_Triggered      | The triggered `WCSimRootEvent` for the ID (digits are sorted into trigger windows) | BONSAI | DataOut |
|  `WCSimRootEvent *`       | ODWCSimEvent_Triggered      | The triggered `WCSimRootEvent` for the OD (digits are sorted into trigger windows) | | DataOut |

 Geometry Information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `std::vector<PMTInfo>`   | IDGeom              | Store x, y, z, PMT ID of every ID PMT        | | WCSimReader |
|  `std::vector<PMTInfo>`   | ODGeom              | Store x, y, z, PMT ID of every OD PMT        | | WCSimReader |
|  `double`                 | IDPMTDarkRate       | The dark rate for ID PMTs | nhits | WCSimReader |
|  `double`                 | ODPMTDarkRate       | The dark rate for OD PMTs | nhits | WCSimReader |
|  `int`                    | IDNPMTs             | The number of ID PMTs     | nhits | WCSimReader |
|  `int`                    | ODNPMTs             | The number of OD PMTs     | nhits | WCSimReader |
|  `bool`                   | HasOD             | Does the geometry include the OD? | DataOut | WCSimReader |

Trigger Information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `TriggerInfo`            | IDTriggers          | Store trigger type, time, readout window start/end times, additional info vector, for ID triggers | WCSimReader | nhits |
|  `TriggerInfo`            | ODTriggers          | Store trigger type, time, readout window start/end times, additional info vector, for OD triggers | WCSimReader | nhits |
|  `bool`                   | triggeroutput       | Did a trigger fire?       | TriggerOutput | nhits, test_vertices |

Reconstruction information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `ReconInfo`              | RecoInfo            | Store reconstruction information (vertex time/position, fit likelihoods, optionally direction) | dimfit, ReconDataOut | BONSAI, ReconRandomiser, ReconDataIn |

Pass-through information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `TChain *`               | WCSimGeomTree       | The `WCSimRootGeom` tree from input WCSim file(s)    | DataOut, BONSAI | WCSimReader |
|  `TChain *`               | WCSimOptionsTree    | The `WCSimRootOptions` tree from input WCSim file(s) | DataOut | WCSimReader |
|  `TChain *`               | WCSimEventTree      | The `WCSimRootEvent` tree from input WCSim file(s)   | | WCSimReader |
|  `std::vector<int>`       | CurrentWCSimEventNums | The original WCSim files' event number(s) for the current event | DataOut | WCSimReader |
|  `TObjArray *`            | CurrentWCSimFiles     | The original WCSim files' filename(s) for the current event     | DataOut | WCSimReader |
|  `WCSimRootEvent *`       | IDWCSimEvent_Raw      | The original, unmodified `WCSimRootEvent` for the ID | DataOut | WCSimReader |
|  `WCSimRootEvent *`       | ODWCSimEvent_Raw      | The original, unmodified `WCSimRootEvent` for the OD | DataOut | WCSimReader |
| Misc |
|  `bool`                   | IsMC              | Is the input MC? | | WCSimReader |
 
TODO: setup to allow multiple types of PMT in the ID (e.g. 20" + mPMT hybrid geometry)

TODO: fill 'Reset By' column
