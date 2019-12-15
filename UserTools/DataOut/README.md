# DataOut

DataOut takes WCSim pass through information and trigger results, writing out a new TFile with only the triggered digits, and some extra variables so that triggered events can be linked with WCSim events

## Data

DataOut
* Creates an output `TFile`
* Sets up 3 TTrees
  1. `wcsimGeoT` has 1 entry and is a direct clone of the input tree
    * `wcsimrootgeom` of type `WCSimRootGeom` is the only branch
  2. `wcsimRootOptionsT` has N input file entries
    * `wcsimrootoptions` of type `WCSimRootOptions` is a direct clone of the input tree
    * `wcsimfilename` of type `TObjString` is the WCSim file this options class came from
  3. `wcsimT` has N events entries
    * `wcsimrootevent` of type `WCSimRootEvent` stores the ID events. This is identical to the WCSim event tree, with digits outside the trigger window removed*
      * This `WCSimRootEvent` with removed digits can be accessed by other tools using `IDWCSimEvent_Triggered` in the data model
    * `wcsimrootevent_OD` of type `WCSimRootEvent` stores the OD events. This is identical to the WCSim event tree, with digits outside the trigger window removed*
      * This `WCSimRootEvent` with removed digits can be accessed by other tools using `ODWCSimEvent_Triggered` in the data model
    * `wcsimfilename` of type `TObjArray` of `TObjString` stores the WCSim filename(s) of the current event
    * `wcsimeventnums` of type `vector<int>` stors the WCSim event number(s) of the current event
* *Any digit that is not in the trigger window is removed from the output `TClonesArray`
  * Currently a fixed cutoff of 1000 ns is used

## Configuration

```
outfilename /path/to/file
verbose LEVEL
```

* `outfilename` File path to output file
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)