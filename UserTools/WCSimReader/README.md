# WCSimReader

WCSimReader reads in WCSim root files

## Data

WCSimReader
* reads in WCSim root files
* adds to the transient data model
  * PMT geometry (ID, position, angle)
    Stored in a `vector` of `PMTInfo`
    * `IDGeom` ID PMTs
    * `ODGeom` OD PMTs
  * Digitised hit information (charge, time, ID)
    Stored in a `vector` of `SubSample`
    * `IDSamples` for ID digits
    * `ODSamples` for OD digits
  * Number of PMTs
    Stored as an `int`
    * `IDNPMTs` for ID number of PMTs
    * `ODNPMTs` for OD number of PMTs
  * PMT dark rate
    Stored as `double`
    * `IDPMTDarkRate` for ID PMT dark rate
    * `ODPMTDarkRate` for OD PMT dark rate
  * Flag to state whether the OD exists.
    Stored as a `bool`
    * `HasOD`
  * Flag to state whether the "data" is from Monte Carlo
    Stored as `bool`
    * `IsMC`
  * Linking of new events to original WCSim files/event numbers.
    * `CurrentWCSimFiles` is a `TObjArray` of `TObjString`. Note that these are the files after the wildcard expansion.
    * `CurrentWCSimEventNums` us a `vector<int>`
  * WCSim pass through information
    * `TChain *` are stored for each of the WCSim trees
      * `WCSimGeomTree` for the geometry
      * `WCSimOptionsTree` for the options
      * `WCSimEventTree` for the events
    * Additionally, the current `WCSimRootEvent *` are stored
      *  `IDWCSimEvent_Raw` for the ID events
      *  `ODWCSimEvent_Raw` for the OD events
    

## Configuration

```
infile /path/to/file(s)
filelist /path/to/txt/file/list
nevents N
first_event N
verbose LEVEL
```

* `infile` File path to WCSim root file. Can use wildcards to pickup multiple files
* `filelist` File path to a text file with a filepath to a WCSim root file on each line. Can also contain wildcards
* `nevents` will read only this number of events. If `N <= 0`, read all. If `N > number of available events`, read all.
* `first_event` will skip to this event number. If `N > number of available events`, will read the final event only. If `N < 0`, will set it to 0. Default is 0
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)