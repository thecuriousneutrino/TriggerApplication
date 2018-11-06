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
  * Flag to state whether the "data" is from Monte Carlo
    Stored as `bool`
    * `IsMC`
  * List of input WCSim files.
    Note that these are the files after the wildcard expansion.
    Stored in a `vector` of `string`
    * `WCSimFiles`
  * WCSim pass through information
    This isn't implemented yet
    

## Configuration

Describe any configuration variables for WCSimReader.

```
infile /path/to/file(s)
filelist /path/to/txt/file/list
nevents N
verbose LEVEL
```

* `infile` File path to WCSim root file. Can use wildcards to pickup multiple files
* `filelist` File path to a text file with a filepath to a WCSim root file on each line. Can also contain wildcards
* `nevents` will read only this number of events. If `N <= 0`, read all. If `N > number of available events`, read all.
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)