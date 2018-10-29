# WCSimReader

WCSimReader reads in WCSim root files

## Data

WCSimReader
* reads in WCSim root files
* adds to the transient data model
  * PMT geometry (ID, position, angle)
  * Digitised hit information (charge, time, ID)
  * WCSim pass through information


## Configuration

Describe any configuration variables for WCSimReader.

```
infile /path/to/file(s)
filelist /path/to/txt/file/list
nevents N
```

* `infile` can use wildcards to pickup multiple files
* `filelist` can also contain wildcards
* `nevents` will read only this number of events. If `N <= 0`, read all. If `N > number of available events`, read all.