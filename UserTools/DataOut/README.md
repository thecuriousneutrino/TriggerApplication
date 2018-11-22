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
    * `wcsimrootevent_OD` of type `WCSimRootEvent` stores the OD events. This is identical to the WCSim event tree, with digits outside the trigger window removed*
    * `wcsimfilename` of type `TObjArray` of `TObjString` stores the WCSim filename(s) of the current event
    * `wcsimeventnums` of type `vector<int>` stors the WCSim event number(s) of the current event
* *Any digit that is not in the trigger window is removed from the output `TClonesArray`
  * Currently a fixed cutoff of 1000 ns is used

## Configuration

```
outfilename /path/to/file
verbose LEVEL
save_multiple_digits_per_trigger [0,1]
trigger_offset OFFSET
```

* `outfilename` File path to output file
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
* `save_multiple_digits_per_trigger` is a boolean flag. If false, will only allow one digit per PMT per trigger window to be written. If true, writes out as many as exist
* `trigger_offset` Offset applied to trigger time to account for `WCSimWCTriggerBase::offset` constant (set to 950 ns by default). This is related to SKI delay in the electronics/DAQ
