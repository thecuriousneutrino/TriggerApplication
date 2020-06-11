# DataOut

DataOut takes WCSim pass through information and trigger results, writing out a new TFile with only the triggered digits, and some extra variables so that triggered events can be linked with WCSim events

## Data

DataOut
* Creates an output `TFile`
* Sets up 3 TTrees
  1. `wcsimGeoT` has 1 entry and is a direct clone of the input tree
  
     `wcsimrootgeom` of type `WCSimRootGeom` is the only branch
  2. `wcsimRootOptionsT` has N input file entries
  
     `wcsimrootoptions` of type `WCSimRootOptions` is a direct clone of the input tree
     
     `wcsimfilename` of type `TObjString` is the WCSim file this options class came from
  3. `wcsimT` has N events entries
    * `wcsimrootevent` of type `WCSimRootEvent` stores the ID events. This is identical to the WCSim event tree, with digits outside the trigger window removed*
      * This `WCSimRootEvent` with removed digits can be accessed by other tools using `IDWCSimEvent_Triggered` in the data model
    * `wcsimrootevent_OD` of type `WCSimRootEvent` stores the OD events. This is identical to the WCSim event tree, with digits outside the trigger window removed*
      * This `WCSimRootEvent` with removed digits can be accessed by other tools using `ODWCSimEvent_Triggered` in the data model
    * `wcsimfilename` of type `TObjArray` of `TObjString` stores the WCSim filename(s) of the current event
    * `wcsimeventnums` of type `vector<int>` stors the WCSim event number(s) of the current event
* *Any digit that is not in a trigger window is removed from the output `TClonesArray`
  * This is done using a combination of `IDTriggers` and `ODTriggers` that triggers have filled
    * e.g. if an OD digit is doesn't have any OD triggers, but is within an ID trigger window, the OD digit will be saved
    * This is different to WCSim which handles ID/OD triggers separately
  * The logic of which trigger to add a digit too is:
    * Time order the triggers
    * It is then the first trigger the digit is in
  * If there are no trigger windows, every digit will be removed
  * Digits that are triggered, but not in the 0th trigger window, are added to the relevant window before removal from the 0th trigger
  * Caveat: turning `save_only_failed_hits` on will save only digits that don't pass a trigger
* Truth tracks are also moved into their corresponding trigger window
  * However the logic is slightly different (since they are never dropped)
    * If they are in the 0th trigger readout window, store in 0th trigger
    * If they are after the 0th readout window and before the end of the 1st trigger window, store in the 1st trigger
    * etc
    * But note that WCSimRootTrigger object aren't created just for tracks, therefore if the track time is beyond the last trigger window, it is stored in the last trigger
  * Note that this is not exactly equivalent to WCSim. WCSim uses a fixed "`trigger_time` + 950 ns" for this check, rather than "`trigger_time` + `postrigger_save_window`"

## Configuration

```
outfilename /path/to/file
verbose LEVEL
save_multiple_hits_per_trigger [0,1]
trigger_offset OFFSET
save_only_failed_hits [0,1]
```

* `outfilename` File path to output file
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
* `save_multiple_hits_per_trigger` is a boolean flag. If false, will only allow one digit per PMT per trigger window to be written. If true, writes out as many as exist
* `trigger_offset` Offset applied to trigger time to account for `WCSimWCTriggerBase::offset` constant (set to 950 ns by default). This is related to SKI delay in the electronics/DAQ
* `save_only_failed_hits` is a boolean flag. If false, will do the normal triggering behaviour - save only digits that pass a trigger. If true, will do the opposite - save only digits that don't pass a trigger
