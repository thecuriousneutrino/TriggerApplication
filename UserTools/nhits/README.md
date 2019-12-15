# nhits

Trigger on more than x-digits in a y ns sliding-window

## Data

* Picks the relevant digits information from the data model (IDSample or ODSamples)
* Calls the GPU or CPU algorithm
  * CPU version
    * Loop over all digits to find first/last times
    * Loop from first hit time to last hit time, with a sliding trigger decision window (size `trigger_search_window`, step `trigger_search_window_step`)
        * For every digit in the trigger decision window, fill a vector with the digit time
    	* If the number of digits is more than the threshold, issue a trigger
     	  * The trigger time is the `trigger_threshold`'ed entry in the (sorted) digit time vector (the time of the first digit above threshold)
	    * i.e. if the threshold is 25, the trigger time is taken as the time of the 26th digit
	  * Store the trigger in the data model (IDTriggers or ODTriggers)
        * Increment the loop by `posttrigger_save_window` (rather than `trigger_search_window_step`)
  * GPU version
    * TODO ensure the algorithm is similar (may not be identical due to GPU-related optimisations that can be made), the definitions of trigger time etc are identical, and the results are identical
    * TODO make the GPU version of the code work with the WCSimReader tool
* TODO make both versions work with the WCSimASCIReader tool

## Configuration

```
trigger_search_window INT
trigger_search_window_step INT
trigger_threshold INT
trigger_threshold_adjust_for_noise BOOL
pretrigger_save_window INT
posttrigger_save_window INT
trigger_od BOOL
verbose INT
```
* `trigger_search_window` Width of the sliding window, in ns
* `trigger_search_window_step` Step size of the sliding window, in ns
* `trigger_threshold` Trigger threshold - number of triggers must be above this value (equal to does not fire the trigger)
* `trigger_threshold_adjust_for_noise` Add the average dark noise rate to the threshold?
* `pretrigger_save_window` After a positive trigger is found, save digits from `trigger_time - pretrigger_save_window` to `trigger_time + posttrigger_save_window`
* `posttrigger_save_window` After a positive trigger is found, save digits from `trigger_time - pretrigger_save_window` to `trigger_time + posttrigger_save_window`
* `trigger_od` Trigger on OD digits, rather than ID digits?
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)

There are also parameters used by the GPU version of the code
```
  std::string PMTFile;
  std::string DetectorFile;
  std::string ParameterFile;
```
TODO these should be removed (use WCSimReader / WCSimASCIReader instead)