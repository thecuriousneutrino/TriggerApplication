# nhits

Trigger on more than x-digits in a y ns sliding-window

CPU and CUDA-GPU versions are available

## Data

* Adjusts threshold for noise (if set)
* Picks the relevant digits information from the data model (IDSample or ODSamples)
* Calls the GPU or CPU algorithm
  * CPU version
    * Loop over all digits to find first/last times
    * Loop from first hit time to last hit time, with a sliding trigger decision window (size `trigger_search_window`, step `trigger_search_window_step`)
        * For every digit in the trigger decision window, fill a vector with the digit time
    	* If the number of digits is more than the threshold, issue a trigger
	  * Store the trigger in the data model (IDTriggers or ODTriggers)
	    * Trigger type `kTriggerNDigits`
	    * Trigger readout window [trigger time - `pretrigger_save_window`, trigger time + `posttrigger_save_window`] ns
	    * Trigger time: `trigger_threshold`'ed entry in the (sorted) digit time vector (the time of the first digit above threshold)
	      * i.e. if the threshold is 25, the trigger time is taken as the time of the 26th digit
	    * Trigger info: one entry with number of ID digits in the sliding window that issued the trigger
        * Increment the loop by `posttrigger_save_window` (rather than `trigger_search_window_step`)
  * GPU version
    * TODO add explanation here
    * TODO ensure the algorithm is similar (may not be identical due to GPU-related optimisations that can be made), the definitions of trigger time etc are identical, and the results are identical
    * TODO make the GPU version of the code work with the WCSimReader tool
* TODO make both versions work with the WCSimASCIReader tool

## Configuration

```
trigger_search_window WINDOW
trigger_search_window_step STEP
trigger_threshold THRESHOLD
trigger_threshold_adjust_for_noise BOOL
pretrigger_save_window PRETRIGGER
posttrigger_save_window POSTTRIGGER
trigger_od BOOL
use_stopwatch BOOL
stopwatch_file FILENAME
verbose LEVEL
```
* `trigger_search_window` Width of the sliding window, in ns
* `trigger_search_window_step` Step size of the sliding window, in ns
* `trigger_threshold` Trigger threshold - number of triggers must be above this value (equal to does not fire the trigger)
* `trigger_threshold_adjust_for_noise` Add the average dark noise rate to the threshold?
* `pretrigger_save_window` After a positive trigger is found, save digits from `trigger_time - pretrigger_save_window` to `trigger_time + posttrigger_save_window`
* `posttrigger_save_window` After a positive trigger is found, save digits from `trigger_time - pretrigger_save_window` to `trigger_time + posttrigger_save_window`
* `trigger_od` Trigger on OD digits, rather than ID digits?
* `use_stopwatch` Use the Stopwatch functionality implemented for this tool?
* `stopwatch_file` Save the time it takes for each run of `Execute()` to a histogram. Should end in .pdf, .eps, etc.
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)

There are also parameters used by the GPU version of the code
```
  std::string PMTFile;
  std::string DetectorFile;
  std::string ParameterFile;
```
TODO these GPU parameters should be removed (use WCSimReader / WCSimASCIReader instead)
