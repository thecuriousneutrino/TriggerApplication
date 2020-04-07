# nhits

Trigger on more than x-digits in a y ns sliding-window

CPU and CUDA-GPU versions are available

## Data

* Adjusts threshold for noise (if set)
* Picks the relevant digits information from the data model (IDSample or ODSamples)
* Calls the GPU or CPU algorithm
  * CPU version
    * Sort digits by time
    * Loop over all digits
    * Find digit with more than threshold digits in window preceding digit time
      * If found, issue trigger with pre and post trigger window around that hit
      * Continue with first hit outside the post-trigger window
  * GPU version
    * TODO add explanation here
    * TODO ensure the algorithm is similar (may not be identical due to GPU-related optimisations that can be made), the definitions of trigger time etc are identical, and the results are identical
    * TODO make the GPU version of the code work with the WCSimReader tool
* TODO make both versions work with the WCSimASCIReader tool

## Configuration

```
trigger_search_window WINDOW
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
* `trigger_threshold` Trigger threshold - number of triggers must be above this value (equal to does not fire the trigger)
* `trigger_threshold_adjust_for_noise` Add the average dark noise rate to the threshold?
* `pretrigger_save_window` After a trigger is found, save digits from `trigger_time - pretrigger_save_window` to `trigger_time + posttrigger_save_window`
* `posttrigger_save_window` After a trigger is found, save digits from `trigger_time - pretrigger_save_window` to `trigger_time + posttrigger_save_window`
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
