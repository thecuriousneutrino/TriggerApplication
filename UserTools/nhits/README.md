# nhits

nhits is a trigger that sums the number of digits in a sliding time window. If the sum is above a threshold, a trigger is issued.
CPU and CUDA-GPU versions are available

## Data

* [GPU] Initialises GPU
* Adjusts threshold for noise (if set)
* For each Execute() call
  * Gets the relevant list of hits (i.e. ID or OD)
  * Runs the CPU or GPU algorithm
  * For each trigger found, adds to `IDTriggers` or `ODTriggers` with the properties
    * Type `kTriggerNDigits`
    * Trigger readout window [trigger time - `pretrigger_save_window`, trigger time + `posttrigger_save_window`] ns
    * Trigger time: the time of the first hit above threshold in the sliding window.
      * If the window has 40 hits
      * And the threshold is 25
      * The trigger time will be the time of the 26th digit
      	* Note sorting by time is done to ensure this
    * Info: one entry with number of ID digits in the sliding window that issued the trigger
* [GPU] Finalises GPU


## Configuration

```
trigger_search_window WINDOW
trigger_search_window_step STEP
trigger_threshold THRESHOLD
trigger_threshold_adjust_for_noise [0,1]
pretrigger_save_window PRETRIGGER
posttrigger_save_window POSTTRIGGER
trigger_od [0,1]
verbose LEVEL
```

* `trigger_search_window` The width of the window that sums nhits
* `trigger_search_window_step` The slide step of the window that sums hits
* `trigger_threshold` The threshold, above which the trigger fires
  * *Not equal to* does not fire the trigger
* `trigger_threshold_adjust_for_noise` Boolean. If true, the trigger threshold will be increased by the average dark noise occupancy in the sliding window. If false, the trigger threshold will not be modified
* `pretrigger_save_window` Once a trigger is issued, how much data should be read out before it
* `posttrigger_save_window` Once a trigger is issued, how much data should be read out after it
* `trigger_od` Boolean. If true, the trigger runs on OD hits. If false, the trigger runs on ID hits.
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
