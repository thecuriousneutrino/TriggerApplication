# dimfit

dimfit analyses the distribution of reconstructed vertices and determines whether the dimensonality corresponds to:
| 0 | Point-like  |
| 1 | Line-like   |
| 2 | Plane-like  |
| 3 | Volume-like |

Supernovae should appear volume-like

Backgrounds should appear non-volume-like. e.g.
* Radioactive sources should appear point-like
* Muon-induced backgrounds should appear line-like

## Data

* Over a sliding time window
  * Reads reconstructed event information from `RecoInfo` in data model
  * For reconstructed events that pass a filter (see `ReconFilter`), or all events, fill a vector of positions
  * If there are enough events that pass the criteria, run dimfit
  * Printout the result of dimfit
  * Compare the number of reconstructed vertices to the thresholds for golden, normal and silent warnings
  * Pass dimensionality, nclusters and highest nclusters_warning passed to data model


## Configuration

### Parameters that specify which reconstructed events to use
```
input_filter_name NAME
min_events NUM
time_window DURATION
time_window_step SIZE
```
* `input_filter_name` Name of the `ReconInfo` object to take from the map `RecoInfoMap`
  * If ALL takes the all reconstructed events from the global object `RecoInfo`
* `min_events` When the number of events in the sliding time window is below this number, dimfit will not be called
* `time_window` Duration of the sliding time window, in seconds
* `time_window_step` Step size for the sliding time window, in seconds

### dimfit running parameters
```
R2MIN NUM
LOWDBIAS NUM
GOODPOINT NUM
MAXMEANPOS NUM
```
* `R2MIN` The chisq must be above this value to reconstruct as volume-like
* `LOWDBIAS` Bias chisq towards/away from(???) volume-like
* `GOODPOINT` If the chisq is less than this, always return point-like
* `MAXMEANPOS` If the average vertex position is outside a sphere of radius `MAXMEANPOS`, don't return volume-like

### nclusters running parameters
```
nclusters_silent_warning NUM
nclusters_normal_warning NUM
nclusters_golden_warning NUM
```
* `nclusters_silent_warning` The number of vertices needed to trigger a silent warning
* `nclusters_normal_warning` The number of vertices needed to trigger a normal warning
* `nclusters_golden_warning` The number of vertices needed to trigger a golden warning

### Misc
```
verbose LEVEL
```
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
