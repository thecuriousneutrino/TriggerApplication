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
  * For reconstructed events that pass criteria (fiducial volume, reconstruction tool, etc.) fill a vector of positions
  * If there are enough events that pass the criteria, run dimfit
  * Printout the result of dimfit


## Configuration

Parameters that specify which reconstructed events to use
```
min_events NUM
reconstruction_algorithm ALGSTRING
min_recon_likelihood LIKELIHOOD
min_recon_time_likelihood LIKELIHOOD
max_r_pos DISTANCE
max_z_pos DISTANCE
```
* `min_events` When the number of events in the sliding time window is below this number, dimfit will not be called
* `reconstruction_algorithm` Use the results of which reconstruction algorithm?
* `min_recon_likelihood` Events with a likelihood smaller than this will not be included in dimfit calculations
* `min_recon_time_likelihood` Events with a time-fit likelihood smaller than this will not be included in dimfit calculations
* `max_r_pos` Events with larger reconstructed `r` (in cm) than this will not be included in dimfit calculations
* `max_z_pos` Events with larger reconstructed `z` (in cm) than this will not be included in dimfit calculations
  * Note that this is a detector half-height because the co-ordinate origin is at the detector centre

TODO potentially move the above parameters into a separate tool so that NClusters, dimfit, reconstruction can be run on the same set of reconstructed events without having to keep track of the same variables in different config files

dimfit running parameters
```
time_window DURATION
time_window_step SIZE
R2MIN NUM
LOWDBIAS NUM
GOODPOINT NUM
MAXMEANPOS NUM
verbose LEVEL
```
* `time_window` Duration of the sliding time window, in seconds
* `time_window_step` Step size for the sliding time window, in seconds
* `R2MIN` The chisq must be above this value to reconstruct as volume-like
* `LOWDBIAS` Bias chisq towards/away from(???) volume-like
* `GOODPOINT` If the chisq is less than this, always return point-like
* `MAXMEANPOS` If the average vertex position is outside a sphere of radius `MAXMEANPOS`, don't return volume-like
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
