# ReconRandomiser

Produce randomised vertex distributions

Multiple versions of the tools can be run in a toolChain to create complicated event distributions
* e.g. to add a volume-like distribution on top of a point-like distribution

Optionally
* Apply smearing (emulating reconstruction, to 0th order) (TODO)
* Apply fixing points on a grid (emulating fixed grid-based reconstruction, to 0th order)
* Also produce directions (TODO)

## Data

* Writes output to `RecoInfo`
  * Likelihoods and NHits are set to a large number, so they are always accepted by future tools
  * The Reconstructer type is set to `kReconRandomNoDirection` or `kReconRandom`



## Configuration

### Vertex distribution

```
n_vertices_mean N

x_mean_pos POS
x_width WIDTH
y_mean_pos POS
y_width WIDTH
z_mean_pos POS
z_width WIDTH

max_z_pos FLOAT
max_r_pos FLOAT

flat_r BOOL
```

Specify the simulated vertex distribution to generate.

In order to create a uniform distribution, set `max_z_pos` and `max_r_pos` to the size of your detector, and `x_width`, `y_width`, `z_width` all to negative values
* `n_vertices_mean` Poisson mean of the number of vertices to generate
* `x_mean_pos` The mean `x` position (cm)
* `x_width` The width of the Gaussian in `x (cm)`
  * If 0, no throws
  * If <0, throw uniformly between `±max_r_pos`. In this case, `x_mean_pos` is ignored
* `y_mean_pos` The mean `y` position (cm)
* `y_width` The width of the Gaussian in `y` (cm)
  * If `0`, no throws
  * If `<1E-6`, throw uniformly between `±max_r_pos`. In this case, `y_mean_pos` is ignored
* `z_mean_pos` The mean `z` position (cm)
* `z_width` The width of the Gaussian in `z` (cm)
  * If 0, no throws
  * If <0, throw uniformly between `±max_z_pos`. In this case, `z_mean_pos` is ignored
* `max_r_pos` Events with larger `r` (in cm) than this will be rethrown
* `max_z_pos` Events with larger `|z|` (in cm) than this will be rethrown
  * Note that this is a detector half-height because the co-ordinate origin is at the detector centre
* `flat_r` If this is true, and both `x_width` and `y_width` are negative (i.e. generating a random distribution in the circularly plane), use this to generate a flat distribution in `r` (rather than flat in `x` and `y`)

### Time distribution

```
t_min TIME
t_max TIME
```

* `t_min` Times are generated uniformly in the range `t_min` to `t_max` (ns)
* `t_max` Times are generated uniformly in the range `t_min` to `t_max` (ns)

### Misc

```
nevents N
seed SEED
verbose LEVEL
```

* `nevents` Run Execute() this number of times. If not given or negative, will revert to 1
* `seed` The random seed to use. Default 0 (always different, but not reproducable!)
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
