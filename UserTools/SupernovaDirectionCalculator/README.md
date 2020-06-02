# SupernovaDirectionCalculator

SupernovaDirectionCalculator

## Data

The SupernovaDirectionCalculator iterates over all reconstructed vertices
provided by an input filter and calculates the average event direction.
This is should correspond to the supernova neutrino direction.

## Configuration

Describe any configuration variables for SupernovaDirectionCalculator.

```
input_filter_name filter1
weight_events 1
weights_file configfiles/SNTriggering/SN_dir_weights.txt
use_stopwatch 1
#stopwatch_file dimfit_stopwatch.pdf
verbose 3
```

* `input_filter_name` Which filter to use as input.
* `weight_events` Whether to weight events by their reconstructed energy.
* `weights_file` Path to the CSV file containing the events
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
* `use_stopwatch` Use the Stopwatch functionality implemented for this tool?
* `stopwatch_file` Save the time it takes for each run of `Execute()` to a histogram. Should end in .pdf, .eps, etc.

## The weights file

The file containing the weights needs to look like this:

```
log10_Energy,weight
0.22297752156211326,0.0
0.29327299363978826,0.0016633804259996556
0.3635684657174633,0.006585896779828453
0.4338639377951383,0.010466441505762652
0.5041594098728133,0.02799834738253676
...
```

The first line describes the columns. It will be ignored by the tool. The
first column contains the `log10(E_reco)` values, the second column the
weights to be applited.

The actual weights will be calculated by interpolating the given values.
Events with energies out of range will be given the weight corresponding to
the first/last entry in the file.
