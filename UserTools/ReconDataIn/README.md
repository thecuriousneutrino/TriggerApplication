# ReconDataIn

Read a `reconTree` `TTree` into the `ReconInfo` of `DataModel`

## Data

* Sets up a `TChain` for `reconTree` using `infilenames` as the filename(s)
* For every entry in the tree
  * Add an entry to `ReconInfo` in the `DataModel`

* WARNING: if you use this tool for `>1` events, the same data will be added to `ReconInfo` every time (i.e. the entire `TChain` is added to `ReconInfo` every event)

## Configuration

```
infilenames /path/to/files
verbose LEVEL
```
* `infilenames` Path to the input file(s). Accepts wildcards (results are read into a `TChain`)
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)