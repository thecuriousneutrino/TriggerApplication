# ReconReset

Reset `RecoInfo` objects held in the data model

* WARNING `RecoInfo`, and every `ReconInfo` object within `RecoInfoMap`, are cleared by this tool. All tools that use a `ReconInfo` object in the data model (e.g. `dimfit`) should be run *BEFORE* this tool

* WARNING If you don't use `ReconReset`, then your `RecoInfo` objects won't reset between `Execute()` calls

## Data

* Calls `Reset()` on
  * `RecoInfo`
  * Every entry in `RecoInfoMap`


## Configuration

```
verbose LEVEL
```
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)