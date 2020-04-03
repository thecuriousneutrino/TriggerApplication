# MyTool

MyTool

## Data

Describe any data formats MyTool creates, destroys, changes, analyzes, or its usage.




## Configuration

Describe any configuration variables for MyTool.

```
param1 value1
param2 value2
use_stopwatch BOOL
stopwatch_file FILENAME
verbose LEVEL
```
* `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
* `use_stopwatch` Use the Stopwatch functionality implemented for this tool?
* `stopwatch_file` Save the time it takes for each run of `Execute()` to a histogram. Should end in .pdf, .eps, etc.
