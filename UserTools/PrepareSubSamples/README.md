# PrepareSubSamples

PrepareSubSamples will take the provided SubSamples in the DataModel and make
them correspond to the expected format from real data. Specifically it will:

1.  Sort all digits in the SubSamples by their time.
2.  Split SubSamples that are too long into multiple (slightly overlapping)
    ones.
3.  Ensure positive, small digit times by adjusting the timestamp of each
    SubSample.

## Data

PrepareSubSamples will change/replace the ID and OD SubSamples in the DataModel
if necessary.

## Configuration


```
sample_width WIDTH
sample_overlap OVERLAP
verbose LEVEL
```

*   `sample_width` The (maximum) width of each SubSample in ns
*   `sample_overlap` The overlap of consecutive samples in ns
*   `verbose` Verbosity level. Runs from 0 (low verbosity) to 9 (high verbosity)
