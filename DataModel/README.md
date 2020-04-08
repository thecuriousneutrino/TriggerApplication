# DataModel

The DataModel class can be defined how ever the user requires. A Store is
provided which ineficently maps variables to string lkeys via conversion to
stringstream and can be used for debuging or other useful vairables.

## TODO

Set up to allow multiple types of PMT in the ID (e.g. 20" + mPMT hybrid geometry). For this, a proposal (Tom) is to
* replace all ID/OD variables in the data model by `std::vector<SubSample> IDTriggers, ODTriggers` with a `std::map<PhotoDetectorType_t, std::vector<SubSample> * >`
* add entries to the map (i.e. create std::vector<SubSample> *) in tools that read the geometry (e.g. `WCSimReader`)
* remove `HasOD`
* PhotoDetectorType_t will be e.g. kOD3Inch, kID20Inch, kIDmPMT for now, but can be given more representative names in the future (e.g. add model numbers)

## Related classes

### SubSample

Stores (in vectors) relative digit time and PMTID. Optionally, store digit charge.

Vectors of hit information are added using the `Append` method.

All digit times are relative to the `m_timestamp`. When comparing times between
different SubSamples, this needs to be taken into account.

### TimeDelta

Universal class to store bothe long and short time information. These are used
for trigger times that are aware of the timestamp realtive to which they were
created.

### TriggerInfo

Stores `m_N` triggers, with information including `TriggerType_t`, readout window start/end times, the trigger time, and additional trigger information `std::vector<float>`

Triggers can be added one at a time, or by copying them all from another `TriggerInfo` instance

Triggers can be sorted into time order using `SortByStartTime()`

### PMTInfo

Stores the PMT ID, and the x, y, z position of each tube

#### Important data members
```
int m_tubeno;
float m_x, m_y, m_z;
```

#### Important methods

```
```

### ReconInfo

Stores `fNRecons` results of reconstruction algorithms
* Vertex position / time (required)
* Directionality (optional)
* Energy (optional) TODO
* Likelihoods associated with each fit

Reconstruction results can be added one at a time, or by copying them all from another `TriggerInfo` instance

Keeps track of the earliest/latest reconstructed event time (useful for looping by future tools)

#### Important data members
```
int    fNRecons;
double fFirstTime;
double fLastTime;
//event info
std::vector<Reconstructer_t> fReconstructer;
std::vector<int>             fTriggerNum;
std::vector<int>             fNHits;
//vertex info
std::vector<double>          fTime;
std::vector<Pos3D>           fVertex;
std::vector<double>          fGoodnessOfFit;
std::vector<double>          fGoodnessOfTimeFit;
//direction info
std::vector<bool>            fHasDirection;
std::vector<DirectionEuler>  fDirectionEuler;
std::vector<CherenkovCone>   fCherenkovCone;
std::vector<double>          fDirectionLikelihood;
```

#### Important methods

```
// Add vertex result
void AddRecon(Reconstructer_t reconstructer, int trigger_num, int nhits, double time, double * vertex, double goodness_of_fit, double goodness_of_time_fit, bool fill_has\
_direction = true)

// Add vertex + direction result
void AddRecon(Reconstructer_t reconstructer, int trigger_num, int nhits, double time, double * vertex, double goodness_of_fit, double goodness_of_time_fit,
                double * direction_euler, double * cherenkov_cone, double direction_likelihood)

// Add vertex(+ direction) result from reconstruction irecon within in
void AddReconFrom(ReconInfo * in, const int irecon)

void Reset()
```

#### Related things

* `enum Reconstructer_t` - an enumeration of reconstruction tools. E.g. `kReconBONSAI`
* `enum NClustersWarning_t` - an enumeration of the nclusters warning thresholds used in the supernova trigger
* `enum SNWarning_t` - an enumeration of supernova warning levels that are assigned based on the values of nclusters and dimensionality
* `struct SNWarningParams` holds nclusters, dim and highest nclusters threshold passed
* `struct Pos3D` holds x, y, z positions
* `struct DirectionEuler` holds theta, phi, alpha directions
* `struct CherenkovCone` holds cos_angle, ellipticity of the Chrenkov cone

Note: this is the format that BONSAI returns directions. Other conventions might be more useful for SN trigger purposes
