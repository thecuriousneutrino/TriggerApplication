# Data Model
*************************

Data Model Class can be defined how ever the User requires. A Store is provided which ineficently maps variables to string lkeys via conversion to stringstream and can be used for debuging or other useful vairables.

A TTree map with getter and setter functions is provided and can be uncommented if required.

Table of Contents
=================

   * [Data Model](#data-model)
      * [TriggerApplication data members](#triggerapplication-data-members)
         * [Digit Information](#digit-information)
         * [Geometry Information](#geometry-information)
         * [Trigger Information](#trigger-information)
         * [Reconstruction information](#reconstruction-information)
            * [Helper functions](#helper-functions)
         * [WCSim pass-through information](#wcsim-pass-through-information)
         * [Supernova Trigger](#supernova-trigger)
         * [Misc](#misc)
      * [Related classes](#related-classes)
         * [SubSample](#subsample)
            * [Important data members](#important-data-members)
            * [Important methods](#important-methods)
         * [TriggerInfo](#triggerinfo)
            * [Important data members](#important-data-members-1)
            * [Important methods](#important-methods-1)
         * [PMTInfo](#pmtinfo)
            * [Important data members](#important-data-members-2)
            * [Important methods](#important-methods-2)
         * [ReconInfo](#reconinfo)
            * [Important data members](#important-data-members-3)
            * [Important methods](#important-methods-3)
            * [Related things](#related-things)

## TriggerApplication data members

The variables in this DataModel used by TriggerApplication tools are

### Digit Information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `std::vector<SubSample>` | IDSamples           | Store charge, time, PMT ID of every ID digit | nhits | WCSimReader, WCSimASCIReader | WCSimReader |
|  `std::vector<SubSample>` | ODSamples           | Store charge, time, PMT ID of every OD digit | nhits | WCSimReader | WCSimReader |
|  `WCSimRootEvent *`       | IDWCSimEvent_Triggered      | The triggered `WCSimRootEvent` for the ID (digits are sorted into trigger windows) | BONSAI | DataOut | DataOut |
|  `WCSimRootEvent *`       | ODWCSimEvent_Triggered      | The triggered `WCSimRootEvent` for the OD (digits are sorted into trigger windows) | - | DataOut | DataOut |

### Geometry Information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `std::vector<PMTInfo>`   | IDGeom              | Store x, y, z, PMT ID of every ID PMT        | - | WCSimReader | - |
|  `std::vector<PMTInfo>`   | ODGeom              | Store x, y, z, PMT ID of every OD PMT        | - | WCSimReader | - |
|  `double`                 | IDPMTDarkRate       | The dark rate for ID PMTs | nhits | WCSimReader | - |
|  `double`                 | ODPMTDarkRate       | The dark rate for OD PMTs | nhits | WCSimReader | - |
|  `int`                    | IDNPMTs             | The number of ID PMTs     | nhits, DataOut | WCSimReader | - |
|  `int`                    | ODNPMTs             | The number of OD PMTs     | nhits, DataOut | WCSimReader | - |
|  `bool`                   | HasOD               | Does the geometry include the OD? | DataOut | WCSimReader | - |

### Trigger Information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `TriggerInfo`            | IDTriggers          | Store trigger type, time, readout window start/end times, additional info vector, for ID triggers | DataOut | nhits, pass_all | DataOut |
|  `TriggerInfo`            | ODTriggers          | Store trigger type, time, readout window start/end times, additional info vector, for OD triggers | DataOut | nhits | DataOut |
|  `bool`                   | triggeroutput       | Did a trigger fire?       | TriggerOutput | nhits, test_vertices | - |

### Reconstruction information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `ReconInfo`              | RecoInfo            | Store reconstruction information (vertex time/position, fit likelihoods, optionally direction) | dimfit, ReconDataOut, ReconFilter | BONSAI, ReconRandomiser, ReconDataIn | ReconReset |
| `std::map<std::string, ReconInfo *> | RecoInfoMap | Store filtered reconstruction information (vertex time/position, fit likelihoods, optionally direction)`| dimfit, ReconDataOut, ReconFilter | ReconFilter | ReconReset |

#### Helper functions

| Name | Returns |
| ---- | ------- |
| ReconInfo * GetFilter(std::string name, bool can_create) | If `name == ALL`: pointer to all events (`RecoInfo`) <br> Otherwise, returns pointer to `RecoInfoMap` entry name <br> Caveat: if `!can_create` and `name` not found, return `0` |

### WCSim pass-through information

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `TChain *`               | WCSimGeomTree       | The `WCSimRootGeom` tree from input WCSim file(s)    | DataOut, BONSAI | WCSimReader | - |
|  `TChain *`               | WCSimOptionsTree    | The `WCSimRootOptions` tree from input WCSim file(s) | DataOut | WCSimReader | - |
|  `TChain *`               | WCSimEventTree      | The `WCSimRootEvent` tree from input WCSim file(s)   | | WCSimReader | - |
|  `std::vector<int>`       | CurrentWCSimEventNums | The original WCSim files' event number(s) for the current event | DataOut | WCSimReader | WCSimReader |
|  `TObjArray *`            | CurrentWCSimFiles     | The original WCSim files' filename(s) for the current event     | DataOut | WCSimReader | WCSimReader |
|  `WCSimRootEvent *`       | IDWCSimEvent_Raw      | The original, unmodified `WCSimRootEvent` for the ID | DataOut | WCSimReader | - |
|  `WCSimRootEvent *`       | ODWCSimEvent_Raw      | The original, unmodified `WCSimRootEvent` for the OD | DataOut | WCSimReader | - |

### Supernova Trigger

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
| `std::vector<SNWarningParams>` | SupernovaWarningParameters | Store the dimensionality, number of reconstructed vertices and the highest nclusters warning threshold passed | - | dimfit | - |

### Misc

| Type                      | Name                | Purpose | Read by | Modified by | Reset by |
| ------------------------- | ------------------- | ------- | ------- | ----------- | -------- |
|  `bool`                   | IsMC                | Is the input MC? | | WCSimReader | - |
 
TODO: setup to allow multiple types of PMT in the ID (e.g. 20" + mPMT hybrid geometry). For this, a proposal (Tom) is to
* replace all ID/OD variables in the data model by `std::vector<SubSample> IDTriggers, ODTriggers` with a `std::map<PhotoDetectorType_t, std::vector<SubSample> * >`
* add entries to the map (i.e. create std::vector<SubSample> *) in tools that read the geometry (e.g. `WCSimReader`)
* remove `HasOD`
* PhotoDetectorType_t will be e.g. kOD3Inch, kID20Inch, kIDmPMT for now, but can be given more representative names in the future (e.g. add model numbers)

## Related classes

### SubSample

Stores (in vectors) digit time and PMTID. Optionally, store digit charge

Vectors must be complete at time of creating the SubSample (i.e. there is no `AddDigit()` method)

#### Important data members
```
std::vector<int> m_PMTid;
std::vector<float> m_time;
std::vector<float> m_charge;
std::vector<int> m_time_int;
std::vector<int> m_charge_int;
```
* Note `m_charge` and `m_charge_int` store the same information, in floating/integer formats respectively. Same for `m_time`

#### Important methods

```
```

### TriggerInfo

Stores `m_N` triggers, with information including `TriggerType_t`, readout window start/end times, the trigger time, and additional trigger information `std::vector<float>`

Triggers can be added one at a time, or by copying them all from another `TriggerInfo` instance

Triggers can be sorted into time order using `SortByStartTime()`

#### Important data members
```
unsigned int m_N;
std::vector<TriggerType_t> m_type;
std::vector<double>        m_starttime;
std::vector<double>        m_endtime;
std::vector<double>        m_triggertime;
std::vector<std::vector<float> > m_info;
```
* Note `TriggerType_t` is defined in WCSim in `WCSimEnumerations.hh`

#### Important methods

```
void AddTrigger(TriggerType_t type, double starttime, double endtime, double triggertime, std::vector<float> info)
void AddTriggers(TriggerInfo * in)
void Clear()
void SortByStartTime()
```

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