#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <map>
#include <string>
#include <vector>

#include "TChain.h"

#include "WCSimRootOptions.hh"
#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"
#include "WCSimEnumerations.hh"

#include "Store.h"
#include "BoostStore.h"
#include "Logging.h"

#include <zmq.hpp>

#include <SubSample.h>
#include <PMTInfo.h>
#include <TriggerInfo.h>
#include <ReconInfo.h>

class DataModel {

 public:

  DataModel();

  /// Get filtered reconstructed information.
  ///
  /// If `name == ALL`: pointer to all events (`RecoInfo`)
  /// Otherwise, returns pointer to `RecoInfoMap` entry name
  /// Caveat: if `!can_create` and `name` not found, return `0`
  ReconInfo * GetFilter(std::string name, bool can_create);

  Store vars;
  BoostStore CStore;
  std::map<std::string,BoostStore*> Stores;

  Logging *Log;

  zmq::context_t* context;

  /// Inner detector digit collections
  std::vector<SubSample> IDSamples;
  /// Outer detector digit collections
  std::vector<SubSample> ODSamples;

  /// Geometry information for the inner detector
  std::vector<PMTInfo> IDGeom;
  /// Geometry information for the outer detector
  std::vector<PMTInfo> ODGeom;

  /// Triggered time windows for the inner detector
  TriggerInfo IDTriggers;
  /// Triggered time windows for the outer detector
  TriggerInfo ODTriggers;

  /// DEPRECATED! Use IDTriggers and ODTriggers instead.
  __attribute__((deprecated))
  bool triggeroutput;

  /// Store reconstruction information (vertex time/position, fit likelihoods, optionally direction)
  ReconInfo RecoInfo;
  /// Store filtered reconstruction information
  std::map<std::string, ReconInfo*> RecoInfoMap;

  /// Dark noise rate of inner detector PMTs, unit: ?
  double IDPMTDarkRate;
  /// Dark noise rate of outer detector PMTs, unit: ?
  double ODPMTDarkRate;

  /// Number of inner detector PMTs
  int IDNPMTs;
  /// Number of outer detector PMTs
  int ODNPMTs;

  /// height of water tank
  double detector_length;
  /// radius of water tank
  double detector_radius;
  /// radius of each PMT
  double pmt_radius;

  /// The `WCSimRootGeom` tree from input WCSim file(s)
  TChain * WCSimGeomTree;
  /// The `WCSimRootOptions` tree from input WCSim file(s)
  TChain * WCSimOptionsTree;
  /// The `WCSimRootEvent` tree from input WCSim file(s)
  TChain * WCSimEventTree;
  /// The original WCSim files' event number for the current event
  int CurrentWCSimEventNum;
  /// The original WCSim files' filename for the current event
  TObjString CurrentWCSimFile;
  /// The original, unmodified `WCSimRootEvent` for the ID
  WCSimRootEvent * IDWCSimEvent_Raw;
  /// The original, unmodified `WCSimRootEvent` for the OD
  WCSimRootEvent * ODWCSimEvent_Raw;
  /// The triggered `WCSimRootEvent` for the ID (digits are sorted into trigger windows)
  WCSimRootEvent * IDWCSimEvent_Triggered;
  /// The triggered `WCSimRootEvent` for the OD (digits are sorted into trigger windows)
  WCSimRootEvent * ODWCSimEvent_Triggered;

  /// Store the dimensionality, number of reconstructed vertices and the highest nclusters warning threshold passed
  std::vector<SNWarningParams> SupernovaWarningParameters;

  /// Does the geometry include the outer detector?
  bool HasOD;
  /// Is this simulated data?
  bool IsMC;

 private:

};

#endif
