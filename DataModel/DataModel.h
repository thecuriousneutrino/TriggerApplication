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
#include "Utilities.h"

#include <zmq.hpp>

#include <SubSample.h>
#include <PMTInfo.h>
#include <TriggerInfo.h>
#include <ReconInfo.h>

/**
* \class DataModel
 *
 * This class Is a transient data model class for your Tools within the ToolChain. If Tools need to comunicate they pass all data objects through the data model. There fore inter tool data objects should be deffined in this class.
 *
 *
 * $Author: B.Richards $
 * $Date: 2019/05/26 18:34:00 $
 * Contact: b.richards@qmul.ac.uk
 *          
 */

class DataModel {

 public:
  
  DataModel(); ///< Simple constructor

  /// Get filtered reconstructed information.
  ///
  /// If `name == ALL`: pointer to all events (`RecoInfo`)
  /// Otherwise, returns pointer to `RecoInfoMap` entry name
  /// Caveat: if `!can_create` and `name` not found, return `0`
  ReconInfo * GetFilter(std::string name, bool can_create);

  Store vars; ///< This Store can be used for any variables. It is an inefficent ascii based storage    
  BoostStore CStore; ///< This is a more efficent binary BoostStore that can be used to store a dynamic set of inter Tool variables.
  std::map<std::string,BoostStore*> Stores; ///< This is a map of named BooStore pointers which can be deffined to hold a nammed collection of any tipe of BoostStore. It is usefull to store data that needs subdividing into differnt stores.
  
  Logging *Log; ///< Log class pointer for use in Tools, it can be used to send messages which can have multiple error levels and destination end points  

  zmq::context_t* context; ///< ZMQ contex used for producing zmq sockets for inter thread,  process, or computer communication

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

  /// Store the dimensionality, number of reconstructed vertices and the highest nclusters warning threshold passed
  std::vector<SNWarningParams> SupernovaWarningParameters;

  /// Does the geometry include the outer detector?
  bool HasOD;
  /// Is this simulated data?
  bool IsMC;

 private:

};

#endif
