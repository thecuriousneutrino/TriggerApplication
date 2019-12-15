#include "BONSAI.h"

BONSAI::BONSAI():Tool(){}

bool BONSAI::FileExists(std::string pathname, std::string filename) {
  string filepath = pathname + "/" + filename;
  bool exists = access(filepath.c_str(), F_OK) != -1;
  if(!exists) {
    ss << "FATAL: " << filepath << " not found or inaccessible";
    StreamToLog(FATAL);
    return false;
  }
  return true;
}

bool BONSAI::Initialise(std::string configfile, DataModel &data){
  
  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  if(!FileExists(std::getenv("BONSAIDIR"), "libWCSimBonsai.so")) {
    Log("FATAL: BONSAI library not found. Ensure the BONSAI library exists at $BONSAIDIR/libWCSimBonsai.so. For more information about BONSAI, see https://github.com/hyperk/hk-BONSAI", FATAL, verbose);
    return false;
  }

  m_variables.Get("nhitsmin", fNHitsMin);
  m_variables.Get("nhitsmax", fNHitsMax);

  //setup BONSAI with the geometry
  _bonsai = new WCSimBonsai();
  WCSimRootGeom * geo = 0;
  m_data->WCSimGeomTree->SetBranchAddress("wcsimrootgeom", &geo);
  m_data->WCSimGeomTree->GetEntry(0);
  _bonsai->Init(geo);
  m_data->WCSimGeomTree->ResetBranchAddresses();

  //allocate memory for the digit vectors
  _in_PMTIDs = new std::vector<int>  (fNHitsMax);
  _in_Ts     = new std::vector<float>(fNHitsMax);
  _in_Qs     = new std::vector<float>(fNHitsMax);

  //make other tools aware that there exists a tool that reconstructs
  m_data->HasReconstructionTool = true;

  return true;
}


bool BONSAI::Execute(){
  Log("DEBUG: BONSAI::Execute() Starting", DEBUG1, verbose);

  float out_vertex[4], out_direction[6], out_maxlike[500];
  int   out_nsel[2];
  double dout_vertex[3], dout_direction[3], dout_cone[2];
  
  for (int itrigger = 0 ; itrigger < m_data->IDWCSimEvent_Triggered->GetNumberOfEvents(); itrigger++) {
    _trigger = m_data->IDWCSimEvent_Triggered->GetTrigger(itrigger);

    //clear the previous triggers' digit information
    _in_PMTIDs->clear();
    _in_Ts->clear();
    _in_Qs->clear();

    //get the number of digits
    _in_nhits = _trigger->GetNcherenkovdigihits();
    int nhits_slots = _trigger->GetNcherenkovdigihits_slots();

    //don't run bonsai on large or small events
    if(_in_nhits < fNHitsMin || _in_nhits > fNHitsMax) {
      ss << "INFO: " << _in_nhits << " digits in current trigger. Not running BONSAI";
      StreamToLog(INFO);
      return true;
    }

    //fill the inputs to BONSAI with the current triggers' digit information
    long n_not_found = 0;
    for (long idigi=0; idigi < nhits_slots; idigi++) {
      TObject *element = (_trigger->GetCherenkovDigiHits())->At(idigi);
      WCSimRootCherenkovDigiHit *digi = 
	dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
      if(!digi) {
	n_not_found++;
	//this happens regularly because removing digits doesn't shrink the TClonesArray
	ss << "DEBUG: Digit " << idigi << " of " << _in_nhits << " not found in WCSimRootTrigger";
	StreamToLog(DEBUG2);
	continue;
      }
      ss << "DEBUG: Digit " << idigi << " at time " << digi->GetT();
      StreamToLog(DEBUG2);
      _in_PMTIDs->push_back(digi->GetTubeId());
      _in_Ts    ->push_back(digi->GetT());
      _in_Qs    ->push_back(digi->GetQ());
    }//idigi
    int digits_found = nhits_slots - n_not_found;
    if(_in_nhits != digits_found) {
      ss << "WARN: BONSAI expected " << _in_nhits << " digits. Found " << digits_found;
      StreamToLog(WARN);
      _in_nhits = digits_found;
    }
    
    ss << "DEBUG: BONSAI running over " << _in_nhits << " digits";
    StreamToLog(DEBUG1);

    //call BONSAI
    _bonsai->BonsaiFit( out_vertex, out_direction, out_maxlike, out_nsel, &_in_nhits, _in_PMTIDs->data(), _in_Ts->data(), _in_Qs->data());

    ss << "DEBUG: Vertex reconstructed at x, y, z, t:";
    for(int i = 0; i < 4; i++) {
      ss << " " << out_vertex[i] << ",";
    }
    StreamToLog(DEBUG1);

    //fill the data model with the result
    // need to convert to double...
    for(int i = 0; i < 3; i++) {
      dout_vertex[i]    = out_vertex[i];
      dout_direction[i] = out_direction[i];
    }
    for(int i = 0; i < 2; i++)
      dout_cone[i] = out_direction[i+3];
    
    m_data->RecoInfo.AddRecon(kReconBONSAI, itrigger, _in_nhits, out_vertex[3], &(dout_vertex[0]), out_maxlike[2], out_maxlike[1],
			      &(dout_direction[0]), &(dout_cone[0]), out_direction[5]);

  }//itrigger

  Log("DEBUG: BONSAI::Execute() Done", WARN, verbose);

  return true;
}


bool BONSAI::Finalise(){

  delete _bonsai;
  delete _in_PMTIDs;
  delete _in_Ts;
  delete _in_Qs;
  
  return true;
}
