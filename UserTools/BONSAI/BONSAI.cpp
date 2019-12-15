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

  _bonsai = new WCSimBonsai();
  WCSimRootGeom * geo = 0;
  m_data->WCSimGeomTree->SetBranchAddress("wcsimrootgeom", &geo);
  m_data->WCSimGeomTree->GetEntry(0);
  _bonsai->Init(geo);
  m_data->WCSimGeomTree->ResetBranchAddresses();

  _in_PMTIDs = new std::vector<int>  (1000);
  _in_Ts     = new std::vector<float>(1000);
  _in_Qs     = new std::vector<float>(1000);

  //open the output file
  if(! m_variables.Get("outfilename", fOutFilename)) {
    Log("ERROR: outfilename configuration not found. Cancelling initialisation", ERROR, verbose);
    return false;
  }
  fOutFile.Open(fOutFilename.c_str(), "RECREATE");

  //open & format the tree
  fTVertexInfo = new TTree("vertexInfo", "Vertex information");
  fTVertexInfo->Branch("EventNum", &fEventNum);
  fTVertexInfo->Branch("TriggerNum", &fTriggerNum);
  fTVertexInfo->Branch("NDigits", &_in_nhits);
  fTVertexInfo->Branch("Vertex", fVertex, "Vertex[4]/D");
  fTVertexInfo->Branch("DirectionEuler", fDirectionEuler, "DirectionEuler[3]/D");
  fTVertexInfo->Branch("CherenkovCone", fCherenkovCone, "CherenkovCone[2]/D");
  fTVertexInfo->Branch("DirectionLikelihood", &fDirectionLikelihood);
  fTVertexInfo->Branch("GoodnessOfFit", &fGoodnessOfFit);
  fTVertexInfo->Branch("GoodnessOfTimeFit", &fGoodnessOfTimeFit);

  fEventNum = 0;
  
  return true;
}


bool BONSAI::Execute(){
  Log("DEBUG: BONSAI::Execute() Starting", DEBUG1, verbose);

  float out_vertex[4], out_direction[6], out_maxlike[500];
  int   out_nsel[2];
  
  for (int itrigger = 0 ; itrigger < m_data->IDWCSimEvent_Triggered->GetNumberOfEvents(); itrigger++) {
    _trigger = m_data->IDWCSimEvent_Triggered->GetTrigger(itrigger);

    //clear the previous triggers' digit information
    _in_PMTIDs->clear();
    _in_Ts->clear();
    _in_Qs->clear();

    //fill the inputs to BONSAI with the current triggers' digit information
    _in_nhits = _trigger->GetNcherenkovdigihits();
    int nhits_slots = _trigger->GetNcherenkovdigihits_slots();
    if(_in_nhits <= 0) {
      Log("INFO: No digits in current trigger. Not running BONSAI", INFO, verbose);
      return true;
    }

    long n_not_found = 0;
    for (long idigi=0; idigi < nhits_slots; idigi++) {
      TObject *element = (_trigger->GetCherenkovDigiHits())->At(idigi);
      WCSimRootCherenkovDigiHit *digi = 
	dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
      if(!digi) {
	n_not_found++;
	//this happens regularly because removing digits doesn't shrink the TClonesArray
	ss << "DEBUG: Digit " << idigi << " of " << _in_nhits << "not found in WCSimRootTrigger";
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

    //fill the output tree variables
    fTriggerNum = itrigger;
    ss << "DEBUG: Vertex reconstructed at x, y, z, t:";
    for(int i = 0; i < 4; i++) {
      fVertex[i] = out_vertex[i];
      ss << " " << fVertex[i] << ",";
    }
    StreamToLog(DEBUG1);
    for(int i = 0; i < 3; i++)
      fDirectionEuler[i] = out_direction[i];
    for(int i = 0; i < 2; i++)
      fCherenkovCone[i] = out_direction[i+3];
    fDirectionLikelihood = out_direction[5];
    fGoodnessOfFit = out_maxlike[2];
    fGoodnessOfTimeFit = out_maxlike[1];

    fTVertexInfo->Fill();
  }//itrigger

  fEventNum++;
  
  Log("DEBUG: BONSAI::Execute() Done", WARN, verbose);

  return true;
}


bool BONSAI::Finalise(){
  //multiple TFiles may be open. Ensure we save to the correct one
  fOutFile.cd(TString::Format("%s:/", fOutFilename.c_str()));
  fTVertexInfo->Write();
  fOutFile.Close();
  
  delete _bonsai;
  delete _in_PMTIDs;
  delete _in_Ts;
  delete _in_Qs;
  delete fTVertexInfo;
  
  return true;
}
