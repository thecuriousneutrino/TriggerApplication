#include "BONSAI.h"

BONSAI::BONSAI():Tool(){}

bool BONSAI::FileExists(const char * filename) {
  bool exists = access(filename, F_OK) != -1;
  if(!exists) {
    ss << "FATAL: " << filename << " not found or inaccessible";
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

  if(!FileExists("$BONSAIDIR/libBONSAI.so")) {
    Log("FATAL: BONSAI library not found. Ensure the BONSAI library exists at $BONSAIDIR/libBONSAI.so. For more information about BONSAI, see https://github.com/hyperk/hk-BONSAI", FATAL, verbose);
    return false;
  }

  _bonsai = new WCSimBonsai();

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
  fTVertexInfo->Branch("Vertex", &fVertex, "Vertex/D[4]");
  fTVertexInfo->Branch("DirectionEuler", &fDirectionEuler, "DirectionEuler/D[3]");
  fTVertexInfo->Branch("CherenkovCone", &fCherenkovCone, "CherenkovCone/D[2]");
  fTVertexInfo->Branch("DirectionLikelihood", &fDirectionLikelihood);
  fTVertexInfo->Branch("GoodnessOfFit", &fGoodnessOfFit);
  fTVertexInfo->Branch("GoodnessOfTimeFit", &fGoodnessOfTimeFit);

  fEventNum = 0;
  
  return true;
}


bool BONSAI::Execute(){
  float out_vertex[4], out_direction[6], out_maxlike[500];
  int   out_nsel[2];
  
  for (int itrigger = 0 ; itrigger < m_data->WCSimEventID->GetNumberOfEvents(); itrigger++) {
    _trigger = m_data->WCSimEventID->GetTrigger(itrigger);

    //clear the previous triggers' digit information
    _in_PMTIDs->clear();
    _in_Ts->clear();
    _in_Qs->clear();

    //fill the inputs to BONSAI with the current triggers' digit information
    _in_nhits = _trigger->GetNcherenkovdigihits();
    if(_in_nhits <= 0) {
      Log("INFO: No digits in current trigger. Not running BONSAI", INFO, verbose);
      return true;
    }
    long n_not_found = 0;
    for (long idigi=0; idigi < _in_nhits; idigi++) {
      TObject *element = (_trigger->GetCherenkovDigiHits())->At(idigi);
      WCSimRootCherenkovDigiHit *digi = 
	dynamic_cast<WCSimRootCherenkovDigiHit*>(element);
      if(!digi) {
	n_not_found++;
	ss << "WARN: Digit " << idigi << " of " << _in_nhits << "not found in WCSimRootTrigger";
	StreamToLog(WARN);
	continue;
      }
      _in_PMTIDs->push_back(digi->GetTubeId());
      _in_Ts    ->push_back(digi->GetT());
      _in_Qs    ->push_back(digi->GetQ());
    }//idigi
    if(n_not_found) {
      _in_nhits -= n_not_found;
      ss << "WARN: Missing " << n_not_found << " digits";
      StreamToLog(WARN);
    }
    
    //call BONSAI
    _bonsai->BonsaiFit( out_vertex, out_direction, out_maxlike, out_nsel, &_in_nhits, _in_PMTIDs->data(), _in_Ts->data(), _in_Qs->data());

    //fill the output tree variables
    fTriggerNum = itrigger;
    for(int i = 0; i < 4; i++)
      fVertex[i] = out_vertex[i];
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
  
  return true;
}


bool BONSAI::Finalise(){
  fOutFile.Write();
  fOutFile.Close();
  
  delete _bonsai;
  delete _in_PMTIDs;
  delete _in_Ts;
  delete _in_Qs;
  delete fTVertexInfo;
  
  return true;
}
