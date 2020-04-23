#include "ReconDataOut.h"
#include "TimeDelta.h"

ReconDataOut::ReconDataOut():Tool(){}


bool ReconDataOut::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_verbose = 0;
  m_variables.Get("verbose", m_verbose);

  //Setup and start the stopwatch
  bool use_stopwatch = false;
  m_variables.Get("use_stopwatch", use_stopwatch);
  m_stopwatch = use_stopwatch ? new util::Stopwatch("ReconDataOut") : 0;

  m_stopwatch_file = "";
  m_variables.Get("stopwatch_file", m_stopwatch_file);

  if(m_stopwatch) m_stopwatch->Start();

  m_data= &data;

  //open the output file
  if(! m_variables.Get("outfilename", fOutFilename)) {
    Log("ERROR: outfilename configuration not found. Cancelling initialisation", ERROR, m_verbose);
    return false;
  }
  fOutFile.Open(fOutFilename.c_str(), "RECREATE");

  //open & format the tree
  fTreeRecon = new TTree("reconTree", "Reconstruction information");
  fTreeRecon->Branch("EventNum", &fEvtNum);
  fTreeRecon->Branch("TriggerNum", &fRTTriggerNum);
  fTreeRecon->Branch("NDigits", &fRTNHits);
  fTreeRecon->Branch("Energy", &fRTEnergy);
  fTreeRecon->Branch("Reconstructer", &fRTReconstructer, "Reconstruter/I");
  fTreeRecon->Branch("Time", &fRTTime);
  fTreeRecon->Branch("Vertex", fRTVertex, "Vertex[3]/D");
  fTreeRecon->Branch("GoodnessOfFit", &fRTGoodnessOfFit);
  fTreeRecon->Branch("GoodnessOfTimeFit", &fRTGoodnessOfTimeFit);
  fTreeRecon->Branch("HasDirection", &fRTHasDirection);
  fTreeRecon->Branch("DirectionEuler", fRTDirectionEuler, "DirectionEuler[3]/D");
  fTreeRecon->Branch("CherenkovCone", fRTCherenkovCone, "CherenkovCone[2]/D");
  fTreeRecon->Branch("DirectionLikelihood", &fRTDirectionLikelihood);

  //Get the reconstructed events filter you want to save
  if(!m_variables.Get("input_filter_name", fInputFilterName)) {
    Log("INFO: input_filter_name not given. Using ALL", WARN, m_verbose);
    fInputFilterName = "ALL";
  }
  fInFilter  = m_data->GetFilter(fInputFilterName, false);
  if(!fInFilter) {
    ss << "FATAL: no filter named " << fInputFilterName << " found. Returning false";
    StreamToLog(FATAL);
    return false;
  }

  fEvtNum = 0;

  if(m_stopwatch) Log(m_stopwatch->Result("Initialise"), INFO, m_verbose);

  return true;
}


bool ReconDataOut::Execute(){
  if(m_stopwatch) m_stopwatch->Start();

  const int nrecons = fInFilter->GetNRecons();
  ss << "DEBUG: Saving the result of " << nrecons << " reconstructions";
  StreamToLog(DEBUG1);
  for(int irecon = 0; irecon < nrecons; irecon++) {
    fRTTriggerNum = fInFilter->GetTriggerNum(irecon);
    fRTNHits = fInFilter->GetNHits(irecon);
    fRTEnergy = fInFilter->GetEnergy(irecon);
    fRTReconstructer = fInFilter->GetReconstructer(irecon);
    fRTTime = fInFilter->GetTime(irecon) / TimeDelta::ns;
    Pos3D pos = fInFilter->GetVertex(irecon);
    fRTVertex[0] = pos.x;
    fRTVertex[1] = pos.y;
    fRTVertex[2] = pos.z;
    fRTGoodnessOfFit = fInFilter->GetGoodnessOfFit(irecon);
    fRTGoodnessOfTimeFit = fInFilter->GetGoodnessOfTimeFit(irecon);
    //Direction
    fRTHasDirection = fInFilter->GetHasDirection(irecon);
    if(fRTHasDirection) {
      DirectionEuler direct = fInFilter->GetDirectionEuler(irecon);
      fRTDirectionEuler[0] = direct.theta;
      fRTDirectionEuler[1] = direct.phi;
      fRTDirectionEuler[2] = direct.alpha;
      CherenkovCone cone = fInFilter->GetCherenkovCone(irecon);
      fRTCherenkovCone[0] = cone.cos_angle;
      fRTCherenkovCone[1] = cone.ellipticity;
      fRTDirectionLikelihood = fInFilter->GetDirectionLikelihood(irecon);
    }
    else {
      for(int i = 0; i < 3; i++)
	fRTDirectionEuler[i] = -9E7;
      for(int i = 0; i < 2; i++)
	fRTCherenkovCone [i] = -9E7;
    }
    //Fill the tree
    fTreeRecon->Fill();
  }

  //increment event number
  fEvtNum++;

  if(m_stopwatch) m_stopwatch->Stop();

  return true;
}


bool ReconDataOut::Finalise(){
  if(m_stopwatch) {
    Log(m_stopwatch->Result("Execute", m_stopwatch_file), INFO, m_verbose);
    m_stopwatch->Start();
  }

  //multiple TFiles may be open. Ensure we save to the correct one
  fOutFile.cd(TString::Format("%s:/", fOutFilename.c_str()));
  fTreeRecon->Write();
  fOutFile.Close();
  delete fTreeRecon;

  if(m_stopwatch) {
    Log(m_stopwatch->Result("Finalise"), INFO, m_verbose);
    delete m_stopwatch;
  }

  return true;
}
