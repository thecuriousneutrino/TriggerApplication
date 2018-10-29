#include "WCSimReader.h"

using std::cout;
using std::cerr;
using std::endl;

WCSimReader::WCSimReader():Tool(){}


bool WCSimReader::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  //config reading
  if(! m_variables.Get("nevents",  fNEvents) ) {
    cout << "nevents configuration not found. Reading all events" << endl;
    fNEvents = -1;
  }
  if(! (m_variables.Get("infile",   fInFile) ||
	m_variables.Get("filelist", fFileList))) {
    cerr << "You must use one of the following options: "
	 << " infile filelist" << endl;
    return false;
  }
     
  cout << "fNEvents  \t" << fNEvents  << endl;
  cout << "fInFile   \t" << fInFile   << endl;
  cout << "fFileList \t" << fFileList << endl;

  //open the trees
  fChainOpt   = new TChain("wcsimRootOptionsT");
  fChainEvent = new TChain("wcsimT");
  fChainGeom  = new TChain("wcsimGeoT");

  if(!ReadTree(fChainOpt))
    return false;
  if(!ReadTree(fChainEvent))
    return false;
  if(!ReadTree(fChainGeom))
    return false;

  //ensure that the geometry & options are the same for each file
  if(!CompareTree(fChainOpt))
    return false;
  if(!CompareTree(fChainGeom))
    return false;

  //set number of events
  if(fNEvents <= 0)
    fNEvents = fChainEvent->GetEntries();
  else if (fNEvents > fChainEvent->GetEntries())
    fNEvents = fChainEvent->GetEntries();
  fCurrEvent = 0;

  return true;
}

bool WCSimReader::ReadTree(TChain * chain) {
  //use InFile
  if(fInFile.size()) { 
    if(! chain->Add(fInFile.c_str(), -1)) {
      cerr << "Could not load tree: " << chain->GetName()
	   << " in file(s): " << fInFile << endl;
      return false;
    }
    cout << "Loaded tree: " << chain->GetName()
	 << " from file(s): " << fInFile
	 << " with: " << chain->GetEntries()
	 << " entries" << endl;
    return true;
  }
  //use FileList
  else if(fFileList.size()) {
    cerr << "FileList not implemented" << endl;
    return false;
  }
  else {
    cerr << "Must use one of the following options: "
	 << " infile filelist" << endl;
    return false;
  }
}

bool WCSimReader::CompareTree(TChain * chain)
{
  cerr << "This function to ensure that the geometry (and at least some of the options tree) are identical is not yet implemented" << endl;
  return true;
}

bool WCSimReader::Execute(){
  cout << "Event " << fCurrEvent << " of " << fNEvents << endl;


  //and finally, increment event counter
  fCurrEvent++;
  if(fCurrEvent > fNEvents)
    return false;

  return true;
}


bool WCSimReader::Finalise(){

  delete fChainOpt;
  delete fChainEvent;
  delete fChainGeom;

  return true;
}
