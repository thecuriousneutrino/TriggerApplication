#include "../Unity.cpp"

Tool* Factory(std::string tool){
Tool* ret=0;

// if (tool=="Type") tool=new Type;
if (tool=="DummyTool") ret=new DummyTool;

if (tool=="TriggerOutput") ret=new TriggerOutput;
if (tool=="WCSimASCIReader") ret=new WCSimASCIReader;
if (tool=="NHits") ret=new NHits;
if (tool=="test_vertices") ret=new test_vertices;
if (tool=="WCSimReader") ret=new WCSimReader;
if (tool=="DataOut") ret=new DataOut;
#ifdef BONSAIEXISTS
if (tool=="BONSAI") ret=new BONSAI;
#endif //BONSAIEXISTS
if (tool=="ReconDataOut") ret=new ReconDataOut;
if (tool=="dimfit") ret=new dimfit;
if (tool=="ReconRandomiser") ret=new ReconRandomiser;
if (tool=="ReconDataIn") ret=new ReconDataIn;
if (tool=="pass_all") ret=new pass_all;

if (tool=="ReconFilter") ret=new ReconFilter;
if (tool=="ReconReset") ret=new ReconReset;
if (tool=="PrepareSubSamples") ret=new PrepareSubSamples;
#ifdef EBONSAIEXISTS
if (tool=="EnergeticBONSAI") ret=new EnergeticBONSAI;
#endif //EBONSAIEXISTS
return ret;
}

