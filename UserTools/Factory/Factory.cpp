#include "Factory.h"

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
#ifdef FLOWEREXISTS
if (tool=="FLOWERRecon") ret=new FLOWERRecon;
#endif //FLOWEREXISTS
if (tool=="SupernovaDirectionCalculator") ret=new SupernovaDirectionCalculator;
if (tool=="Chunker") ret=new Chunker;
if (tool=="bob") ret=new bob;
if (tool=="bill") ret=new bill;
if (tool=="b1") ret=new b1;
if (tool=="b3") ret=new b3;
if (tool=="b2") ret=new b2;
if (tool=="c1") ret=new c1;
if (tool=="c2") ret=new c2;
  if (tool=="c3") ret=new c3;
return ret;
}

