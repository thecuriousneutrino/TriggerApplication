#include "../Unity.cpp"

Tool* Factory(std::string tool){
  Tool* ret=0;

  // if (tool=="Type") tool=new Type;
  if (tool=="DummyTool") ret=new DummyTool;

  if (tool=="TriggerOutput") ret=new TriggerOutput;
  if (tool=="WCSimASCIReader") ret=new WCSimASCIReader;
  if (tool=="nhits") ret=new nhits;
  if (tool=="test_vertices") ret=new test_vertices;
  if (tool=="WCSimReader") ret=new WCSimReader;
  if (tool=="DataOut") ret=new DataOut;
  if (tool=="BONSAI") ret=new BONSAI;

  return ret;
}

