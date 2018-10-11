#include "../UnityGPU.cpp"

Tool* Factory(std::string tool){
Tool* ret=0;

// if (tool=="Type") tool=new Type;
if (tool=="DummyTool") ret=new DummyTool;

if (tool=="GPUProcessor") ret=new GPUProcessor;
if (tool=="TriggerOutput") ret=new TriggerOutput;
  if (tool=="WCSimASCIReader") ret=new WCSimASCIReader;
return ret;
}

