#include "DataModel.h"

DataModel::DataModel(){}

ReconInfo * DataModel::GetFilter(std::string name, bool can_create)
{
  if(name.compare("ALL") == 0) {
    return &(this->RecoInfo);
  }
  if(this->RecoInfoMap.find(name) == this->RecoInfoMap.end()) {
    if(can_create) {
      this->RecoInfoMap.insert(std::pair<std::string, ReconInfo *>(name, new ReconInfo()));
    }
    else {
      //It makes no sense for certain tools (e.g. ReconDataOut) to create an entry in the map, just to read nothing from it. So don't let it
      return 0;
    }
  }
  return this->RecoInfoMap[name];
}
