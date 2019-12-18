#include "DataModel.h"

DataModel::DataModel(){}

/*
TTree* DataModel::GetTTree(std::string name){

  return m_trees[name];

}


void DataModel::AddTTree(std::string name,TTree *tree){

  m_trees[name]=tree;

}


void DataModel::DeleteTTree(std::string name){

  m_trees.erase(name);

}

*/

ReconInfo * DataModel::GetFilter(std::string name)
{
  if(name.compare("ALL") == 0) {
    return &(this->RecoInfo);
  }
  if(this->RecoInfoMap.find(name) == this->RecoInfoMap.end())
    this->RecoInfoMap.insert(std::pair<std::string, ReconInfo *>(name, new ReconInfo()));
  return this->RecoInfoMap[name];
}
