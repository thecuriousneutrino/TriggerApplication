#include "ReconInfo.h"

ReconInfo::ReconInfo()
 : fNRecons(0),
   fFirstTime(+9E20),
   fLastTime(-9E20)
{

}

void ReconInfo::AddRecon(Reconstructer_t reconstructer, int trigger_num,
			 int nhits, double time, double * vertex,
			 double goodness_of_fit, double goodness_of_time_fit,
			 bool fill_has_direction)
{
  if(fill_has_direction) {
    if(ReconInfo::ShouldProvideDirection(reconstructer)) {
      std::cerr << "Reconstructer " << ReconInfo::EnumAsString(reconstructer) << " provides direction information but we are filling reconstruction information without it" << std::endl;
    }
    fHasDirection.push_back(false);
  }
  fReconstructer.push_back(reconstructer);
  fTriggerNum.push_back(trigger_num);
  fNHits.push_back(nhits);
  fTime.push_back(time);
  Pos3D pos;
  pos.x = vertex[0];
  pos.y = vertex[1];
  pos.z = vertex[2];
  fVertex.push_back(pos);
  fGoodnessOfFit.push_back(goodness_of_fit);
  fGoodnessOfTimeFit.push_back(goodness_of_time_fit);
  fNRecons++;
  UpdateTimeBoundaries(time);
}

void ReconInfo::AddRecon(Reconstructer_t reconstructer, int trigger_num,
			 int nhits, double time, double * vertex,
			 double goodness_of_fit, double goodness_of_time_fit,
			 double * direction_euler, double * cherenkov_cone,
			 double direction_likelihood)
{
  AddRecon(reconstructer, trigger_num, nhits, time, vertex, goodness_of_fit, goodness_of_time_fit, false);
  fHasDirection.push_back(true);
  DirectionEuler direct;
  direct.theta = direction_euler[0];
  direct.phi   = direction_euler[1];
  direct.alpha = direction_euler[2];
  fDirectionEuler.push_back(direct);
  CherenkovCone cone;
  cone.cos_angle = cherenkov_cone[0];
  cone.ellipticity = cherenkov_cone[1];
  fCherenkovCone.push_back(cone);
  fDirectionLikelihood.push_back(direction_likelihood);
}

void ReconInfo::AddReconFrom(ReconInfo * in, const int irecon)
{
  fReconstructer.push_back(in->GetReconstructer(irecon));
  fTriggerNum.push_back(in->GetTriggerNum(irecon));
  fNHits.push_back(in->GetNHits(irecon));
  fTime.push_back(in->GetTime(irecon));
  fVertex.push_back(in->GetVertex(irecon));
  fGoodnessOfFit.push_back(in->GetGoodnessOfFit(irecon));
  fGoodnessOfTimeFit.push_back(in->GetGoodnessOfTimeFit(irecon));
  fHasDirection.push_back(in->GetHasDirection(irecon));
  if(in->GetHasDirection(irecon)) {
    fDirectionEuler.push_back(in->GetDirectionEuler(irecon));
    fCherenkovCone.push_back(in->GetCherenkovCone(irecon));
    fDirectionLikelihood.push_back(in->GetDirectionLikelihood(irecon));
  }
  fNRecons++;
  UpdateTimeBoundaries(in->GetTime(irecon));
}

std::string ReconInfo::EnumAsString(Reconstructer_t r)
{
  switch(r) {
  case (kReconBONSAI):
    return "BONSAI";
    break;
  case (kReconTestVerticesNoDirection):
    return "TestVertices_NoDirection";
    break;
  case (kReconTestVertices):
    return "TestVertices";
    break;
  case(kReconRandomNoDirection):
    return "Random_NoDirection";
    break;
  case (kReconRandom):
    return "Random";
    break;
  default:
    return "";
  }
  return "";
}

Reconstructer_t ReconInfo::ReconstructerFromString(std::string s)
{
  for(int i = int(kReconUndefined)+1; i <= kReconRandom; i++) {
    if(s.compare(ReconInfo::EnumAsString((Reconstructer_t)i)) == 0) {
      return (Reconstructer_t)i;
    }
  }
  std::cerr << "ReconInfo::ReconstructerFromString() Unknown string value " << s << std::endl;
  return kReconUndefined;
}

std::string ReconInfo::EnumAsString(NClustersWarning_t w)
{
  switch(w) {
  case (kNClustersStandard):
    return "NoNClustersWarning";
    break;
  case (kNClustersSilent):
    return "NClustersSilentWarning";
    break;
  case (kNClustersNormal):
    return "NClustersNormalWarning";
    break;
  case (kNClustersGolden):
    return "NClustersGoldenWarning";
    break;
  default:
    return "";
  }
  return "";
}

NClustersWarning_t ReconInfo::NClustersWarningFromString(std::string s)
{
  for(int i = int(kNClustersUndefined)+1; i <= kNClustersGolden; i++) {
    if(s.compare(ReconInfo::EnumAsString((NClustersWarning_t)i)) == 0) {
      return (NClustersWarning_t)i;
    }
  }
  std::cerr << "ReconInfo::NClustersWarningFromString() Unknown string value " << s << std::endl;
  return kNClustersUndefined;
}

std::string ReconInfo::EnumAsString(SNWarning_t w)
{
  switch(w) {
  case (kSNWarningStandard):
    return "NoSupernovaWarning";
    break;
  case (kSNWarningSilent):
    return "SupernovaSilentWarning";
    break;
  case (kSNWarningNormal):
    return "SupernovaNormalWarning";
    break;
  case (kSNWarningGolden):
    return "SupernovaGoldenWarning";
    break;
  default:
    return "";
  }
  return "";
}

SNWarning_t ReconInfo::SNWarningFromString(std::string s)
{
  for(int i = int(kSNWarningUndefined)+1; i <= kSNWarningGolden; i++) {
    if(s.compare(ReconInfo::EnumAsString((SNWarning_t)i)) == 0) {
      return (SNWarning_t)i;
    }
  }
  std::cerr << "ReconInfo::SNWarningFromString() Unknown string value " << s << std::endl;
  return kSNWarningUndefined;
}

bool ReconInfo::ShouldProvideDirection(Reconstructer_t r)
{
  switch(r) {
  case(kReconBONSAI):
  case(kReconTestVertices):
  case(kReconRandom):
    return true;
  break;
  default:
    return false;
  }
  return false;
}

void ReconInfo::Reset()
{
  fNRecons = 0;
  fFirstTime = +9E20;
  fLastTime = -9E20;
  //event
  fReconstructer.clear();
  fTriggerNum.clear();
  fNHits.clear();
  //vertex
  fTime.clear();
  fVertex.clear();
  fGoodnessOfFit.clear();
  fGoodnessOfTimeFit.clear();
  //direction
  fHasDirection.clear();
  fDirectionEuler.clear();
  fCherenkovCone.clear();
  fDirectionLikelihood.clear();
}

void ReconInfo::UpdateTimeBoundaries(double time)
{
  if(time < fFirstTime)
    fFirstTime = time;
  if(time > fLastTime)
    fLastTime = time;
}
