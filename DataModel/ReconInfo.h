#ifndef RECONINFO_H
#define RECONINFO_H

#include <iostream>
#include <vector>

typedef enum EReconstructers {
  kReconUndefined = -1,
  kReconBONSAI,
  kReconTestVerticesNoDirection,
  kReconTestVertices,
  kReconRandomNoDirection,
  kReconRandom //ensure this stays at the end, for looping purposes
} Reconstructer_t;

struct Pos3D
{
  double x, y, z;
  double R() { return sqrt(x*x + y*y + z*z); }
};

struct DirectionEuler
{
  double theta, phi, alpha;
};

struct CherenkovCone
{
  double cos_angle, ellipticity;
};

class ReconInfo
{
 public:
   ReconInfo() : fNRecons(0) {}

  void AddRecon(Reconstructer_t reconstructer, int trigger_num, int nhits, double time, double * vertex, double goodness_of_fit, double goodness_of_time_fit, bool fill_has_direction = true) {
    if(fill_has_direction) {
      if(ShouldProvideDirection(reconstructer)) {
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
  }
 
  void AddRecon(Reconstructer_t reconstructer, int trigger_num, int nhits, double time, double * vertex, double goodness_of_fit, double goodness_of_time_fit,
		double * direction_euler, double * cherenkov_cone, double direction_likelihood) {
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

  static std::string EnumAsString(Reconstructer_t r) {
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
  
  static Reconstructer_t ReconstructerFromString(std::string s) {
    for(int i = int(kReconUndefined)+1; i <= kReconRandom; i++) {
      if(s.compare(ReconInfo::EnumAsString((Reconstructer_t)i)) == 0) {
	return (Reconstructer_t)i;
      }
    }
    std::cerr << "ReconInfo::ReconstructerFromString() Unknown string value " << s << std::endl;
    return kReconUndefined;
  }

  bool ShouldProvideDirection(Reconstructer_t r) {
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

  int             GetNRecons() { return fNRecons; }
  Reconstructer_t GetReconstructer    (int irecon) { return fReconstructer[irecon]; }
  int             GetTriggerNum       (int irecon) { return fTriggerNum[irecon]; }
  int             GetNHits            (int irecon) { return fNHits[irecon]; }
  double          GetTime             (int irecon) { return fTime[irecon]; }
  Pos3D           GetVertex           (int irecon) { return fVertex[irecon]; }
  double          GetGoodnessOfFit    (int irecon) { return fGoodnessOfFit[irecon]; }
  double          GetGoodnessOfTimeFit(int irecon) { return fGoodnessOfTimeFit[irecon]; }
  //direction
  bool            GetHasDirection       (int irecon) { return fHasDirection[irecon]; }
  DirectionEuler  GetDirectionEuler     (int irecon) { return fDirectionEuler[irecon]; }
  CherenkovCone   GetCherenkovCone      (int irecon) { return fCherenkovCone[irecon]; }
  double          GetDirectionLikelihood(int irecon) { return fDirectionLikelihood[irecon]; }
  
  void Reset() {
    fNRecons = 0;
    fReconstructer.clear();
    fTriggerNum.clear();
    fNHits.clear();
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

 private:
  int fNRecons;
  std::vector<Reconstructer_t> fReconstructer;
  std::vector<int>             fTriggerNum;
  std::vector<int>             fNHits;
  std::vector<double>          fTime;
  std::vector<Pos3D>           fVertex;
  std::vector<double>          fGoodnessOfFit;
  std::vector<double>          fGoodnessOfTimeFit;
  //direction
  std::vector<bool>            fHasDirection;
  std::vector<DirectionEuler>  fDirectionEuler;
  std::vector<CherenkovCone>   fCherenkovCone;
  std::vector<double>          fDirectionLikelihood;

};

#endif //RECONINFO_H
