#ifndef RECONINFO_H
#define RECONINFO_H

#include <iostream>
#include <vector>
#include <cmath>

typedef enum EReconstructers {
  kReconUndefined = -1,
  kReconBONSAI,
  kReconTestVerticesNoDirection,
  kReconTestVertices,
  kReconRandomNoDirection,
  kReconRandom //ensure this stays at the end, for looping purposes
} Reconstructer_t;

typedef enum NClustersWarnings {
  kNClustersUndefined = -1,
  kNClustersStandard,
  kNClustersSilent,
  kNClustersNormal,
  kNClustersGolden //ensure this stays at the end, for looping purposes
} NClustersWarning_t;

typedef enum SNWarnings {
  kSNWarningUndefined = -1,
  kSNWarningStandard,
  kSNWarningSilent,
  kSNWarningNormal,
  kSNWarningGolden //ensure this stays at the end, for looping purposes
} SNWarning_t;

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

struct SNWarningParams
{
  int m_dim, m_nclusters;
  NClustersWarning_t m_nclusters_warning;
  SNWarningParams(int nclusters,int dim, NClustersWarning_t nclusters_warning){m_nclusters = nclusters; m_dim = dim; m_nclusters_warning = nclusters_warning;};
};

class ReconInfo
{
 public:
  ReconInfo();

  void AddRecon(Reconstructer_t reconstructer, int trigger_num, int nhits, double time, double * vertex, double goodness_of_fit, double goodness_of_time_fit, bool fill_has_direction = true);
 
  void AddRecon(Reconstructer_t reconstructer, int trigger_num, int nhits, double time, double * vertex, double goodness_of_fit, double goodness_of_time_fit,
		double * direction_euler, double * cherenkov_cone, double direction_likelihood);

  void AddReconFrom(ReconInfo * in, const int irecon);

  static std::string EnumAsString(Reconstructer_t r);
  
  static Reconstructer_t ReconstructerFromString(std::string s);

  static std::string EnumAsString(NClustersWarning_t w);

  static NClustersWarning_t NClustersWarningFromString(std::string s);

  static std::string EnumAsString(SNWarning_t w);

  static SNWarning_t SNWarningFromString(std::string s);

  static bool ShouldProvideDirection(Reconstructer_t r);

  int             GetNRecons  () { return fNRecons;   }
  double          GetFirstTime() { return fFirstTime; }
  double          GetLastTime () { return fLastTime;  }
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
  
  void Reset();

 private:

  //collection
  int    fNRecons;
  double fFirstTime;
  double fLastTime;

  //event
  std::vector<Reconstructer_t> fReconstructer;
  std::vector<int>             fTriggerNum;
  std::vector<int>             fNHits;

  //vertex
  std::vector<double>          fTime;
  std::vector<Pos3D>           fVertex;
  std::vector<double>          fGoodnessOfFit;
  std::vector<double>          fGoodnessOfTimeFit;

  //direction
  std::vector<bool>            fHasDirection;
  std::vector<DirectionEuler>  fDirectionEuler;
  std::vector<CherenkovCone>   fCherenkovCone;
  std::vector<double>          fDirectionLikelihood;

  void UpdateTimeBoundaries(double time);

};

#endif //RECONINFO_H
