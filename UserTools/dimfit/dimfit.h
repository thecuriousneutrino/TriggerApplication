#ifndef dimfit_H
#define dimfit_H

#include <string>
#include <iostream>

#include "Tool.h"

class dimfit: public Tool {


 public:

  dimfit();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:

  ReconInfo * fInFilter;
  std::string fInputFilterName;

  std::vector<double> * fEventPos;
  double time_window_s;
  double time_window_ns;
  double time_window_step_s;
  double time_window_step_ns;
  int min_events;

  double fCentr[15];
  double fRot[9];
  double fRMean[5];
  int fDim;
  int fExitPoint;

  double R2MIN;
  double LOWDBIAS;
  double GOODPOINT;
  double MAXMEANPOS;

  int nclusters_silent_warning;
  int nclusters_normal_warning;
  int nclusters_golden_warning;

  int dimfit_(int n,double *points,double *centr,double *rot,double *rmean, int &dim,int &exitpoint, bool verbose);
  double d_pythag(double a,double b);
  int d_iszero(double *matrix,int sta);
  void setvec(double *vectors,short int vect,double val1,double val2,double val3);
  void rotate(double *vectors,short int vect,double si,double co);
  int planegivens(double *matrix,double *rot,double shift);
  void tridiag(double *matrix,double *rot);
  void d_swap(double *val,double *rot,int c1,int c2);
  void eigen(double *matrix,double *rot);

  int verbose;

  std::stringstream ss;

  void StreamToLog(int level) {
    Log(ss.str(), level, verbose);
    ss.str("");
  }

  enum LogLevel {FATAL=-1, ERROR=0, WARN=1, INFO=2, DEBUG1=3, DEBUG2=4, DEBUG3=5};


};


#endif
