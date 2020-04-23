#include "dimfit.h"

dimfit::dimfit():Tool(){}


bool dimfit::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  verbose = 0;
  m_variables.Get("verbose", verbose);

  m_data= &data;

  //parameters determining which events to read in
  if(!m_variables.Get("input_filter_name", fInputFilterName)) {
    Log("INFO: input_filter_name not given. Using ALL", WARN, verbose);
    fInputFilterName = "ALL";
  }
  fInFilter  = m_data->GetFilter(fInputFilterName, false);
  if(!fInFilter) {
    ss << "FATAL: no filter named " << fInputFilterName << " found. Returning false";
    StreamToLog(FATAL);
    return false;
  }

  //sliding time window parameters
  double time_window_s;
  if(!m_variables.Get("time_window", time_window_s)) {
    time_window_s = 20;
    Log("WARN: No time_window parameter found. Using a value of 20 (seconds)", WARN, verbose);
  }
  m_time_window = time_window_s * TimeDelta::s;
  double time_window_step_s;
  if(!m_variables.Get("time_window_step", time_window_step_s)) {
    time_window_step_s = 0.2;
    Log("WARN: No time_window_step parameter found. Using a value of 0.2 (seconds)", WARN, verbose);
  }
  m_time_window_step = time_window_step_s * TimeDelta::s;
  if(!m_variables.Get("min_events", min_events)) {
    min_events = 3;
    Log("WARN: No min_events parameter found. Using a value of 3", WARN, verbose);
  }

  //dimfit parameters
  if(!m_variables.Get("R2MIN", R2MIN)) {
    R2MIN = 300000;
    Log("WARN: No R2MIN parameter found. Using a value of 300000", WARN, verbose);
  }
  if(!m_variables.Get("LOWDBIAS", LOWDBIAS)) {
    LOWDBIAS = 1.15;
    Log("WARN: No LOWDBIAS parameter found. Using a value of 1.15", WARN, verbose);
  }
  if(!m_variables.Get("GOODPOINT", GOODPOINT)) {
    GOODPOINT = 40000;
    Log("WARN: No GOODPOINT parameter found. Using a value of 40000", WARN, verbose);
  }
  if(!m_variables.Get("MAXMEANPOS", MAXMEANPOS)) {
    MAXMEANPOS = 250000;
    Log("WARN: No MAXMEANPOS parameter found. Using a value of 250000", WARN, verbose);
  }

  //nclusters parameters
  if(!m_variables.Get("nclusters_silent_warning", nclusters_silent_warning)) {
    nclusters_silent_warning = 200;
    Log("WARN: No nclusters_silent_warning parameter found. Using a value of 200", WARN, verbose);
  }
  if(!m_variables.Get("nclusters_normal_warning", nclusters_normal_warning)) {
    nclusters_normal_warning = 435;
    Log("WARN: No nclusters_normal_warning parameter found. Using a value of 435", WARN, verbose);
  }
  if(!m_variables.Get("nclusters_golden_warning", nclusters_golden_warning)) {
    nclusters_golden_warning = 630;
    Log("WARN: No nclusters_golden_warning parameter found. Using a value of 630", WARN, verbose);
  }

  //allocate memory for a relatively large number of events
  const int n_event_max = 10000;
  fEventPos = new std::vector<double>(n_event_max * 3);

  return true;
}


bool dimfit::Execute(){

  const int N = fInFilter->GetNRecons();

  //get first/last times
  TimeDelta tstart = fInFilter->GetFirstTime();
  TimeDelta tend   = fInFilter->GetLastTime();
  ss << "DEBUG: dimfit looping in time from " << tstart << " to " << tend << " in steps of " << m_time_window_step;
  StreamToLog(DEBUG1);

  //use a sliding window to loop over the events
  TimeDelta tloop = tstart;
  TimeDelta tloopend = tloop + m_time_window;
  tloopend = tloopend < tend ? tloopend : tend; //ensure the loop runs at least once
  while(tloopend <= tend) {
    fEventPos->clear();

    unsigned int nclusters = 0;
    for(int irecon = 0; irecon < N; irecon++) {
      //skip events reconstructed outside the time window
      TimeDelta t = fInFilter->GetTime(irecon);
      if(t < tloop || t > tloopend)
	continue;

      //get the vertex position
      Pos3D pos = fInFilter->GetVertex(irecon);
      
      //we'll use this event! Put it in the vertex array
      fEventPos->push_back(pos.x);
      fEventPos->push_back(pos.y);
      fEventPos->push_back(pos.z);
      nclusters++;

      ss << "DEBUG: Adding vertex " << pos.x << ", " << pos.y << "\t" << pos.z << " to run through dimfit";
      StreamToLog(DEBUG3);
    }//irecon

    //only call dimfit if there are over (or equal to) the minimum number of vertices
    if(nclusters >= min_events) {
      ss << "DEBUG: Running " << nclusters << " event positions through dimfit";
      StreamToLog(DEBUG1);

      dimfit_(nclusters, fEventPos->data(), fCentr, fRot, fRMean, fDim, fExitPoint, verbose);

      ss << "INFO: Dimfit returns " << fDim << " Exited at " << fExitPoint;
      StreamToLog(INFO);
    }

    //compare nclusters to nclusters warning thresholds
    NClustersWarning_t nclusters_warning = kNClustersUndefined;
    if     (nclusters > nclusters_golden_warning) nclusters_warning = kNClustersGolden;
    else if(nclusters > nclusters_normal_warning) nclusters_warning = kNClustersNormal;
    else if(nclusters > nclusters_silent_warning) nclusters_warning = kNClustersSilent;
    else if(nclusters)                            nclusters_warning = kNClustersStandard;

    ss << "INFO: nclusters_warning = " << ReconInfo::EnumAsString(nclusters_warning) << ", nclusters = " << nclusters << " in " << m_time_window;
    StreamToLog(INFO);
    
    SNWarningParams supernova_warning_parameters(nclusters,fDim,nclusters_warning);
    m_data->SupernovaWarningParameters.push_back(supernova_warning_parameters);

    //increment the sliding time window
    tloop += m_time_window_step;
    tloopend = tloop + m_time_window;

  }//while(tloop < tend)

  return true;
}


bool dimfit::Finalise(){

  delete fEventPos;

  return true;
}


/**************************************************
The following functions are taken verbatim from SK.
It is what SK use for SN monitoring (up to at least 2019)
https://arxiv.org/abs/1601.04778
Credit for the code (and allowing us to use it) go to Michael Smy
**************************************************/

//this is the main SK function. From what I understand
//INPUTS
// n           = the number of vertices
// points[n*3] = vector of the points. In the order [x1,y1,z1,x2,y2,z2,...]
//OUTPUTS
// centr[15] = [0]-[2] have centroids
//             [3]-[8] have the covariance matix
//             [9]-[14] have the rotated matix (guess: this is approximately diagonal), with the eigenvalues in [9]-[11]
// rot[9]    = the rotation matix (I presume used to go from covariance matrix to rotated matrix
// rmean[5]  = [0] stores the sqrt of the chosen reduced chisq (i.e. root3 for d=2)
//             [1]-[3] store the reduced chisq (not sqrt!) of root, root2, root3 
//                  note this is code convention. SK technote convention is lambda1, 1/2(lambda1 + lambda2), 1/3(lambda1 + lambda2 + lambda3)
//                  These are what's plotted in fig 3 of the SK paper
//             [4] mean position
// dim       = the D value the algoithm thinks it is
//RETURN VALUE
// -1 if input n < 1
// dim otherwise

int dimfit::dimfit_(int n,double *points,double *centr,double *rot,double *rmean, int &dim,int &exitpoint, bool verbose)
{
  double *matrix;
  double x,y,z,root,root2,root3;
  int    i;

  for(i=0; i<12; i++) centr[i]=0;
  if (n<1) {
    exitpoint=-1;
    return -1;
  }
  //matrix[0] is centr[3]
  matrix=centr+3;
  for(i=0; i<n; i++)
    {
      x=*points++; y=*points++; z=*points++;
      // printf("in dimfit position %f %f %f\n", x, y, z );
      *centr+=x;
      centr[1]+=y;
      centr[2]+=z;
      centr[3]+=x*x;
      centr[4]+=y*y;
      centr[5]+=z*z;
      centr[6]+=x*y;
      centr[7]+=y*z;
      centr[8]+=x*z;
    }
  for(i=0; i<9; i++) centr[i]/=n;
  //centr[0] to centr[8] have the average values of x, y, z, xx, yy, zz, xy, yz, xz
  for(i=0; i<3; i++) matrix[i]-=centr[i]*centr[i];
  //centr[3] to centr[5] have xx/n - (x/n)^2, yy/n - (y/n)^2, zz/n - (z/n)^2
  matrix[3]-=*centr*centr[1];
  //centr[6] has xy/n - (x/n)(y/n)
  matrix[4]-=centr[1]*centr[2];
  //centr[7] has yz/n - (y/n)(z/n)
  matrix[5]-=*centr*centr[2];
  //centr[8] has xz/n - (x/n)(z/n)
  for(i=0; i<6; i++)
    matrix[i+6]=matrix[i];
  //centr[9] has centr[3], centr[10] has centr[4], ..., centr[14] has centr[8]

  //if only 1 vertex, assume a point source
  if (n==1)
    {
      rmean[0]=0;
      dim=0;
      exitpoint = 1;
      return(0); // assume point source
    }

  //call the eigen function with centr[9]
  //I don't know the maths of this, but you should look into it.
  //From what I can tell (i.e. what's used next) centr[9] to centr[11] have been
  // modified to the eigenvalues
  //And rot is also set. I'm guessing it's a rotation matrix
  //The printout statements here should help you
  eigen(matrix+6,rot);
  if(verbose) {
    printf("centroid:   %lf %lf %lf\n",centr[0],centr[1],centr[2]);
    printf("covariance: %lf %lf %lf\n",centr[3],centr[6],centr[8]);
    printf("covariance: %lf %lf %lf\n",centr[6],centr[4],centr[7]);
    printf("covariance: %lf %lf %lf\n",centr[8],centr[7],centr[5]);
    printf("rotated:    %lf %lf %lf\n",centr[9],centr[12],centr[14]);
    printf("rotated:    %lf %lf %lf\n",centr[12],centr[10],centr[13]);
    printf("rotated:    %lf %lf %lf\n",centr[14],centr[13],centr[11]);
    printf("rotation:   %lf %lf %lf\n",rot[0],rot[1],rot[2]);
    printf("rotation:   %lf %lf %lf\n",rot[3],rot[4],rot[5]);
    printf("rotation:   %lf %lf %lf\n",rot[6],rot[7],rot[8]);
  }

  //the root values here are the reduced chisq from the technote.
  root=n*(centr[9]+centr[10]+centr[11])/(3*(n-1));
  //std::cout << root << std::endl;
  root2=n*(centr[9]+centr[10])/(2*(n-2));
  if (n>3) root3=n*centr[9]/(n-3); else root3=1e10;
  double Pos = centr[0]*centr[0]+centr[1]*centr[1]+centr[2]*centr[2];
  if(verbose)
    std::cout << "dimfit    av x,y,z\t" << centr[0] << "\t" << centr[1] << "\t" << centr[2] << std::endl;
  rmean[1] = root;
  rmean[2] = root2;
  rmean[3] = root3;
  rmean[4] = Pos;
  if(verbose) {
    printf("3 values %lf %lf %lf\n",sqrt(root),sqrt(root2),sqrt(root3));
    printf("Mean position %lf\n", Pos);
  }

  //Here the decisions are made as to how many dimensions the vertex distribution is
  if ((n==2) || root<GOODPOINT)
    {
      rmean[0]=sqrt(root);
      dim=0;
      exitpoint = 2;
      return(0); // it is from a point source!
    }
  if ((n>3) && (root3*LOWDBIAS<root2) && (root3*LOWDBIAS<root))
    {
      if ((root3>R2MIN) && 
	  (Pos < MAXMEANPOS))
	{
	  rmean[0]=sqrt(root);
          dim=3;
	  exitpoint = 3;
	  return(3); // it is a uniform volume source!
	}
      else
	{
	  rmean[0]=sqrt(root3);
          dim=2;
	  exitpoint = 4;
	  return(2); // it is an area source!
	}
    }
  if (root2*LOWDBIAS<root)
    {
      if ((root2>R2MIN) && 
	  (Pos < MAXMEANPOS))
	{
	  rmean[0]=sqrt(root);
          dim=3;
	  exitpoint = 5;
	  return(3); // it is a uniform volume source!
	}
      else
	{
	  rmean[0]=sqrt(root2);
          dim=1;
	  exitpoint = 6;
	  return(1); // it is a line source
	}
    }
  rmean[0]=sqrt(root);

  if( (root > R2MIN ) && 
      (Pos < MAXMEANPOS)){
    dim=3;
    exitpoint = 7;
    return(3); // it is a uniform volume source!
  }
  dim=0;
  exitpoint = 8;
  return(0);   // it is a point source!
}

/* add in quadrature */
double dimfit::d_pythag(double a,double b)
{
  double ma=fabs(a),mb=fabs(b),rat;

  if (ma>mb)
    {
      rat=mb/ma;
      return(ma*sqrt(1+rat*rat));
    }
  if (mb==0) return(0);
  rat=ma/mb;
  return(mb*sqrt(1+rat*rat));
}

/* test zero offdiagonal element */
int dimfit::d_iszero(double *matrix,int sta)
{
  double add=fabs(matrix[sta+1])+fabs(matrix[sta]);
  return(add+fabs(matrix[sta+3])==add);
}

void dimfit::setvec(double *vectors,short int vect,double val1,double val2,double val3)
{
  vectors[vect]=val1;
  vectors[vect+3]=val2;
  vectors[vect+6]=val3;
}

void dimfit::rotate(double *vectors,short int vect,double si,double co)
{
  double save;

  /* correct trafo matrix */
  save=vectors[vect+1];
  vectors[vect+1]=si*vectors[vect]+co*save;
  vectors[vect]  =co*vectors[vect]-si*save;
  save=vectors[vect+4];
  vectors[vect+4]=si*vectors[vect+3]+co*save;
  vectors[vect+3]=co*vectors[vect+3]-si*save;
  save=vectors[vect+7];
  vectors[vect+7]=si*vectors[vect+6]+co*save;
  vectors[vect+6]=co*vectors[vect+6]-si*save;
}

/* first a plane rotation, then a Givens rotation */
int dimfit::planegivens(double *matrix,double *rot,double shift)
{
  double si,co,sub,siodiag,coodiag,denom,pkiip1;
  int    ii,iii;

  shift+=matrix[2];
  si=co=1;
  sub=0;
  for(ii=1; ii>=0; ii--)
    {
      siodiag=si*matrix[ii+3];
      coodiag=co*matrix[ii+3];
      denom=d_pythag(siodiag,shift);
      matrix[ii+4]=denom;
      if (denom==0)
	{
	  matrix[ii+1]-=sub;
	  matrix[5]=0;
	  return(-1);
	}
      si=siodiag/denom; /* do rotation */
      co=shift/denom;
      shift=matrix[ii+1]-sub;
      denom=(matrix[ii]-shift)*si+2*co*coodiag;
      sub=si*denom;
      matrix[ii+1]=shift+sub;
      shift=co*denom-coodiag;
      rotate(rot,ii,si,co);
    }
  matrix[0]-=sub;
  matrix[3]=shift;
  matrix[5]=0;
  return(0);
}

/* tridiagonalize matrix with a Householder transformation */
void dimfit::tridiag(double *matrix,double *rot)
{
  double n=d_pythag(matrix[3],matrix[5]);
  double u1=matrix[3]-n,u2=matrix[5],u=u1*u1+u2*u2;
  double h,p1,p2;

  if ((n==0) || (u==0))  return;

  h=2/u;
  p1=h*(matrix[1]*u1+matrix[4]*u2);
  p2=h*(matrix[4]*u1+matrix[2]*u2);

  setvec(rot,0,1,0,        0);
  setvec(rot,1,0,1-u1*u1*h, -u1*u2*h);
  setvec(rot,2,0, -u1*u2*h,1-u2*u2*h);
  h*=0.5*(u1*p1+u2*p2);
  p1-=h*u1;
  p2-=h*u2;

  matrix[1]-=2*p1*u1;  matrix[4]-=p1*u2+p2*u1;  matrix[5]=0;
                       matrix[2]-=2*p2*u2;
                                                matrix[3]=n;
}

/* swap eigenvalues and eigenvectors */
void dimfit::d_swap(double *val,double *rot,int c1,int c2)
{
  double save;

  save=val[c1];   val[c1]  =val[c2];   val[c2]=  save;
  save=rot[c1];   rot[c1]  =rot[c2];   rot[c2]=  save;
  save=rot[c1+3]; rot[c1+3]=rot[c2+3]; rot[c2+3]=save;
  save=rot[c1+6]; rot[c1+6]=rot[c2+6]; rot[c2+6]=save;
}

void dimfit::eigen(double *matrix,double *rot)
{
  double rat,root,shift,add,add2,si,co,u2i;
  int   sta,iter;

  tridiag(matrix,rot); //tridiagonalize
  for(sta=0; sta<2; sta++)
    for(iter=0; iter<20; iter++)
      {
	if (d_iszero(matrix,sta)) break;
	rat=(matrix[sta+1]-matrix[sta])/(2*matrix[sta+3]);
	root=d_pythag(1,rat);
	if (rat*root<0) root=-root;
	root+=rat;
	add=matrix[sta+3]/root;
	/* in case a simple plane rotation is needed */
	if ((sta==1) || d_iszero(matrix,sta+1))
	  {
	    co=root/d_pythag(1,root);
	    si=co/root;
	    matrix[sta]-=add;
	    matrix[sta+1]+=add;
	    matrix[sta+3]=0;
	    rotate(rot,sta,si,co);
	    if (matrix[2]<matrix[1])
	      d_swap(matrix,rot,2,1);
	    if (matrix[1]<matrix[0])
	      d_swap(matrix,rot,0,1);
	    if (matrix[2]<matrix[1])
	      d_swap(matrix,rot,2,1);
	    return;
	  }
	/* otherwise do a QL algorithm with implicit shift */
	planegivens(matrix,rot,add-matrix[sta]);
      }
  if (matrix[2]<matrix[1]) d_swap(matrix,rot,2,1);
  if (matrix[1]<matrix[0]) d_swap(matrix,rot,0,1);
  if (matrix[2]<matrix[1]) d_swap(matrix,rot,2,1);
  return;
}
