
%module(directors="1") SWIG_MODULE_NAME
%{
/* includes that are needed to compile */
#include "macros.h"
#include "DestinIterationFinishedCallback.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/ml/ml.hpp"
#include "VideoSource.h"
#include "VideoWriter.h"
#include "Transporter.h"
#include "INetwork.h"
#include "DestinNetworkAlt.h"
#include "ImageSourceBase.h"
#include "CifarSource.h"
#include "ImageSourceImpl.h"
#include "ISom.h"
#include "ClusterSom.h"
#include "SomPresentor.h"
#include "BeliefExporter.h"
#include "DestinTreeManager.h"
#include "CMOrderedTreeMinerWrapper.h"
#include "CztMod.h"

void dumpParams(CvANN_MLP_TrainParams & params){
    cout << "scale: " << params.bp_dw_scale << endl;

    cout << "scale: " << params.bp_moment_scale << endl;
    cout << "rp_dw0: " << params.rp_dw0 << endl;
    cout << "rp_dw_max: " << params.rp_dw_max << endl;
    cout << "rp_dw_min: " << params.rp_dw_min << endl;
    cout << "rp_dw_minus: " << params.rp_dw_minus << endl;
    cout << "rp_dw_plus: " << params.rp_dw_plus << endl;
    cout << "term_crit.epsilon " << params.term_crit.epsilon << endl;
    cout << "term_crit.max_iter: " << params.term_crit.max_iter << endl;
    cout << "term_crit.type: " << params.term_crit.type << endl;
    cout << "train_method: " << params.train_method << endl;
    return;
}

cv::Mat float32PointerToMat(float * pointer, uint length){
	cv::Mat out(1, length, CV_32F, (void*)pointer);
	return out;
}
%}


%include "macros.h"
typedef unsigned int uint;

/* Lets you use script strings easily with c++ strings */
%include "std_string.i"

/* 
turn on director wrapping callback, so c++ code can call methods defined in the target language
See http://www.swig.org/Doc2.0/SWIGDocumentation.html#Java_directors
See https://swig.svn.sourceforge.net/svnroot/swig/trunk/Examples/java/callback/
*/
%feature("director") DestinIterationFinishedCallback;
%include "DestinIterationFinishedCallback.h"


/* be able to use INetwork as an abstract interface in Java */
%feature("director") INetwork; 
%include "INetwork.h"


/* the other classes to generate wrappers for */
%include "destin.h"
%include "node.h"
%include "VideoSource.h"
%include "VideoWriter.h"
%include "Transporter.h"
%include "DestinNetworkAlt.h"
%include "learn_strats.h"
%include "ImageSourceBase.h"
%include "CifarSource.h"
%include "ImageSourceImpl.h"
%include "ISom.h"
%include "ClusterSom.h"
%include "SomPresentor.h"
%include "BeliefExporter.h"
%include "cent_image_gen.h"
%include "belief_transform.h"
%include "DestinTreeManager.h"
%include "CztMod.h"
%include "CMOrderedTreeMinerWrapper.h"

/* use c++ vector like a python list */
%include "std_vector.i"
namespace std {
%template(IntVector) vector<int>;
%template(ShortVector) vector<short>;
%template(FloatVector) vector<float>;
}


/* carrays.i so you can use a c++ pointer like an array */
%include "carrays.i" 
%array_class(int, SWIG_IntArray);
%array_class(float, SWIG_FloatArray);
%array_functions(float *, SWIG_Float_p_Array);
%array_class(uint, SWIG_UInt_Array);
%array_functions(Node *, SWIG_Node_p_Array);
%array_class(Node, SWIG_NodeArray);

/* some opencv functions */
namespace cv {
class Mat;
void imshow( const string& winname, const Mat& mat );
int waitKey(int delay=0);
}
typedef struct CvPoint
{
      int x;
      int y;
}
CvPoint;

/* CvTermCriteria borrowed from types_c.h */
#define CV_TERMCRIT_ITER    1
#define CV_TERMCRIT_NUMBER  CV_TERMCRIT_ITER
#define CV_TERMCRIT_EPS     2

typedef struct CvTermCriteria
{
    int    type;  /* may be combination of
                     CV_TERMCRIT_ITER
                     CV_TERMCRIT_EPS */
    int    max_iter;
    double epsilon;
}
CvTermCriteria;

CvTermCriteria  cvTermCriteria( int type, int max_iter, double epsilon )
{
    CvTermCriteria t;

    t.type = type;
    t.max_iter = max_iter;
    t.epsilon = (float)epsilon;

    return t;
}

/* borrowed from ml.hpp of opencv */
#define CV_PROP_RW
struct CvANN_MLP_TrainParams
{
    CvANN_MLP_TrainParams();
    CvANN_MLP_TrainParams( CvTermCriteria term_crit, int train_method,
                           double param1, double param2=0 );
    ~CvANN_MLP_TrainParams();

    enum { BACKPROP=0, RPROP=1 };

    CV_PROP_RW CvTermCriteria term_crit;
    CV_PROP_RW int train_method;

    // backpropagation parameters
    CV_PROP_RW double bp_dw_scale, bp_moment_scale;

    // rprop parameters
    CV_PROP_RW double rp_dw0, rp_dw_plus, rp_dw_minus, rp_dw_min, rp_dw_max;
};

void dumpParams(CvANN_MLP_TrainParams & params);
cv::Mat float32PointerToMat(float * pointer, uint length);
