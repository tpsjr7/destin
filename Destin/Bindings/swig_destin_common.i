
%module(directors="1") DESTIN_MODULE 
%{
/* includes that are needed to compile */
#include "macros.h"
#include "DestinIterationFinishedCallback.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
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
#include "CztMod.h"
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
