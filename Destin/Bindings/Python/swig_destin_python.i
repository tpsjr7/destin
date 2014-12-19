%{
#define SWIG_FILE_WITH_INIT
%}
%include "pylist.i"

/* see numpy info at http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html# */
%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1, float* ARGOUT_ARRAY1) {(int size, float* output)}; 

#define SWIG_MODULE_NAME pydestin
%include "../swig_destin_common.i"
