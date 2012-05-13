#ifndef DEFINE_H
#define DEFINE_H

#ifdef _MSVC_

#ifdef Sophus_EXPORTS
#	define LIBRARY_API __declspec(dllexport)
#else
#	define LIBRARY_API __declspec(dllimport)
#endif

#define M_PI_2     1.57079632679489661923
#define isnan(x) _isnan(x) 

typedef unsigned int uint;


#endif //_MSVC_

#endif // DEFINE_H