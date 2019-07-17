#ifndef DEMOSAICING_CUDA_CUH
#define DEMOSAICING_CUDA_CUH

#include <cuda_runtime.h>
#include <string>
#include <sstream>

using std::string;
using std::stringstream;

inline void _safe_cuda_call(cudaError_t err, const char* file, const int line, bool throwEx = true)
{
	if (err)
	{
		string error_str = cudaGetErrorString(err);
		stringstream ss;
		ss << "CUDA Error: " << error_str << "\nFile: " << file << "\nLine: " << line << "\n";
		string message = ss.str();

		if (throwEx)
			throw std::runtime_error(message);
	}
}

#define SAFE_CALL(err) _safe_cuda_call((err), __FILE__, __LINE__)

#endif