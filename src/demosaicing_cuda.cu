#include "demosaicing.hpp"
#include "demosaicing_cuda.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

texture<unsigned char, cudaTextureType2D> tex8u_cfa;



template<BayerPattern pattern> __device__ inline bool isColorBlue(bool xEven, bool yEven)
{
	if (pattern == BAYER_BGGR)
		return (xEven && yEven);
	else if (pattern == BAYER_RGGB)
		return (!xEven && !yEven);
	else if (pattern == BAYER_GBRG)
		return (!xEven && yEven);
	else if (pattern == BAYER_GRBG)
		return (xEven && !yEven);
	else
		return false; //never going to happen :)
}

template<BayerPattern pattern> __device__ inline bool isColorRed(bool xEven, bool yEven)
{
	if (pattern == BAYER_BGGR)
		return (!xEven && !yEven);
	else if (pattern == BAYER_RGGB)
		return (xEven && yEven);
	else if (pattern == BAYER_GBRG)
		return (xEven && !yEven);
	else if (pattern == BAYER_GRBG)
		return (!xEven && yEven);
	else
		return false; //never going to happen :)
}

template<BayerPattern pattern> __device__ inline bool isColorGreenInRedRow(bool xEven, bool yEven)
{
	if (pattern == BAYER_BGGR)
		return (xEven && !yEven);
	else if (pattern == BAYER_RGGB)
		return (!xEven && yEven);
	else if (pattern == BAYER_GBRG)
		return (!xEven && !yEven);
	else if (pattern == BAYER_GRBG)
		return (xEven && yEven);
	else
		return false; //never going to happen :)
}

template<BayerPattern pattern>
__device__ void bayer_to_bgr(int x, int y, unsigned char& r, unsigned char& g, unsigned char& b)
{
	bool xEven = (x & 1) == 0;
	bool yEven = (y & 1) == 0;

	//Determine if the current color is Red, Green or Blue according to a specific Bayer pattern.
	bool isBlue = isColorBlue<pattern>(xEven, yEven);
	bool isRed = isColorRed<pattern>(xEven, yEven);
	bool isGreenInRedRow = isColorGreenInRedRow<pattern>(xEven, yEven);

	//Read current and neighboring pixels
	unsigned char current = tex2D(tex8u_cfa, x, y);
	float up = tex2D(tex8u_cfa, x, y - 1);
	float down = tex2D(tex8u_cfa, x, y + 1);
	float left = tex2D(tex8u_cfa, x - 1, y);
	float right = tex2D(tex8u_cfa, x + 1, y);

	if (isRed || isBlue)
	{
		//Read diagonal neighbors
		float up_left = tex2D(tex8u_cfa, x - 1, y - 1);
		float up_right = tex2D(tex8u_cfa, x + 1, y - 1);
		float down_left = tex2D(tex8u_cfa, x - 1, y + 1);
		float down_right = tex2D(tex8u_cfa, x + 1, y + 1);

		g = static_cast<unsigned char>( (up + down + left + right) / 4.0f); //Average of neighbours

		if (isRed)
		{
			r = current;
			b = static_cast<unsigned char>( (up_left + up_right + down_left + down_right) / 4.0f );	//Average of diagonal neighbours
		}
		else
		{
			b = current;
			r = static_cast<unsigned char>( (up_left + up_right + down_left + down_right) / 4.0f ); //Average of diagonal neighbours
		}
	}
	else
	{
		g = current;

		if (isGreenInRedRow)
		{
			r = static_cast<unsigned char>( (left + right) / 2.0f );
			b = static_cast<unsigned char>( (up + down) / 2.0f);
		}
		else
		{
			b = static_cast<unsigned char>((left + right) / 2.0f);
			r = static_cast<unsigned char>((up + down) / 2.0f);
		}
	}
}


template<BayerPattern pattern>
__global__ void kernel_demosaic_bilinear(unsigned char* output, int width, int height, int rgb_pitch)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		const int tid = y * rgb_pitch + (3 * x);
		unsigned char r, g, b;

		bayer_to_bgr<pattern>(x, y, r, g, b);
		
		output[tid] = b;
		output[tid + 1] = g;
		output[tid + 2] = r;
	}
}


void demosaic_bilinear_8u_device(const unsigned char* input, unsigned char* output, int width, int height, int cfa_pitch, int rgb_pitch, BayerPattern pattern)
{
	//Specify block and grid size
	dim3 block(16, 16);
	dim3 grid;
	grid.x = (width + block.x - 1) / block.x;
	grid.y = (height + block.y - 1) / block.y;

	//Bind input image to CUDA texture
	size_t offset;
	SAFE_CALL( cudaBindTexture2D(&offset, tex8u_cfa, input, tex8u_cfa.channelDesc, width, height, cfa_pitch) );

	//Call kernel
	switch (pattern)
	{
	case BAYER_BGGR:
		kernel_demosaic_bilinear <BAYER_BGGR> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	case BAYER_RGGB:
		kernel_demosaic_bilinear <BAYER_RGGB> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	case BAYER_GBRG:
		kernel_demosaic_bilinear <BAYER_GBRG> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	case BAYER_GRBG:
		kernel_demosaic_bilinear <BAYER_GRBG> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	default:
		kernel_demosaic_bilinear <BAYER_BGGR> << < grid, block >> > (output, width, height, rgb_pitch);
	}

	//Unbind texture
	SAFE_CALL(cudaUnbindTexture(tex8u_cfa));
}

void demosaic_bilinear_8u_device(const unsigned char* input, unsigned char* output, int width, int height, int cfa_pitch, int rgb_pitch, BayerPattern pattern, float& milliseconds)
{
	//Specify block and grid size
	dim3 block(16, 16);
	dim3 grid;
	grid.x = (width + block.x - 1) / block.x;
	grid.y = (height + block.y - 1) / block.y;

	static cudaEvent_t start = NULL, stop = NULL;
	
	if (start == NULL)
	{
		SAFE_CALL(cudaEventCreate(&start));
		SAFE_CALL(cudaEventCreate(&stop));
	}

	//Bind input image to CUDA texture
	size_t offset;
	SAFE_CALL(cudaBindTexture2D(&offset, tex8u_cfa, input, tex8u_cfa.channelDesc, width, height, cfa_pitch));

	cudaEventRecord(start);

	//Call kernel
	switch (pattern)
	{
	case BAYER_BGGR:
		kernel_demosaic_bilinear <BAYER_BGGR> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	case BAYER_RGGB:
		kernel_demosaic_bilinear <BAYER_RGGB> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	case BAYER_GBRG:
		kernel_demosaic_bilinear <BAYER_GBRG> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	case BAYER_GRBG:
		kernel_demosaic_bilinear <BAYER_GRBG> << < grid, block >> > (output, width, height, rgb_pitch);
		break;
	default:
		kernel_demosaic_bilinear <BAYER_BGGR> << < grid, block >> > (output, width, height, rgb_pitch);
	}

	SAFE_CALL(cudaEventRecord(stop));
	SAFE_CALL(cudaEventSynchronize(stop));
	SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

	//Unbind texture
	SAFE_CALL(cudaUnbindTexture(tex8u_cfa));
}