#include "demosaicing.hpp"
#include "demosaicing_cuda.cuh"
#include <opencv2/opencv.hpp>
#include <chrono>


//*********************************************Device Function Declarations************************************************//
void demosaic_bilinear_8u_device(const unsigned char* input, unsigned char* output, int width, int height, int cfa_pitch, int rgb_pitch, BayerPattern pattern);
void demosaic_bilinear_8u_device(const unsigned char* input, unsigned char* output, int width, int height, int cfa_pitch, int rgb_pitch, BayerPattern pattern, float& milliseconds);


struct App::AppImpl
{
	/* Single image parameters */
	string inputPath, outputPath;
	cv::Mat input;
	cv::Mat output;
	bool isImage;

	/* Video parameters */
	cv::VideoCapture capture;
	cv::VideoWriter writer;
	bool isVideoFile;
	bool isVideoLive;
	int camera_id;

	string windowTitle;

	bool showOutput;
	bool writeOutput;

	/************Device variables*******************/
	unsigned char *dInput, *dOutput;
	int dInputPitch, dOutputPitch;

	BayerPattern pattern;
};

App::App() : impl(nullptr)
{
	impl = new AppImpl;
}

App::~App()
{
	delete impl;
	impl = nullptr;
}

BayerPattern str2bayer(std::string str)
{
	if (str == "BGGR")	return BAYER_BGGR;
	else if (str == "RGGB")	return BAYER_RGGB;
	else if (str == "GBRG")	return BAYER_GBRG;
	else if (str == "GRBG")	return BAYER_GRBG;
	else throw std::runtime_error("Invaid Bayer Pattern!");
}

void App::parse_arguments(int argc, char** argv)
{
	if (argc < 4 || argc > 5)
	{
		throw std::runtime_error("Invalid Number of Arguments!");
	}

	string flag = argv[1];
	impl->isVideoFile = (flag == "-v");
	impl->isVideoLive = (flag == "-c");
	impl->isImage = (flag == "-i");

	if (!(impl->isVideoFile || impl->isVideoLive || impl->isImage))
	{
		throw std::runtime_error("Invalid input flag!");
	}

	if (argc == 4)
	{
		impl->pattern = str2bayer(argv[2]);

		if (impl->isVideoLive)
			impl->camera_id = std::atoi(argv[3]);
		else
			impl->inputPath = argv[3];

		impl->showOutput = true;
		impl->writeOutput = false;
		impl->windowTitle = "Output";
	}

	if (argc == 5)
	{
		impl->pattern = str2bayer(argv[2]);

		if (impl->isVideoLive)
			impl->camera_id = std::atoi(argv[3]);
		else
			impl->inputPath = argv[3];
		
		impl->outputPath = argv[4];
		impl->showOutput = true;
		impl->writeOutput = true;
	}
}

void App::usage()
{
	printf("Usage:\ndemosaicing <Input Source Flag> <Bayer Pattern> <Input Image Path> <Output Image Path (optional)>\n\nExample:\n\ndemosaicing -i BGGR input.tif\ndemosaicing -i BGGR input.tif output.tif\ndemosaicing -v BGGR input.mp4\n");
}

template<bool video>
void App::process()
{
	if (video)
	{
		int width = static_cast<int>(impl->capture.get(cv::CAP_PROP_FRAME_WIDTH));
		int height = static_cast<int>(impl->capture.get(cv::CAP_PROP_FRAME_HEIGHT));

		demosaic_bilinear_8u_device(impl->dInput, impl->dOutput, width, height, impl->dInputPitch, impl->dOutputPitch, impl->pattern);
	}
	else
	{
		demosaic_bilinear_8u_device(impl->dInput, impl->dOutput, impl->input.cols, impl->input.rows, impl->dInputPitch, impl->dOutputPitch, impl->pattern);
	}
}

void App::runImage()
{
	impl->input = cv::imread(impl->inputPath, cv::IMREAD_UNCHANGED);

	if (impl->input.empty())
	{
		throw std::runtime_error("Invalid input image path!");
	}

	impl->output.create(impl->input.size(), CV_MAKE_TYPE(impl->input.depth(), 3));

	size_t cfa_pitch, rgb_pitch;

	SAFE_CALL(cudaMallocPitch(&(impl->dInput), &cfa_pitch, impl->input.cols, impl->input.rows));
	SAFE_CALL(cudaMallocPitch(&(impl->dOutput), &rgb_pitch, impl->output.cols * impl->output.channels(), impl->output.rows));

	impl->dInputPitch = static_cast<int>(cfa_pitch);
	impl->dOutputPitch = static_cast<int>(rgb_pitch);

	SAFE_CALL(cudaMemcpy2D(impl->dInput, cfa_pitch, impl->input.ptr(), impl->input.step, impl->input.cols, impl->input.rows, cudaMemcpyHostToDevice));

	process<false>();

	SAFE_CALL(cudaMemcpy2D(impl->output.ptr(), impl->output.step, impl->dOutput, rgb_pitch, impl->output.cols * impl->output.channels(), impl->output.rows, cudaMemcpyDeviceToHost));

	if (impl->showOutput)
	{
		cv::namedWindow(impl->windowTitle, cv::WINDOW_NORMAL);
		cv::imshow(impl->windowTitle, impl->output);
		cv::waitKey();
		cv::destroyWindow(impl->windowTitle);
	}

	if (impl->writeOutput)
	{
		if (!cv::imwrite(impl->outputPath, impl->output))
		{
			throw std::runtime_error("Invalid output image path!");
		}
	}

	SAFE_CALL(cudaFree(impl->dInput));
	SAFE_CALL(cudaFree(impl->dOutput));
}

void App::runVideo()
{
	
	double seconds, fps;

	
	if (impl->isVideoFile)
	{
		if (!impl->capture.open(impl->inputPath))
			throw std::runtime_error("Invalid input video path!");
	}
	else if (impl->isVideoLive)
	{
		if (impl->capture.open(impl->camera_id))
			throw std::runtime_error("Invalid camera input!");
	}
	else
	{
		throw;
	}

	int width = static_cast<int>(impl->capture.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(impl->capture.get(cv::CAP_PROP_FRAME_HEIGHT));

	bool validWriter = true;

	if (impl->writeOutput)
	{
		int fcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
		double output_fps = impl->capture.get(cv::CAP_PROP_FPS);
		if (impl->isVideoLive)	output_fps = 30.0;
		
		if (!impl->writer.open(impl->outputPath, fcc, output_fps, cv::Size(width, height), true))
		{
			printf("Warning! Unable to write output video. Continuing without writing!\n");
			validWriter = false;
		}
	}

	impl->writeOutput = false;

	impl->output.create(height, width, CV_8UC3);

	size_t cfa_pitch, rgb_pitch;
	SAFE_CALL(cudaMallocPitch(&(impl->dInput), &cfa_pitch, width, height));
	SAFE_CALL(cudaMallocPitch(&(impl->dOutput), &rgb_pitch, width * 3, height));

	impl->dInputPitch = static_cast<int>(cfa_pitch);
	impl->dOutputPitch = static_cast<int>(rgb_pitch);

	cv::namedWindow(impl->windowTitle, cv::WINDOW_NORMAL);

	cv::Mat frame;
	bool showFPS = true; //Display FPS or not
	bool showRaw = false; //Display bayer video or demosaicked video

	while (impl->capture.grab())
	{
		impl->capture.retrieve(impl->input);
		
		if (impl->input.channels() == 3)
			cv::cvtColor(impl->input, frame, cv::COLOR_BGR2GRAY);
		else
			frame = impl->input.clone();

		auto start = std::chrono::steady_clock::now();
		
		SAFE_CALL( cudaMemcpy2D(impl->dInput, cfa_pitch, frame.ptr(), frame.step, width, height, cudaMemcpyHostToDevice) );

		process<true>();

		SAFE_CALL(cudaMemcpy2D(impl->output.ptr(), impl->output.step, impl->dOutput, rgb_pitch, impl->output.cols * impl->output.channels(), impl->output.rows, cudaMemcpyDeviceToHost));

		auto stop = std::chrono::steady_clock::now();

		seconds = std::chrono::duration_cast<std::chrono::duration<double> > (stop - start).count();

		fps = 1000.0 / seconds;
		stringstream ss;
		ss << (int)fps<<" FPS";

		if (impl->showOutput)
		{
			if (showFPS)
				cv::putText(impl->output, ss.str(), cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 5.0, CV_RGB(0, 0, 255), 3);

			if (showRaw)	cv::imshow(impl->windowTitle, impl->input);
			else			cv::imshow(impl->windowTitle, impl->output);

			int key = cv::waitKey(10);

			if (key == 'r')
			{
				impl->writeOutput = true;
			}
			else if (key == 's')
			{
				impl->writeOutput = false;
				validWriter = false;
				impl->writer.release();
			}
			else if (key == 'f')
			{
				showFPS = !showFPS;
			}
			else if (key == 'd')
			{
				showRaw = false;
			}
			else if (key == 'w')
			{
				showRaw = true;
			}
		}
		if (validWriter && impl->writeOutput)
		{
			impl->writer << impl->output;
		}
	}

	cv::destroyWindow(impl->windowTitle);
	impl->capture.release();
	impl->writer.release();
	
	SAFE_CALL(cudaFree(impl->dInput));
	SAFE_CALL(cudaFree(impl->dOutput));
}

void App::run(int argc, char** argv)
{
	if (impl)
	{
		parse_arguments(argc, argv);

		if (impl->isImage)
		{
			runImage();
		}
		else
		{
			runVideo();
		}
		cudaDeviceReset();
	}
	else
	{
		throw std::runtime_error("Invalid Application");
	}
}
