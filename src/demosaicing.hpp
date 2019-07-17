#pragma once

//Main application class
class App
{
public:

	struct AppImpl;
	AppImpl* impl;
	
	App();

	//Start the application
	void run(int argc, char** argv);

	//Print application usage
	static void usage();

	~App();

private:
	void parse_arguments(int argc, char** argv);
	void runImage();
	void runVideo();
	template<bool video> void process();

};

//Pattern of Bayer Color Filter Array
enum BayerPattern
{
	BAYER_UNKNOWN = -1,
	BAYER_BGGR,
	BAYER_RGGB,
	BAYER_GBRG,
	BAYER_GRBG,
};