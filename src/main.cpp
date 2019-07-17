#include <iostream>
#include "demosaicing.hpp"

using std::cout;
using std::endl;

int main(int argc, char** argv)
{
	try
	{
		App app;
		
		//Run the application with specfied arguments
		app.run(argc, argv);
	}
	catch (std::runtime_error& ex)
	{
		cout << ex.what() << endl;
		App::usage();
	}

	return 0;
}

