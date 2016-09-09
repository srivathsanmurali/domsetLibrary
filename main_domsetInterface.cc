#include <iostream>
#include <ctime>
#include <iomanip>
#include <cstdlib>

#include <omp.h>
#include <domsetInterface.h>
#include <domset.h>
using nomoko::Interface;
using nomoko::Domset;

int main(int argc, char** argv) {
  if(argc < 2){
    std::cout << "Error: domsetInterface bundleFile outputPrefix"<< std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Arguments : "
            << "\nBundleFile      = " << argv[1]
            << "\nPLYFile         = " << argv[2] << std::endl;

  // reading the bundle file
  Interface di;
  di.load(argv[1]);
  Domset d(di.points, di.views, di.cameras, 10.0f);
  d.clusterViews(20,30);
  d.exportToPLY(argv[2]);
}
