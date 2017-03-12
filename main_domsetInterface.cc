// This file is part of OpenMVG (Open Multiple View Geometry) C++ library.
// Copyright (c) 2016 nomoko AG, Srivathsan Murali<srivathsan@nomoko.camera>

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <ctime>
#include <iomanip>
#include <cstdlib>

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
