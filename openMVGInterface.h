#ifndef _NOMOKO_OpenMVG_Interface_H_
#define _NOMOKO_OpenMVG_Interface_H_

#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include <types.h>
namespace nomoko {
  class OpenMVGInterface {
    private:
      std::vector<Camera>     cameras;
      std::vector<View>       views;
      std::vector<Point>      points;

    public:
      std::shared_ptr<OpenMVGInterface> load (const std::string& _filename);
      void save (const std::string & _filename );
  }; // class OpenMVGInterface
} // namespace nomoko
#endif // _NOMOKO_OpenMVG_Interface_H_
