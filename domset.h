/**
 *             __________
 *            / ________ .
 *           / / ______ . .
 *          / / /      . . .
 *         / / /        . . .
 *         . . .        / / /
 *          . . .______/ / /
 *           . .________/ /
 *            .__________/
 *
 *              nomoko AG
 *          www.nomoko.camera
 *
 *        Röschibachstrasse 24
 *           CH-8037 Zurich
 *            Switzerland
 *
 * @endcond
 * @file        domset.h
 * @brief       clustering algorithm to cluster views from SFM.
 * @details     The clustering alogrithm uses dominant clustering
 *              introduced in "http://www.vision.ee.ethz.ch/~rhayko
 *              /paper/3dv2014_mauro_joint_selection_clustering.pdf"
 * @author      Srivathsan Murali<srivathsan@nomoko.camera>
 * @copyright   MIT
 */
// Chuck Norris programs do not accept input.

#ifndef _DOMSET_H_
#define _DOMSET_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>
#include "types.h"

namespace nomoko {
  class Domset{
    private:
      /* Generic clustering */
      void findCommonPoints(const View&, const View&, std::vector<int>&);
      // similarity measures
      float computeViewSimilaity(const View&, const View&);
      Eigen::MatrixXf getSimilarityMatrix(std::map<int,int>&);

      // distance measures
      void getAllDistances();
      float getDistanceMedian(std::map<int,int> &);
      float computeViewDistance(const int&, const int&, const float&);

      void computeInformation();
      void voxelGridFilter(const float&, const float&, const float&);


    public:
      /* cant read from file need to suply data */
      Domset(const std::vector<Point>&   _points,
                  const std::vector<View>&    _views,
                  const std::vector<Camera>&  _cameras,
                  const float& _kVoxelsize):
                  points(_points), views(_views),
                  cameras(_cameras), kVoxelSize(_kVoxelsize) {
        std::cout << " [ Dominant set clustering of views ] " << std::endl;
        computeInformation();
      }

      // AP clustering
      void computeClustersAP(std::map<int,int>&, std::vector<std::vector<int> >&);

      void clusterViews(std::map<int, int>& xId2vId, const int& minClustersize,
          const int& maxClusterSize);

      void clusterViews(const int& minClustersize,
          const int& maxClusterSize);

      /* export function */
      void exportToPLY(const std::string& plyFile, bool exportPoints = false);

      const std::vector<std::vector<int> >& getClusters() {
        return finalClusters;
      }

      void setClusters(std::vector<std::vector<int> > clusters) {
        finalClusters.clear();
        finalClusters.swap(clusters);
      }

      void printClusters();

    private:
      std::vector<Point> points;
      std::vector<Point> origPoints;
      std::vector<Camera> cameras;
      std::vector<View> views;
      Point minPt;
      Point maxPt;

      Eigen::MatrixXf viewDists;

      std::vector<std::vector<int> > finalClusters;

      const float kAngleSigma = M_PI_4;
      const float kAngleSigma_2 = kAngleSigma * kAngleSigma;
      int kMinClusterSize = 10;
      int kMaxClusterSize = 20;
      float kE = 0.01f;

      // AP constants
      const int kNumIter = 100;
      const float lambda = 0.8;

      // voxel grid stuff
      const float kVoxelSize = 0.1f;
  }; // class Domset
} // namespace nomoko
#endif // _DOMSET_H_
