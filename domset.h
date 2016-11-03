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

// 3D Party
#include "nanoflann.hpp"
using namespace nanoflann;

namespace nomoko {
  class Domset{
    private:
      /* Generic clustering */
      void findCommonPoints(const View& v1, const View& v2,
          std::vector<size_t>& commonPts);
      // similarity measures
      const float computeViewSimilaity(const View&, const View&);
      Eigen::MatrixXf getSimilarityMatrix(std::map<size_t,size_t>&);

      // distance measures
      void getAllDistances();
      float getDistanceMedian(std::map<size_t,size_t> &);
      float computeViewDistance(const size_t& vId1, const size_t & vId2,
              const float& medianDist);

      void computeInformation();

      // subsamples the initial point cloud
      void voxelGridFilter(const float& leafSizeX,
            const float& leafSizeY, const float& leafSizeZ);

      // normalizing point distances
      void normalizePointCloud();
      void deNormalizePointCloud();

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
      void computeClustersAP(std::map<size_t,size_t>&, std::vector<std::vector<size_t> >&);

      void clusterViews(std::map<size_t, size_t>& xId2vId, const size_t& minClustersize,
          const size_t& maxClusterSize);

      void clusterViews(const size_t& minClustersize,
          const size_t& maxClusterSize);

      /* export function */
      void exportToPLY(const std::string& plyFile, bool exportPoints = false);

      const std::vector<std::vector<size_t> >& getClusters() {
        return finalClusters;
      }

      void setClusters(std::vector<std::vector<size_t> > clusters) {
        finalClusters.clear();
        finalClusters.swap(clusters);
      }

      void printClusters();

    private:
      std::vector<Point> points;
      std::vector<Point> origPoints;
      std::vector<Camera> cameras;
      std::vector<View> views;

      Eigen::MatrixXf viewDists;

      std::vector<std::vector<size_t > > finalClusters;

      const float kAngleSigma = M_PI_4;
      const float kAngleSigma_2 = kAngleSigma * kAngleSigma;
      size_t kMinClusterSize = 10;
      size_t kMaxClusterSize = 20;
      float kE = 0.01f;

      // AP constants
      const unsigned int kNumIter = 100;
      const float lambda = 0.8;

      // scale normalization
      float normScale;
      Point normMin;
      Point normMax;
      Point pcCentre;

      // voxel grid stuff
      const float kVoxelSize = 0.1f;

    public:
      /* nanoflann function */
      // returns number of data points
      inline size_t kdtree_get_point_count() const { return points.size(); }

      inline float kdtree_distance
          (const float* p1, const size_t idx_p2, size_t /**size**/) const {
        const float d0 = p1[0]-points[idx_p2].pos(0);
        const float d1 = p1[1]-points[idx_p2].pos(1);
        const float d2 = p1[2]-points[idx_p2].pos(2);
        return d0*d0 + d1*d1 + d2*d2;
      }

      inline float kdtree_get_pt(const size_t idx, int dim) const {
        if (dim == 0) return points[idx].pos(0);
        else if (dim == 1) return points[idx].pos(1);
        else return points[idx].pos(2);
      }

      template <class BBOX>
      bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
  }; // class Domset
} // namespace nomoko
#endif // _DOMSET_H_
