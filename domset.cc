#include "domset.h"
#if USE_OpenMP
#include <omp.h>
#endif

namespace nomoko {
  void Domset::computeInformation() {
    normalizePointCloud();
    voxelGridFilter(kVoxelSize, kVoxelSize, kVoxelSize);
    getAllDistances();
  }

  void Domset::normalizePointCloud() {
    // construct a kd-tree index:
    typedef KDTreeSingleIndexAdaptor<
      L2_Simple_Adaptor<float, Domset> ,
      Domset,
      3 /* dim */> my_kd_tree_t;

    my_kd_tree_t index(3, *this, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    const size_t numResults(1);
    const size_t numPoints (points.size());
    float totalDist = 0;
    pcCentre.pos << 0, 0, 0;
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    for (size_t i =0; i < numPoints; i++) {
      const Point & p = points[i];
      float queryPt[3] = {p.pos(0), p.pos(1), p.pos(2)};

      std::vector<size_t> ret_index(2);
      std::vector<float> out_dist_sqr(2);
      index.knnSearch(queryPt,2, &ret_index[0], &out_dist_sqr[0]);

      #if USE_OpenMP
      #pragma omp critical(TotalDistanceUpdate)
      #endif
      {
        totalDist += (p.pos - points[ret_index[1]].pos).norm();
        pcCentre.pos(0) += (p.pos(0) / numPoints);
        pcCentre.pos(1) += (p.pos(1) / numPoints);
        pcCentre.pos(2) += (p.pos(2) / numPoints);
      }
    }

    // calculation the normalization scale
    const float avgDist = totalDist/numPoints;
    std::cerr << "Total distance = " << totalDist<< std::endl;
    std::cerr << "Avg distance = " << avgDist << std::endl;
    normScale = 1/avgDist;
    std::cerr << "Normalization Scale = " << normScale << std::endl;

    // normalizing the distances on points
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    for (size_t i =0; i < numPoints; i++) {
      points[i].pos = (points[i].pos - pcCentre.pos) * normScale;
    }

    // normalizing the camera center positions
    const size_t numViews(views.size());
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < numViews; i++) {
      views[i].trans = (views[i].trans - pcCentre.pos) * normScale;
    }
  }

  void Domset::deNormalizePointCloud() {
    const size_t numPoints (points.size());
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    // denormalizing points
    for (size_t i =0; i < numPoints; i++) {
      points[i].pos = (points[i].pos / normScale) + pcCentre.pos;
    }

    // denormalizing camera centers
    const size_t numViews(views.size());
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < numViews; i++) {
      views[i].trans = (views[i].trans / normScale) + pcCentre.pos;
    }
  }

  void Domset::voxelGridFilter(const float& sizeX, const float& sizeY, const float& sizeZ) {
    if(sizeX <= 0.0f || sizeY <= 0.0f || sizeZ <= 0.0f) {
      std::cerr << "Invalid voxel grid dimensions error.\n";
      exit(0);
    }

    Point minPt;
    Point maxPt;
    const size_t numP = points.size();
    // finding the min and max values for the 3 dimensions
    const float mi = std::numeric_limits<float>::min();
    const float ma = std::numeric_limits<float>::max();
    minPt.pos << ma, ma, ma;
    maxPt.pos << mi, mi, mi;

    for(size_t p = 0; p < numP; p++) {
      const Point newSP = points[p];
      if(newSP.pos(0) < minPt.pos(0)) minPt.pos(0) = newSP.pos(0);
      if(newSP.pos(1) < minPt.pos(1)) minPt.pos(1) = newSP.pos(1);
      if(newSP.pos(2) < minPt.pos(2)) minPt.pos(2) = newSP.pos(2);
      if(newSP.pos(0) > maxPt.pos(0)) maxPt.pos(0) = newSP.pos(0);
      if(newSP.pos(1) > maxPt.pos(1)) maxPt.pos(1) = newSP.pos(1);
      if(newSP.pos(2) > maxPt.pos(2)) maxPt.pos(2) = newSP.pos(2);
    }

    // finding the number of voxels reqired
    size_t numVoxelX = static_cast<size_t>(ceil(maxPt.pos(0) - minPt.pos(0))/sizeX);
    size_t numVoxelY = static_cast<size_t>(ceil(maxPt.pos(1) - minPt.pos(1))/sizeY);
    size_t numVoxelZ = static_cast<size_t>(ceil(maxPt.pos(2) - minPt.pos(2))/sizeZ);
    std::cout << "Max = "<<maxPt.pos.transpose() << std::endl;
    std::cout << "Min = "<<minPt.pos.transpose() << std::endl;
    std::cout << "Max - Min = "<<(maxPt.pos - minPt.pos).transpose() << std::endl;
    std::cout << "VoxelSize X = "   <<sizeX << std::endl;
    std::cout << "VoxelSize Y = "   <<sizeY << std::endl;
    std::cout << "VoxelSize Z = "   <<sizeZ << std::endl;
    std::cout << "Number Voxel X = "<<numVoxelX << std::endl;
    std::cout << "Number Voxel Y = "<<numVoxelY << std::endl;
    std::cout << "Number Voxel Z = "<<numVoxelZ << std::endl;

    /* adding points to the voxels */
    std::map<size_t, std::vector<size_t> > voxels;
    std::vector<size_t> voxelIds;
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    for(size_t p = 0; p < numP; p++) {
      const Point pt = points[p];
      const size_t x = static_cast<size_t>(floor((pt.pos(0) - minPt.pos(0))/sizeX));
      const size_t y = static_cast<size_t>(floor((pt.pos(1) - minPt.pos(1))/sizeY));
      const size_t z = static_cast<size_t>(floor((pt.pos(2) - minPt.pos(2))/sizeZ));
      const size_t id = (z * numVoxelZ) + (y * numVoxelY) + x;
      #if USE_OpenMP
      #pragma omp critical(voxelGridUpdate)
      #endif
      {
        if(voxels.find(id) == voxels.end()) {
          voxels[id] = std::vector<size_t>();
          voxelIds.push_back(id);
        }

        voxels[id].push_back(p);
      }
    }

    std::vector<Point> newPoints;
    const size_t numVoxelMaps = voxelIds.size();
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    for(size_t vmId = 0; vmId < numVoxelMaps; vmId++) {
      const size_t vId = voxelIds[vmId];
      const size_t nPts = voxels[vId].size();
      if(nPts == 0) continue;

      Eigen::Vector3f pos;
      std::set<size_t> vl;
      for(const auto & p : voxels[vId]) {
        const Point pt = points[p];
        pos += pt.pos;
        const size_t numV = pt.viewList.size();
        for(size_t j =0; j < numV; j++)
          vl.insert(pt.viewList[j]);
      }
      pos(0) = pos(0) / nPts;
      pos(1) = pos(1) / nPts;
      pos(2) = pos(2) / nPts;

      Point newSP;
      newSP.pos = pos;
      newSP.viewList = std::vector<size_t>(vl.begin(), vl.end());
      #if USE_OpenMP
      #pragma omp critical(pointsUpdate)
      #endif
      {
        for(const size_t viewID : vl) {
          views[viewID].viewPoints.push_back(newPoints.size());
        }
        newPoints.push_back(newSP);
      }
    }

    std::cerr << "New points = " << newPoints.size() << std::endl;
    origPoints.clear();
    points.swap(origPoints);
    points.swap(newPoints);
    std::cerr << "Number of points = " << points.size() << std::endl;
  } // voxelGridFilter

  Eigen::MatrixXf Domset::getSimilarityMatrix(std::map<size_t,size_t>& xId2vId) {
    std::cout << "Generating Similarity Matrix "<< std::endl;
    const size_t numC = xId2vId.size();
    const size_t numP = points.size();
    if(numC == 0 || numP == 0) {
      std::cerr << "Invalid Data\n";
      exit(0);
    }
    float medianDist =  getDistanceMedian(xId2vId);
    std::cout << "Median dists = " << medianDist << std::endl;
    Eigen::MatrixXf simMat;
    simMat.resize(numC, numC);
    #if USE_OpenMP
    #pragma omp parallel for collapse(2)
    #endif
    for( size_t xId1 = 0; xId1 < numC; xId1++) {
      for( size_t xId2 = 0; xId2 < numC; xId2++) {
        const size_t vId1 = xId2vId[xId1];
        const size_t vId2 = xId2vId[xId2];
        if( vId1 == vId2) {
          simMat(xId1, xId2) = 0;
        } else {
          const View v2 = views[vId2];
          const View v1 = views[vId1];
          const float sv = computeViewSimilaity(v1,v2);
          const float sd  = computeViewDistance(vId1, vId2, medianDist);
          const float sim = sv * sd;
          simMat(xId1, xId2) = sim;
        }
      }
    }
    return simMat;
  } // getSimilarityMatrix

  float Domset::computeViewDistance(const size_t& vId1, const size_t& vId2, const float& medianDist) {
    if(vId1 == vId2) return 1;
    float vd = viewDists(vId1, vId2);
    float dm = 1 + exp(- (vd - medianDist) / medianDist);
    return 1/dm;
  }
  float Domset::getDistanceMedian(std::map<size_t,size_t> & xId2vId) {
    std::cout << "Finding Distance Median\n";
    const size_t numC = xId2vId.size();
    if(numC == 0) {
      std::cerr << "No Views initialized \n";
      exit(0);
    }

    std::vector<float> dists(numC*numC);
    // float totalDist = 0;
    for(size_t i = 0; i < numC; i++) {
      const auto v1 = xId2vId[i];
      for(size_t j = 0; j < numC; j++ ) {
        const auto v2 = xId2vId[j];
        dists.push_back(viewDists(v1,v2));
        // totalDist += viewDists(v1,v2);
      }
    }
    std::sort(dists.begin(), dists.end());
    return dists[dists.size() /2];
    // return totalDist / numC;
  } // getDistanceMedian

  void Domset::getAllDistances() {
    std::cout << "Finding View Distances\n";
    const size_t numC = views.size();
    if(numC == 0) {
      std::cerr << "No Views initialized \n";
      exit(0);
    }
    viewDists.resize(numC, numC);
    for(size_t i = 0; i < numC; i++) {
      const auto v1 = views[i];
      for(size_t j = 0; j < numC; j++ ) {
        const auto v2 = views[j];
        const float dist = (v1.trans - v2.trans).norm();
        viewDists(i,j) = dist;
      }
    }
  }
  void Domset::findCommonPoints(const View& v1, const View& v2,
      std::vector<size_t>& commonPoints){
    commonPoints.clear();
    const size_t numVP1 = v1.viewPoints.size();
    const size_t numVP2 = v2.viewPoints.size();
    const size_t minNum = std::min(numVP1, numVP2);

    //std::sort(v1.viewPoints.begin(), v1.viewPoints.end());
    //std::sort(v2.viewPoints.begin(), v2.viewPoints.end());
    commonPoints.resize(minNum);

    auto it = std::set_intersection(v1.viewPoints.begin(), v1.viewPoints.end(),
      v2.viewPoints.begin(), v2.viewPoints.end(), commonPoints.begin());
    commonPoints.resize(it - commonPoints.begin());
  } // findCommonPoints

  const float Domset::computeViewSimilaity(const View& v1, const View& v2) {
    std::vector<size_t> commonPoints;
    findCommonPoints(v1, v2, commonPoints);
    const size_t numCP = commonPoints.size();

    float w =0;
    #if USE_OpenMP
    #pragma omp parallel for
    #endif
    for( size_t p=0; p < numCP; p++){
      const auto pId = commonPoints[p];
      //for( const auto pId : commonPoints ){
      Eigen::Vector3f c1 = v1.trans - points[pId].pos;
      c1.normalize();
      Eigen::Vector3f c2 = v2.trans - points[pId].pos;
      c2.normalize();
      const float angle = acos(c1.dot(c2));
      const float expAngle = exp(- ( angle * angle) / kAngleSigma_2);
      //std::cerr << angle <<  " = " << expAngle << std::endl;
      #if USE_OpenMP
      #pragma omp atomic
      #endif
      w += expAngle;
    }
    const float ans = w / numCP;
    return (ans != ans)? 0 : ans;
    } // computeViewSimilaity

    void Domset::computeClustersAP(std::map<size_t, size_t>& xId2vId,
        std::vector<std::vector<size_t> >& clusters) {
      const size_t numX = xId2vId.size();
      if(numX == 0) {
        std::cout << "Invalid map size\n";
        exit(0);
      }

      Eigen::MatrixXf S = getSimilarityMatrix(xId2vId);
      Eigen::MatrixXf R(numX, numX);
      R.setConstant(0);
      Eigen::MatrixXf A(numX, numX);
      A.setConstant(0);

      for(size_t m=0; m<kNumIter; m++) {
        //update responsibility
        #if USE_OpenMP
        #pragma omp parallel for collapse(2)
        #endif
        for(size_t i =0; i<numX; i++) {
          for(size_t k=0; k<numX; k++) {
            float max1 = std::numeric_limits<float>::min();
            float max2 = std::numeric_limits<float>::min();

            for(size_t kk=0; kk<k; kk++) {
              if(S(i,kk) +  A(i,kk) >max1)
                max1 = S(i,kk) +A(i,kk);
            }
            for(size_t kk=k+1; kk<numX; kk++) {
              if(S(i,kk) +A(i,kk) >max2)
                max2 = S(i,kk) +A(i,kk);
            }
            float max = std::max(max1,max2);
            R(i,k) = (1-lambda)*(S(i,k) - max) + lambda*R(i,k) ;
          }
        }

        //update availability
        #if USE_OpenMP
        #pragma omp parallel for collapse(2)
        #endif
        for(size_t i=0; i<numX; i++) {
          for(size_t k=0; k<numX; k++) {
            if(i==k) continue;
            const size_t maxik = std::max(i, k);
            const size_t minik = std::min(i, k);
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;
            float r1, r2, r3;
            for(size_t ii=0; ii<minik; ii++) {
              r1 = R(ii,k);
              //sum1 += std::max(0.0f, r1);
              if(r1 > 0.0f)
                sum1 += r1;
            }
            for(size_t ii=minik+1; ii<maxik; ii++) {
              r2 = R(ii,k);
              // sum2 += std::max(0.0f, r2);
              if(r2 > 0.0f)
                sum2 += r2;
            }
            for(size_t ii=maxik+1; ii<numX; ii++) {
              r3 = R(ii,k);
              // sum3 += std::max(0.0f, r3);
              if(r3 > 0.0f)
                sum3 += r3;
            }
            const float r = R(k,k) + sum1 + sum2 + sum3;
            A(i,k) = (1-lambda)*std::min(0.0f, r) + lambda*A(i,k);
          }
        }
      }
      #if USE_OpenMP
      #pragma omp parallel for
      #endif
      for(size_t i =0; i<numX; i++) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float r1, r2;
        for(size_t ii=0; ii<i; ii++) {
          r1 = R(ii,i);
          //sum1 += std::max(0.0f, r1);
          if(r1 > 0.0f)
            sum1 += r1;
        }
        for(size_t ii=i+1; ii<numX; ii++) {
          r2 = R(ii,i);
          //sum2 += std::max(0.0f, r2);
          if(r2 > 0.0f)
            sum2 += r2;
        }
        A(i,i) = (1-lambda)*(sum1+sum2) + lambda*A(i,i);
      }

      //find the exemplar
      Eigen::MatrixXf E(numX, numX);
      E = R + A;

      // getting initial clusters
      std::set<size_t > centers;
      std::map<size_t, std::vector<size_t>> clMap;
      size_t idxForI = 0;
      for(size_t i=0; i<numX; i++) {
        float maxSim = std::numeric_limits<float>::min();
        for(size_t j=0; j<numX; j++) {
          if (E(i,j)>maxSim) {
            maxSim = E(i,j);
            idxForI = j;
          }
        }
        centers.insert(idxForI);
      }

      for(auto const c : centers)
        clMap[c] = std::vector<size_t>();

      for(size_t i = 0; i < numX; i++ ) {
        float maxSim = std::numeric_limits<float>::min();
        for(auto const c : centers) {
          if( S(i,c) > maxSim){
            idxForI = i;
            maxSim = S(i,c);
          }
        }
        clMap[idxForI].push_back(i);
      }

      // enforcing min size constraints
      bool change = false;
      do{
        change = false;
        for(const auto p1 : clMap) {
          const size_t vId1 = xId2vId[p1.first];
          if(p1.second.size() < kMinClusterSize) {
            float minDist = std::numeric_limits<float>::max();
            int minId = -1;
            for(const auto p2 : clMap) {
              if(p1.first == p2.first) continue;
              const size_t vId2 = xId2vId[p2.first];
              if(viewDists(vId1, vId2) < minDist
                  && (p1.second.size() + p2.second.size()) < kMaxClusterSize) {
                minDist = viewDists(vId1, vId2);
                minId = p2.first;
              }
            }
            if(minId > -1) {
              change = true;
              clMap[minId].insert(clMap[minId].end(),
                  p1.second.begin(), p1.second.end());
              //std::cout << "merge " << p1.first << " -> " << minId << std::endl;
            }
            clMap.erase(clMap.find(p1.first));
          }
        }
      }while(change);

      // enforcing max size constraints
      // adding it to clusters vector
      for(const auto p : clMap) {
        std::vector<size_t> cl;
        for(const auto i : p.second){
          cl.push_back(xId2vId[i]);
        }
        if(cl.size() > kMaxClusterSize) {
          // std::cout << "split " << p.first << " | " << p.second.size() << std::endl;
          auto it = cl.begin();
          while(true) {
            auto stop = it + kMaxClusterSize;
            if(stop < cl.end()) {
              auto tmp = std::vector<size_t>(it, stop);
              it = stop;
              std::sort(tmp.begin(), tmp.end());
              clusters.push_back(tmp);
            } else {
              std::vector<size_t> tmp;
              while(it < cl.end()) {
                tmp.push_back(*it);
                it++;
              }
              std::sort(tmp.begin(), tmp.end());
              clusters.push_back(tmp);
              break;
            }
          }
        }else {
          std::sort(cl.begin(), cl.end());
          clusters.push_back(cl);
        }
      }
    }

    void Domset::clusterViews(std::map<size_t,size_t>& xId2vId, const size_t& minClusterSize,
          const size_t& maxClusterSize) {
      std::cout << "[ Clustering Views ] "<< std::endl;
      const size_t umC = views.size();
      kMinClusterSize = minClusterSize;
      kMaxClusterSize = maxClusterSize;

      std::vector<std::vector<size_t> > clusters;
      computeClustersAP(xId2vId, clusters);

      deNormalizePointCloud();
      finalClusters.swap(clusters);
    }

    void Domset::clusterViews(
        const size_t& minClusterSize, const size_t& maxClusterSize){
      std::cout << "[ Clustering Views ] "<< std::endl;
      const size_t numC = views.size();
      kMinClusterSize = minClusterSize;
      kMaxClusterSize = maxClusterSize;

      std::map<size_t,size_t> xId2vId;
      for(size_t i =0; i <numC; i++) {
        xId2vId[i] = i;
      }
      std::vector<std::vector<size_t> > clusters;
      computeClustersAP(xId2vId, clusters);

      deNormalizePointCloud();
      finalClusters.swap(clusters);
    }

    void Domset::printClusters() {
      std::stringstream ss;
      ss << "Clusters : \n";
      for(const auto cl : finalClusters){
        ss << cl.size() << " : ";
        for(const auto id : cl) {
          ss << id << " ";
        }
        ss << "\n\n";
      }
      std::cout << "Number of clusters = " << finalClusters.size() << std::endl;
      std::cout << ss.str();
    }
    void Domset::exportToPLY(const std::string& plyFilename, bool exportPoints) {
      std::stringstream plys;
      plys    << "ply\n"
        << "format ascii 1.0\n";

      size_t totalViews = 0;
      for(const auto cl : finalClusters)
        totalViews += cl.size();
      const size_t numPts = origPoints.size();

      size_t totalPoints = totalViews;
      if (exportPoints) totalPoints += numPts;
      plys    << "element vertex "
              << totalPoints << std::endl
              << "property float x\n"
              << "property float y\n"
              << "property float z\n"
              << "property uchar red\n"
              << "property uchar green\n"
              << "property uchar blue\n"
              << "end_header\n";

      for(const auto cl : finalClusters) {
        uint red, green, blue;
        red = (rand() % 255);
        green = (rand() % 255);
        blue = (rand() % 255);
        for(const auto id : cl) {
          Eigen::Vector3f pos = views[id].trans;
          plys << pos(0) << " " << pos(1) << " " << pos(2) << " "
            << red << " " << green << " " << blue << std::endl;
        }
      }

      if (exportPoints) {
        for(const auto pt : origPoints) {
          Eigen::Vector3f pos = pt.pos.transpose();

          plys << pos(0) << " " << pos(1) << " " << pos(2)
            << " 255 255 255" << std::endl;
        }
      }

      std::ofstream plyFile (plyFilename);
      if(!plyFile.is_open()) {
        std::cout << "Cant open " << plyFilename << " file\n";
      } else {
        plyFile << plys.str();
        plyFile.close();
      }
    }
  }
