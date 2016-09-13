#include "domset.h"
#include <omp.h>

namespace nomoko {
  void Domset::computeInformation() {
    voxelGridFilter(kVoxelSize, kVoxelSize, kVoxelSize);
    getAllDistances();
  }

  void Domset::voxelGridFilter(const float& sizeX, const float& sizeY, const float& sizeZ) {
    if(sizeX <= 0.0f || sizeY <= 0.0f || sizeZ <= 0.0f) {
      std::cerr << "Invalid voxel grid dimensions error.\n";
      exit(0);
    }

    const unsigned int numP = points.size();
    // finding the min and max values for the 3 dimensions
    const float mi = std::numeric_limits<float>::min();
    const float ma = std::numeric_limits<float>::max();
    minPt.pos << ma, ma, ma;
    maxPt.pos << mi, mi, mi;

    for(int p = 0; p < numP; p++) {
      const Point newSP = points[p];
      if(newSP.pos(0) < minPt.pos(0)) minPt.pos(0) = newSP.pos(0);
      if(newSP.pos(1) < minPt.pos(1)) minPt.pos(1) = newSP.pos(1);
      if(newSP.pos(2) < minPt.pos(2)) minPt.pos(2) = newSP.pos(2);
      if(newSP.pos(0) > maxPt.pos(0)) maxPt.pos(0) = newSP.pos(0);
      if(newSP.pos(1) > maxPt.pos(1)) maxPt.pos(1) = newSP.pos(1);
      if(newSP.pos(2) > maxPt.pos(2)) maxPt.pos(2) = newSP.pos(2);
    }

    // finding the number of voxels reqired
    unsigned int numVoxelX = static_cast<int>(ceil(maxPt.pos(0) - minPt.pos(0))/sizeX);
    unsigned int numVoxelY = static_cast<int>(ceil(maxPt.pos(1) - minPt.pos(1))/sizeY);
    unsigned int numVoxelZ = static_cast<int>(ceil(maxPt.pos(2) - minPt.pos(2))/sizeZ);
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
    std::map<unsigned int, std::vector<unsigned int> > voxels;
    std::vector<unsigned int> voxelIds;
#pragma omp parallel for
    for(unsigned int p = 0; p < numP; p++) {
      const Point pt = points[p];
      const unsigned int x = static_cast<int>(floor(pt.pos(0) - minPt.pos(0))/sizeX);
      const unsigned int y = static_cast<int>(floor(pt.pos(1) - minPt.pos(1))/sizeY);
      const unsigned int z = static_cast<int>(floor(pt.pos(2) - minPt.pos(2))/sizeZ);
      const unsigned int id = (z * numVoxelZ) + (y * numVoxelY) + x;
#pragma omp critical(voxelGridUpdate)
      {
        if(voxels.find(id) == voxels.end()) {
          voxels[id] = std::vector<unsigned int>();
          voxelIds.push_back(id);
        }
        voxels[id].push_back(p);
      }
    }

    std::vector<Point> newPoints;
    const unsigned int numVoxelMaps = voxelIds.size();
#pragma omp parallel for
    for(unsigned int vmId = 0; vmId < numVoxelMaps; vmId++) {
      const unsigned int vId = voxelIds[vmId];
      const unsigned int nPts = voxels[vId].size();
      if(nPts == 0) continue;

      Eigen::Vector3f pos;
      std::set<unsigned int> vl;
      for(const unsigned int p : voxels[vId]) {
        const Point pt = points[p];
        pos += pt.pos;
        const int numV = pt.viewList.size();
        for(int j =0; j < numV; j++)
          vl.insert(pt.viewList[j]);
      }
      pos(0) = pos(0) / nPts;
      pos(1) = pos(1) / nPts;
      pos(2) = pos(2) / nPts;

      Point newSP;
      newSP.pos = pos;
      newSP.viewList = std::vector<unsigned int>(vl.begin(), vl.end());
#pragma omp critical(pointsUpdate)
      {
        for(const int viewID : vl) {
          views[viewID].viewPoints.push_back(newPoints.size());
        }
        newPoints.push_back(newSP);
      }
    }

    origPoints.clear();
    points.swap(origPoints);
    points.swap(newPoints);
    std::cerr << "Number of points = " << points.size() << std::endl;
  } // voxelGridFilter

  Eigen::MatrixXf Domset::getSimilarityMatrix(std::map<int,int>& xId2vId) {
    std::cout << "Generating Similarity Matrix "<< std::endl;
    const int numC = xId2vId.size();
    const int numP = points.size();
    if(numC == 0 || numP == 0) {
      std::cerr << "Invalid Data\n";
      exit(0);
    }
    float medianDist =  getDistanceMedian(xId2vId);
    std::cout << "Median dists = " << medianDist << std::endl;
    Eigen::MatrixXf simMat;
    simMat.resize(numC, numC);
#pragma omp parallel for collapse(2)
    for( int xId1 = 0; xId1 < numC; xId1++) {
      for( int xId2 = 0; xId2 < numC; xId2++) {
        const int vId1 = xId2vId[xId1];
        const int vId2 = xId2vId[xId2];
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

  float Domset::computeViewDistance(const int& vId1, const int& vId2, const float& medianDist) {
    if(vId1 == vId2) return 1;
    float vd = viewDists(vId1, vId2);
    float dm = 1 + exp(- (vd - medianDist) / medianDist);
    return 1/dm;
  }
  float Domset::getDistanceMedian(std::map<int,int> & xId2vId) {
    std::cout << "Finding Distance Median\n";
    const int numC = xId2vId.size();
    if(numC == 0) {
      std::cerr << "No Views initialized \n";
      exit(0);
    }

    std::vector<float> dists;
    // float totalDist = 0;
    for(int i = 0; i < numC; i++) {
      const auto v1 = xId2vId[i];
      for(int j = 0; j < numC; j++ ) {
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
    const int numC = views.size();
    if(numC == 0) {
      std::cerr << "No Views initialized \n";
      exit(0);
    }
    viewDists.resize(numC, numC);
    for(int i = 0; i < numC; i++) {
      const auto v1 = views[i];
      for(int j = 0; j < numC; j++ ) {
        const auto v2 = views[j];
        float dist = (v1.trans - v2.trans).norm();
        viewDists(i,j) = dist;
      }
    }
  }
  void Domset::findCommonPoints(const View& v1, const View& v2,
      std::vector<int>& commonPoints){
    commonPoints.clear();
    const int numVP1 = v1.viewPoints.size();
    const int numVP2 = v2.viewPoints.size();

#pragma omp parallel for collapse(2)
    for(int i =0; i <numVP1; i++) {
      for(int j = 0; j < numVP2; j++){
        const int vId1 = v1.viewPoints[i];
        const auto vId2 = v2.viewPoints[j];
        if(vId1 == vId2) {
#pragma omp critical(updateCommonPoints)
          commonPoints.push_back(vId2);
        }
      }
    }
  } // findCommonPoints

  float Domset::computeViewSimilaity(const View& v1, const View& v2) {
    std::vector<int> commonPoints;
    findCommonPoints(v1, v2, commonPoints);
    const int numCP = commonPoints.size();

    float w =0;
#pragma omp parallel for
    for( int p=0; p < numCP; p++){
      const auto pId = commonPoints[p];
      //for( const auto pId : commonPoints ){
      Eigen::Vector3f c1 = v1.trans - points[pId].pos;
      c1.normalize();
      Eigen::Vector3f c2 = v2.trans - points[pId].pos;
      c2.normalize();
      const float angle = acos(c1.dot(c2));
      const float expAngle = exp(- ( angle * angle) / kAngleSigma_2);
      //std::cerr << angle <<  " = " << expAngle << std::endl;
#pragma omp atomic
      w += expAngle;
    }
    float ans = w / numCP;
    return (ans != ans)? 0 : ans;
    } // computeViewSimilaity

    void Domset::computeClustersAP(std::map<int, int>& xId2vId,
        std::vector<std::vector<int> >& clusters) {
      const int numX = xId2vId.size();
      if(numX == 0) {
        std::cout << "Invalid map size\n";
        exit(0);
      }

      Eigen::MatrixXf S = getSimilarityMatrix(xId2vId);
      Eigen::MatrixXf R(numX, numX);
      R.setConstant(0);
      Eigen::MatrixXf A(numX, numX);
      A.setConstant(0);

      for(int m=0; m<kNumIter; m++) {

        //update responsibility
#pragma omp parallel for collapse(2)
        for(int i=0; i<numX; i++) {
          for(int k=0; k<numX; k++) {
            float max1 = std::numeric_limits<float>::min();
            float max2 = std::numeric_limits<float>::min();

            for(int kk=0; kk<k; kk++) {
              if(S(i,kk) +  A(i,kk) >max1)
                max1 = S(i,kk) +A(i,kk);
            }
            for(int kk=k+1; kk<numX; kk++) {
              if(S(i,kk) +A(i,kk) >max2)
                max2 = S(i,kk) +A(i,kk);
            }
            float max = std::max(max1,max2);
            R(i,k) = (1-lambda)*(S(i,k) - max) + lambda*R(i,k) ;
          }
        }

        //update availability
#pragma omp parallel for collapse(2)
        for(int i=0; i<numX; i++) {
          for(int k=0; k<numX; k++) {
            if(i==k) continue;
            const int maxik = std::max(i, k);
            const int minik = std::min(i, k);
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;
            float r1, r2, r3;
            for(int ii=0; ii<minik; ii++) {
              r1 = R(ii,k);
              //sum1 += std::max(0.0f, r1);
              if(r1 > 0.0f)
                sum1 += r1;
            }
            for(int ii=minik+1; ii<maxik; ii++) {
              r2 = R(ii,k);
              // sum2 += std::max(0.0f, r2);
              if(r2 > 0.0f)
                sum2 += r2;
            }
            for(int ii=maxik+1; ii<numX; ii++) {
              r3 = R(ii,k);
              // sum3 += std::max(0.0f, r3);
              if(r3 > 0.0f)
                sum3 += r3;
            }
            float r = R(k,k) + sum1 + sum2 + sum3;
            A(i,k) = (1-lambda)*std::min(0.0f, r) + lambda*A(i,k);
          }
        }
      }
#pragma omp parallel for
      for(int i=0; i<numX; i++) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float r1, r2;
        for(int ii=0; ii<i; ii++) {
          r1 = R(ii,i);
          //sum1 += std::max(0.0f, r1);
          if(r1 > 0.0f)
            sum1 += r1;
        }
        for(int ii=i+1; ii<numX; ii++) {
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
      std::set<int> centers;
      std::map<int, std::vector<int>> clMap;
      int idxForI = 0;
      for(int i=0; i<numX; i++) {
        float maxSim = std::numeric_limits<float>::min();
        for(int j=0; j<numX; j++) {
          if (E(i,j)>maxSim) {
            maxSim = E(i,j);
            idxForI = j;
          }
        }
        centers.insert(idxForI);
      }

      for(auto const c : centers)
        clMap[c] = std::vector<int>();

      for(int i = 0; i < numX; i++ ) {
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
          const int vId1 = xId2vId[p1.first];
          if(p1.second.size() < kMinClusterSize) {
            float minDist = std::numeric_limits<float>::max();
            int minId = -1;
            for(const auto p2 : clMap) {
              if(p1.first == p2.first) continue;
              const int vId2 = xId2vId[p2.first];
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
        std::vector<int> cl;
        for(const auto i : p.second){
          cl.push_back(xId2vId[i]);
        }
        if(cl.size() > kMaxClusterSize) {
          // std::cout << "split " << p.first << " | " << p.second.size() << std::endl;
          auto it = cl.begin();
          while(true) {
            auto stop = it + kMaxClusterSize;
            if(stop < cl.end()) {
              auto tmp = std::vector<int>(it, stop);
              it = stop;
              std::sort(tmp.begin(), tmp.end());
              clusters.push_back(tmp);
            } else {
              std::vector<int> tmp;
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

    void Domset::clusterViews(
        const int& minClusterSize, const int& maxClusterSize){
      std::cout << "[ Clustering Views ] "<< std::endl;
      const int numC = views.size();
      kMinClusterSize = minClusterSize;
      kMaxClusterSize = maxClusterSize;

      std::map<int,int> xId2vId;
      for(int i =0; i <numC; i++) {
        xId2vId[i] = i;
      }
      std::vector<std::vector<int> > clusters;
      computeClustersAP(xId2vId, clusters);

      std::stringstream ss;
      ss << "Clusters : \n";
      for(const auto cl : clusters){
        ss << cl.size() << " : ";
        for(const auto id : cl) {
          ss << id << " ";
        }
        ss << "\n\n";
      }
      std::cout << "Number of clusters = " << clusters.size() << std::endl;
      std::cout << ss.str();

      finalClusters.swap(clusters);
    }

    void Domset::exportToPLY(const std::string& plyFilename) {
      std::stringstream plys;
      plys    << "ply\n"
        << "format ascii 1.0\n";

      int totalViews = 0;
      for(const auto cl : finalClusters)
        totalViews += cl.size();

      const int numPts = origPoints.size();
      plys    << "element vertex " << totalViews + numPts << std::endl
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

      for(const auto pt : origPoints) {
        Eigen::Vector3f pos = pt.pos.transpose();

        plys << pos(0) << " " << pos(1) << " " << pos(2)
          << " 255 255 255" << std::endl;
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
