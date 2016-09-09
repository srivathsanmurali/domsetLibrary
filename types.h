#ifndef _NOMOKO_TYPES_H_
#define _NOMOKO_TYPES_H_

namespace nomoko {
  /* breif:
     Contains the camera parameters
     currently contains only a intrinsic matrix for a pinhole camera
     */
  struct Camera {
    Eigen::Matrix3f K;
    unsigned int width;
    unsigned int height;
  }; // struct Camera

  /* breif:
     Contains the information for each view;
     rot       -> Rotation matrix for the pose
     trans     -> Translation vector for the pose
     cameraID  -> index for the camera used of the vector of Cameras
     filepath  -> filepath to the image
     */
  struct View {
    Eigen::Matrix3f rot;
    Eigen::Vector3f trans;
    unsigned int cameraId;
    std::string filename;
    std::vector<unsigned int> viewPoints;
  }; // struct View

  /* breif:
     Contains the information for each point in the sparse point cloud
     pos       -> 3D position of the point
     color     -> RGB values for the point
     viewList  -> List of views that see this point
     */
  struct Point {
    Eigen::Vector3f pos;
    std::vector<unsigned int> viewList;
  };
} // namespace nomoko
#endif // _NOMOKO_TYPES_H_
