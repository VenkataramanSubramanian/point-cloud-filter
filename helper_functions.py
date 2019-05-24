import pcl
import struct
import numpy as np

# Returns Downsampled version of a point cloud
# The bigger the leaf size the less information retained
def do_voxel_grid_filter(point_cloud, LEAF_SIZE = 0.01):
  voxel_filter = point_cloud.make_voxel_grid_filter()
  voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE) 
  return voxel_filter.filter()

# Returns only the point cloud information at a specific range of a specific axis
def do_passthrough_filter(point_cloud, name_axis = 'z', min_axis = 0.6, max_axis = 1.1):
  pass_filter = point_cloud.make_passthrough_filter()
  pass_filter.set_filter_field_name(name_axis)
  pass_filter.set_filter_limits(min_axis, max_axis)
  return pass_filter.filter()

#Remove the intensity of the pointcloud data for clustring
def XYZRGB_to_XYZ(XYZRGB_cloud):

  XYZ_cloud = pcl.PointCloud()
  points_list = []

  for data in XYZRGB_cloud:
    points_list.append([data[0], data[1], data[2]])

  XYZ_cloud.from_list(points_list)
  return XYZ_cloud  

def get_floatrgb(color):
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])
    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb

def centroid_points(cluster_centroid):
  
  centroid_point = np.mean(cluster_centroid, axis=0)
  centroid_point=np.append(centroid_point,get_floatrgb([255,255,255]))

  return centroid_point