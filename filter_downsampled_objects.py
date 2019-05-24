import pcl
from random import randint 
import struct
import hdbscan
import numpy as np

get_color_list=[]

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

# Use RANSAC planse segmentation to separate plane and not plane points
# Returns inliers (plane) and outliers (not plane)
def do_ransac_plane_segmentation(point_cloud, max_distance = 0.01):

  segmenter = point_cloud.make_segmenter()

  segmenter.set_model_type(pcl.SACMODEL_PLANE)
  segmenter.set_method_type(pcl.SAC_RANSAC)
  segmenter.set_distance_threshold(max_distance)

  #obtain inlier indices and model coefficients
  inlier_indices, _ = segmenter.segment()

  inliers = point_cloud.extract(inlier_indices, negative = False)
  outliers = point_cloud.extract(inlier_indices, negative = True)

  return inliers,outliers

#Remove the intensity of the pointcloud data for clustring
def XYZRGB_to_XYZ(XYZRGB_cloud):

  XYZ_cloud = pcl.PointCloud()
  points_list = []

  for data in XYZRGB_cloud:
    points_list.append([data[0], data[1], data[2]])

  XYZ_cloud.from_list(points_list)
  return XYZ_cloud

def hdbscan_clusters(cloud):
  clusterer = hdbscan.HDBSCAN(metric='euclidean',min_cluster_size=100, gen_min_span_tree=True)
  clusterer.fit(np.asarray(cloud.to_list()))

  all_clusters=[]

  for i in range(clusterer.labels_.max()+1):
    all_clusters.append([ind for ind,j in enumerate(clusterer.labels_) if j==i])
  
  print('Total Number of clusters is :' +str(len(all_clusters)))

  return all_clusters

#Getting the number of clusters
def get_clusters(cloud, tolerance, min_size, max_size):

  tree = cloud.make_kdtree()
  extraction_object = cloud.make_EuclideanClusterExtraction()

  extraction_object.set_ClusterTolerance(tolerance)
  extraction_object.set_MinClusterSize(min_size)
  extraction_object.set_MaxClusterSize(max_size)
  extraction_object.set_SearchMethod(tree)

  # Get clusters of indices for each cluster of points, each clusterbelongs to the same object
  # 'clusters' is effectively a list of lists, with each list containing indices of the cloud
  clusters = extraction_object.Extract()

  print('Total Number of clusters is :' +str(len(clusters)))

  return clusters

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

def get_colored_clusters(clusters, cloud):

  global get_color_list  
  # Get a random unique colors import structfor each object
  number_of_clusters = len(clusters)

  for i in range(len(get_color_list),number_of_clusters):
    color = [randint(0, 255),randint(0,  255),randint(0, 255)]
    float_rgb=get_floatrgb(color)
    get_color_list.append(float_rgb)

  #assert len(get_color_list)==number_of_clusters

  colored_points = []
  total=0

  # Assign a color for each point
  # Points with the same color belong to the same cluster
  for cluster_id, cluster in enumerate(clusters):
    color = get_color_list[cluster_id]
    for i in cluster:
      x, y, z = cloud[i][0], cloud[i][1], cloud[i][2]
      colored_points.append([x, y, z, color])
    cluster_centroid=colored_points[total:total+len(cluster)]
    centroid_point=centroid_points(list(map(lambda x:x[0:3],cluster_centroid)))
    colored_points.append(centroid_point)
    total+=len(cluster)
  return colored_points

##################################################################################
# This pipeline separates the objects in the table from the given scene

# Load the point cloud in memory
cloud = pcl.load_XYZRGB('point_clouds/tabletop.pcd')

# Downsample the cloud as high resolution which comes with a computation cost
downsampled_cloud = do_voxel_grid_filter(point_cloud = cloud, LEAF_SIZE = 0.01)

# Get only information in our region of interest, as we don't care about the other parts
filtered_cloud = do_passthrough_filter(point_cloud = downsampled_cloud, 
                                    name_axis = 'z', min_axis = 0.6, max_axis = 1.1)

# Separate the table from everything else
table_cloud, objects_cloud = do_ransac_plane_segmentation(filtered_cloud, max_distance = 0.01)

#get the colorless_cloud data for clustering
colorless_cloud = XYZRGB_to_XYZ(objects_cloud)

#DBSCAN Euclidean clustering
clusters = get_clusters(colorless_cloud, tolerance = 0.05, min_size = 100, max_size = 1500)
colored_points = get_colored_clusters(clusters, colorless_cloud)
clusters_cloud = pcl.PointCloud_PointXYZRGB()
clusters_cloud.from_list(colored_points)
pcl.save(clusters_cloud,'segmenatation.pcd')

#HDBSCAN Euclidean clustering
hdbscan_cluster=hdbscan_clusters(colorless_cloud)
colored_points = get_colored_clusters(hdbscan_cluster, colorless_cloud)
clusters_cloud = pcl.PointCloud_PointXYZRGB()
clusters_cloud.from_list(colored_points)
pcl.save(clusters_cloud,'segmenatation_hdbscan.pcd')