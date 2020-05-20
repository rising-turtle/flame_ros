/*

	May 20, 2020, He Zhang, hzhang8@vcu.edu

	compute ply map using the input pcds  

*/

#include <cstdio>
#include <ctime>
#include <csignal>

#include <memory>
#include <limits>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PolygonMesh.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include "vtk_viewer.h"

using namespace std;

typedef pcl::PointCloud<pcl::PointXYZRGB>  CloudL;  
typedef typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudLPtr;


template<typename PointT>
void filterPointCloud(float _voxel_size, typename pcl::PointCloud<PointT>::Ptr& in,typename pcl::PointCloud<PointT>::Ptr& out)
{
  // voxelgrid filter
  pcl::VoxelGrid<PointT> vog;
  vog.setInputCloud(in); 
  vog.setLeafSize(_voxel_size, _voxel_size, _voxel_size);
  vog.filter(*out);
}


template<typename PointT>
void cluster_filterPointCloud(float cluster_dis_thresh, typename pcl::PointCloud<PointT>::Ptr& in,typename pcl::PointCloud<PointT>::Ptr& out)
{
  pcl::EuclideanClusterExtraction<PointT> ec;

   // Creating the KdTree object for the search method of the extraction
  typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  tree->setInputCloud (in);
  std::vector<pcl::PointIndices> cluster_indices;
  ec.setClusterTolerance (cluster_dis_thresh); // 2cm
  ec.setMinClusterSize (5000);
  ec.setMaxClusterSize (2500000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (in);
  ec.extract (cluster_indices);

  // out->points.reserve(cluster_indices.size ());
  // for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  out->points.clear(); 
  for(int i=0; i<cluster_indices.size(); i++)
  {
    pcl::PointIndices& vi = cluster_indices[i]; 

    for(int j=0; j<vi.indices.size(); j++){
      out->points.push_back(in->points[vi.indices[j]]); 
    }
  }

  out->height = 1;
  out->width = out->points.size(); 
  out->is_dense = true; 

  cout<<"pcd_2_ply.cpp: before filter points: "<<in->points.size()<<" after filter points: "<<out->points.size()<<endl;
  return ; 
}


void headerPLY(std::ofstream& ouf, int vertex_number);

int main(int argc, char* argv[])
{
   	ros::init(argc, argv, "pcd_2_ply");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug); // Info  

    vector<string> pcd_names; 

    for(int i=1; i<argc; i++){
      pcd_names.push_back(string(argv[i]));
    }
    if(pcd_names.size() <= 0){
      ROS_WARN("usage: ./pcd_2_ply [*.pcd]"); 
      return 0; 
    }

    // merge point cloud 
    CloudLPtr global_pc(new CloudL); 
    for(int i=0; i<pcd_names.size(); i++){
        CloudLPtr tmp(new CloudL); 
        pcl::io::loadPCDFile(pcd_names[i].c_str(), *tmp); 
        {
          // cluster filter 
          CloudLPtr tmp2(new CloudL); 
          cluster_filterPointCloud<pcl::PointXYZRGB>(0.02, tmp, tmp2);
          tmp.swap(tmp2); 
        }

        *global_pc += *tmp; 
        // filter 
        filterPointCloud<pcl::PointXYZRGB>(0.01, global_pc, tmp); 
        global_pc.swap(tmp);
    }

    // save pcd and save ply 
    pcl::io::savePCDFile("output_global_pc.pcd", *global_pc); 

    ofstream ouf("output_global_pc.ply"); 
    int vertex_number = global_pc->points.size(); 
    headerPLY(ouf, vertex_number); 
    for(int i=0; i<vertex_number; i++){
      pcl::PointXYZRGB& pt = global_pc->points[i]; 
      ouf << pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<(int)(pt.r)<<" "<<(int)(pt.g)<<" "<<(int)(pt.b)<<endl; 
    }
    ouf.flush(); 
    ouf.close(); 

    // show the point cloud 
    CVTKViewer<pcl::PointXYZRGB> v;

    v.getViewer()->addCoordinateSystem(1.0, 0, 0, 0); 
    v.addPointCloud(global_pc, "PC in world"); 
    while(!v.stopped() && ros::ok())
    {
      v.runOnce(); 
      usleep(100*1000); 
    } 

    return 0; 

}

void headerPLY(std::ofstream& ouf, int vertex_number)
{
  ouf << "ply"<<endl
    <<"format ascii 1.0"<<endl
    <<"element vertex "<<vertex_number<<endl
    <<"property float x"<<endl
    <<"property float y"<<endl
    <<"property float z"<<endl
    <<"property uchar red"<<endl
    <<"property uchar green"<<endl
    <<"property uchar blue"<<endl
    <<"end_header"<<endl;
}
