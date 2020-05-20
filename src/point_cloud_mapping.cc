/*

	May 13, 2020, He Zhang, hzhang8@vcu.edu

	compute point cloud map using the generated trajectory and rgbd images 

*/

#include <cstdio>
#include <ctime>
#include <csignal>

#include <memory>
#include <limits>
#include <vector>
#include <string>

#include <boost/filesystem.hpp>

#include <Eigen/Core>

#include <sophus/se3.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PolygonMesh.h>

#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include "vtk_viewer.h"

using namespace std;

struct trajItem
{
  long long timestamp; // sequence id 
  // double timestamp; 
  // string timestamp; 
  double px, py, pz; // euler angle roll, pitch, yaw; 
  double qx, qy, qz, qw; // quaternion
};

typedef pcl::PointCloud<pcl::PointXYZRGB>  CloudL;  
typedef typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudLPtr;

bool readImgFiles(std::vector<long long>& vt, std::vector<string>& v_rgb, std::vector<string>& v_dpt);
bool readTraj(std::string f, vector<struct trajItem>& t); 
void generatePointCloud(cv::Mat& rgb, cv::Mat& depth, int skip, pcl::PointCloud<pcl::PointXYZRGB>& pc, int* rect=0);
void transformPointCloud(trajItem& pose_i, pcl::PointCloud<pcl::PointXYZRGB>& pc_loc, pcl::PointCloud<pcl::PointXYZRGB>& pc_glo);
void set_param(); 
void do_it(); 

template<typename PointT>
void filterPointCloud(float _voxel_size, typename pcl::PointCloud<PointT>::Ptr& in,typename pcl::PointCloud<PointT>::Ptr& out)
{
  // voxelgrid filter
  pcl::VoxelGrid<PointT> vog;
  vog.setInputCloud(in); 
  vog.setLeafSize(_voxel_size, _voxel_size, _voxel_size);
  vog.filter(*out);
}


// TODO: make it as a parameter
Eigen::Matrix<double, 4, 4> Tu2c; //  = Eigen::Matrix<float, 4, 4>::Identity(); // specify it 
Eigen::Matrix<double, 3, 3> Ru2c; //  = Eigen::Matrix<float, 4, 4>::Identity(); // specify it 
Eigen::Vector3d tu2c; 

double CX, CY, FX, FY; 

// string img_path("/home/davidz/work/data/phone/room_5_11/2020-05-11T17-26-00/cam0/data"); 
// string img_fileindex("/home/davidz/work/data/phone/room_5_11/2020-05-11T17-26-00/cam0/data.csv"); 
// string pose_filename("/home/davidz/work/data/phone/room_5_11/2020-05-11T17-26-00/trajectory/data.csv"); 

// string img_path("/home/davidz/work/data/phone/ARkit_5_14/2020-05-14T22-53-56/cam0/data"); 
// string img_fileindex("/home/davidz/work/data/phone/ARkit_5_14/2020-05-14T22-53-56/cam0/data.csv"); 
// string pose_filename("/home/davidz/work/data/phone/ARkit_5_14/2020-05-14T22-53-56/trajectory/data.csv"); 

 string img_path("/home/davidz/work/data/phone/ARkit_5_14/2020-05-17T19-20-59/cam0/data"); 
 string img_fileindex("/home/davidz/work/data/phone/ARkit_5_14/2020-05-17T19-20-59/cam0/data.csv"); 
 string pose_filename("/home/davidz/work/data/phone/ARkit_5_14/2020-05-17T19-20-59/trajectory/data.csv"); 

// string img_path("/home/davidz/work/data/euroc/V1_01_easy/mav0/cam0/data"); 
// string img_fileindex("/home/davidz/work/data/euroc/V1_01_easy/mav0/cam0/data.csv"); 
// string pose_filename("/home/davidz/work/data/euroc/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"); 

int main(int argc, char* argv[])
{
   	ros::init(argc, argv, "point_cloud_mapping");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug); // Info

    ROS_WARN("usage: ./point_cloud_mapping image_path image_file_index pose_file_name"); 

    n.param("image_path", img_path, img_path); 
    n.param("image_file_index", img_fileindex, img_fileindex); 
    n.param("pose_file_name", pose_filename, pose_filename); 

    if(argc >= 2){
    	img_path = argv[1];
    }
    if(argc >= 3){
    	img_fileindex = argv[2]; 
    }
    if(argc >= 4){
    	pose_filename = argv[3]; 
    }

    set_param(); 

    do_it(); 

    return 0; 

}


void do_it()
{
	// 1. read pose file 
	// generate a global point cloud 
  	vector<trajItem> v_traj; 
  	if(!readTraj(pose_filename, v_traj)){
  		ROS_ERROR("point_cloud_mapping.cc: failed to read pose file %s", pose_filename.c_str()); 
    	return ; 
  	}

  	// 2. read rgb and depth file names 
  	std::vector<string> v_rgb_files;
  	std::vector<string> v_dpt_files;
  	std::vector<long long> v_timestamp;
  	if(!readImgFiles(v_timestamp, v_rgb_files, v_dpt_files)){
  		ROS_ERROR("point_cloud_mapping.cc: failed to read image files %s", img_fileindex.c_str()); 
      return ;
  	}

  	// 3. main loop, generate point cloud at poses 
    int j = 0; 
    CloudLPtr global_pc(new CloudL); 

    int cnt = 0; 

    for(int i=0; ros::ok() && i<v_traj.size(); i++){

      trajItem& pose_i = v_traj[i]; 

      for(j=0;j<v_timestamp.size(); j++){

        if(v_timestamp[j] == pose_i.timestamp){

          cout <<"point_cloud_mapping.cc: found timestamp: "<<pose_i.timestamp<<endl; 

          // show the result 
          string rgb_file = img_path + "/" + v_rgb_files[j]; 
          string dpt_file = img_path + "/" + v_dpt_files[j]; 
          cv::Mat3b rgb = cv::imread(rgb_file.c_str(), cv::IMREAD_COLOR);
          cv::Mat_<uint16_t> dpt = cv::imread(dpt_file.c_str(), cv::IMREAD_ANYDEPTH);
          if(dpt.empty()) {
            cout<<"dpt_file is missing: "<<dpt_file<<endl; 
            break;  
          }

          // cv::imshow("rgb", rgb); 
          // cv::waitKey(1000); 
          // cv::imshow("dpt", dpt); 
          // cv::waitKey(1000); 

          // generate point cloud 
          CloudLPtr pci(new CloudL);
          CloudLPtr pwi(new CloudL); 
          generatePointCloud(rgb, dpt, 1, *pci); 

          // transform into global coordinate 
          transformPointCloud(pose_i, *pci, *pwi); 

          // voxel filtering 
          *global_pc += *pwi; 
          {
           //  CloudLPtr tmp(new CloudL); 
           // filterPointCloud<pcl::PointXYZRGB>(0.01, global_pc, tmp); 
           // global_pc.swap(tmp);

            if((++cnt)%5 == 0){
              if(global_pc->points.size() > 0){
                CloudLPtr tmp(new CloudL); 
                filterPointCloud<pcl::PointXYZRGB>(0.01, global_pc, tmp); 
                // save 
                stringstream ss; 
                ss <<"./tmp_pc/"<<cnt<<".pcd"; 
                pcl::io::savePCDFile(ss.str(), *tmp); 
                // CloudLPtr tmp(new CloudL); 
                // global_pc.swap(tmp);
                tmp.reset(new CloudL);
                global_pc.swap(tmp);
              }
            }
          }
          break; 
        }

        // if(cnt > 48)
         // break;

      }
    }

  // save pci 
  //if(global_pc->points.size() > 0)
    // pcl::io::savePCDFile("global_pc_map.pcd", *global_pc); 

  // see it
  // CVTKViewer<pcl::PointXYZRGBA> v;
  CVTKViewer<pcl::PointXYZRGB> v;

  v.getViewer()->addCoordinateSystem(1.0, 0, 0, 0); 
  v.addPointCloud(global_pc, "PC in world"); 
  while(!v.stopped() && ros::ok())
  {
    v.runOnce(); 
    usleep(100*1000); 
  }

  return ; 

}

void set_param()
{
  // TODO: use config file set parameters 
  // iphone SE
  /*Tu2c<< 0., -1, 0, 0.092, 
        -1, 0, 0, 0.01,
        0, -0, -1, 0.,
        0, 0, 0, 1.;

  Ru2c = Tu2c.block<3,3>(0,0);
  tu2c = Tu2c.block<3,1>(0,3); 

  CX = 320.6716; 
  CY = 227.0487;
  FX = 515.8693;
  FY = 515.8693;*/

  // ARKit 
  Tu2c<< 1., 0, 0, 0., 
        0, -1, 0, 0.,
        0, 0, -1, 0.,
        0, 0, 0, 1.;

  Ru2c = Tu2c.block<3,3>(0,0);
  tu2c = Tu2c.block<3,1>(0,3); 

  CX = 953.742798; 
  CY = 700.477966;
  FX = 1575.418945;
  FY = 1575.418945;

  // euroc dataset
/*  Tu2c<< 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0;

  Ru2c = Tu2c.block<3,3>(0,0);
  tu2c = Tu2c.block<3,1>(0,3); 

  CX = 367.215; 
  CY = 248.375;
  FX = 458.654;
  FY = 457.296;
*/
}

void transformPointCloud(trajItem& pi, pcl::PointCloud<pcl::PointXYZRGB>& pc_loc, pcl::PointCloud<pcl::PointXYZRGB>& pc_glo)
{
  // 
  Eigen::Quaterniond qwu(pi.qw, pi.qx, pi.qy, pi.qz); 
  Eigen::Vector3d twu(pi.px, pi.py, pi.pz); 

  pc_glo.points.resize(pc_loc.points.size()); 
  for(int i=0; i<pc_loc.points.size(); i++){
    pcl::PointXYZRGB& pt_loc = pc_loc.points[i]; 

    Eigen::Vector3d pt_loc_cam(pt_loc.x, pt_loc.y, pt_loc.z);
    Eigen::Vector3d pt_loc_imu = Ru2c * pt_loc_cam  + tu2c; 
    Eigen::Vector3d pt_loc_w = qwu * pt_loc_imu + twu; 

    pcl::PointXYZRGB& pt_glo = pc_glo.points[i]; 
    pt_glo = pt_loc; 
    pt_glo.x = pt_loc_w(0); 
    pt_glo.y = pt_loc_w(1); 
    pt_glo.z = pt_loc_w(2); 
  }
  pc_glo.height = 1; 
  pc_glo.width = pc_loc.points.size(); 
  pc_glo.is_dense = true; 
  return ; 
}

// void generatePointCloud(cv::Mat& rgb, cv::Mat& depth, int skip, vector<float>& pts, vector<unsigned char>& color, int * rect)
void generatePointCloud(cv::Mat& rgb, cv::Mat& depth, int skip, pcl::PointCloud<pcl::PointXYZRGB>& pc, int* rect)
{
  double z; 
  double px, py, pz; 
  int N = (rgb.rows/skip)*(rgb.cols/skip); 
  // pts.reserve(N*3); 
  // color.reserve(N*3); 

  unsigned char r, g, b; 
  int pixel_data_size = 3; 
  if(rgb.type() == CV_8UC1)
  {
    pixel_data_size = 1; 
  }
  
  int color_idx; 
  char red_idx = 2, green_idx =1, blue_idx = 0;

  int sv, su, ev, eu; 
  // Point pt; 
  pcl::PointXYZRGB pt;

  if(rect == 0)
  {
    sv = su = 0; 
    ev = rgb.rows; 
    eu = rgb.cols; 
  }else{
    su = rect[0]; sv = rect[1]; eu = rect[2]; ev = rect[3]; 
    su = su < 0 ? 0 : su;   sv = sv < 0 ? 0 : sv; 
    eu = eu <= rgb.cols ? eu : rgb.cols; ev = ev <= rgb.rows ? ev : rgb.rows;
  }

  for(int v = sv; v<ev; v+=skip)
  for(int u = su; u<eu; u+=skip)
  {
    z = depth.at<unsigned short>((v), (u))*0.001;
    if(std::isnan(z) || z <= 0.0 || z >= 4.0) continue; 
    // if(std::isnan(z) || z <= 0.0 || z >= 2.4) continue; 
    // m_cam_model.convertUVZ2XYZ(u, v, z, px, py, pz); 
    px = (u - CX) / FX * z; 
    py = (v - CY) / FY * z; 
    pz = z;
    pt.x = px;  pt.y = py;  pt.z = pz; 
    // pts.push_back(px);  pts.push_back(py);  pts.push_back(pz); 
    color_idx = (v*rgb.cols + u)*pixel_data_size;
    if(pixel_data_size == 3)
    {
      r = rgb.at<uint8_t>(color_idx + red_idx);
      g = rgb.at<uint8_t>(color_idx + green_idx); 
      b = rgb.at<uint8_t>(color_idx + blue_idx);
    }else{
      r = g = b = rgb.at<uint8_t>(color_idx); 
    }
    pt.r = r; pt.g = g; pt.b = b; 
    // color.push_back(r); color.push_back(g); color.push_back(b); 
    pc.points.push_back(pt); 
  }
  return ;
}

bool readImgFiles(std::vector<long long>& vt, std::vector<string>& v_rgb, std::vector<string>& v_dpt)
{
	ifstream inf(img_fileindex.c_str()); 
	if(!inf.is_open()){
    	cout<<"failed to open file : "<<img_fileindex<<endl;
    	return false;
  	}
  	
  	vt.clear(); 
  	v_rgb.clear(); 
  	v_dpt.clear(); 

    string s; 
    getline(inf, s);
  	while(!inf.eof()){
  		string s; 
  		getline(inf, s); 
  		if(s.empty())
  			break; 

  		std::string::size_type sz = 0;
  		long long timestamp = std::stoll(s, &sz, 0); 
  		string rgb_s = s.substr(sz+1); // skip comma 
      // rgb_s = rgb_s.substr(0, rgb_s.size()-2); // remove \r \n 
      // rgb_s = rgb_s.substr(0, rgb_s.size()-1); // remove \r \n 

  		// std::size_t found = rgb_s.find_last_of("/\\");
  		// string path = rgb_s.substr(0, found);
  		// string file_name = rgb_s.substr(found+1);  
  		// string dpt_s = path + "/dpt_" + file_name; 
      string dpt_s = "dpt_" + rgb_s; 
      // string dpt_s = "feat_dpt_" + rgb_s; 
  		// printf("point_cloud_mapping.cc: timestamp %ld, rgb_s: %s dpt_s: %s\n", timestamp, rgb_s.c_str(), dpt_s.c_str()); 
      // printf("point_cloud_mapping.cc: timestamp %ld, rgb_s: %s dpt_s: \n", timestamp, rgb_s.c_str()); 
      // printf("point_cloud_mapping.cc: timestamp %ld \n", timestamp); 
      cout<<"timestamp: "<<timestamp<<" rgb_s: "<<rgb_s<<" dpt_s: "<<dpt_s<<endl;

  		vt.push_back(timestamp); 
  		v_rgb.push_back(rgb_s); 
  		v_dpt.push_back(dpt_s);  		

//  		if(vt.size() > 10)
//  			break; 
  	}

  	ROS_DEBUG("point_cloud_mapping.cc: load %d rgbd_files!",v_rgb.size()); 

  	return v_rgb.size() > 0; 
}

bool readTraj(std::string f, vector<struct trajItem>& t)
{
  ifstream inf(f.c_str()); 
  if(!inf.is_open()){
    cout<<"failed to open file : "<<f<<endl;
    return false;
  }

  char buf[4096]; 
  inf.getline(buf, 4096);
  // char b[21]={0}; 
  while(inf.getline(buf, 4096)){
    trajItem ti; 
    sscanf(buf, "%ld, %lf, %lf, %lf, %lf, %lf, %lf, %lf", &ti.timestamp, &ti.px, &ti.py, &ti.pz, &ti.qw, &ti.qx, &ti.qy, &ti.qz); 
    // ti.timestamp = string(b); 
    t.push_back(ti); 
    if(t.size() < 10){
      printf("read %ld %f %f %f %f %f %f %f\n", ti.timestamp, ti.px, ti.py, ti.pz, ti.qw, ti.qx, ti.qy, ti.qz);
    }
  }

  cout << " read "<<t.size()<<" trajectory items"<<endl;
  return true;
}