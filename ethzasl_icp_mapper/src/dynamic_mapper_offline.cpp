#include <fstream>

#include <boost/thread.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION >= 104100
#include <boost/thread/future.hpp>
#endif // BOOST_VERSION >=  104100

#include "pointmatcher/PointMatcher.h"
#include "pointmatcher/Timer.h"
#include "ros/console.h"
#include "ros/ros.h"

#include "nabo/nabo.h"

#include "pointmatcher_ros/get_params_from_server.h"
#include "pointmatcher_ros/point_cloud.h"
#include "pointmatcher_ros/ros_logger.h"
#include "pointmatcher_ros/transform.h"

#include "eigen_conversions/eigen_msg.h"
#include "nav_msgs/Odometry.h"
#include "tf/transform_broadcaster.h"
#include "tf/transform_listener.h"
#include "tf_conversions/tf_eigen.h"

// Services
#include "ethzasl_icp_mapper/CorrectPose.h"
#include "ethzasl_icp_mapper/GetBoundedMap.h" // FIXME: should that be moved to map_msgs?
#include "ethzasl_icp_mapper/GetMode.h"
#include "ethzasl_icp_mapper/LoadMap.h"
#include "ethzasl_icp_mapper/SetMode.h"
#include "map_msgs/GetPointMap.h"
#include "map_msgs/SaveMap.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "std_srvs/Empty.h"

#include <boost/foreach.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sstream>

using namespace std;
using namespace PointMatcherSupport;

typedef PointMatcher<float> PM;

class Mapper {
  typedef PM::DataPoints DP;
  typedef PM::Matches Matches;

  typedef typename Nabo::NearestNeighbourSearch<float> NNS;
  typedef typename NNS::SearchType NNSearchType;

  ros::NodeHandle &n;
  ros::NodeHandle &pn;

  // Subscribers
  ros::Subscriber scanSub;
  ros::Subscriber cloudSub;

  // Publishers
  ros::Publisher mapPub;
  ros::Publisher outlierPub;
  ros::Publisher odomPub;
  ros::Publisher robotPub;
  ros::Publisher odomErrorPub;

  // Services
  ros::ServiceServer getPointMapSrv;
  ros::ServiceServer saveMapSrv;
  ros::ServiceServer loadMapSrv;
  ros::ServiceServer resetSrv;
  ros::ServiceServer correctPoseSrv;
  ros::ServiceServer setModeSrv;
  ros::ServiceServer getModeSrv;
  ros::ServiceServer getBoundedMapSrv;
  ros::ServiceServer reloadAllYamlSrv;

  // Time
  ros::Time mapCreationTime;
  ros::Time lastPoinCloudTime;
  uint32_t lastPointCloudSeq;

  // libpointmatcher
  PM::DataPointsFilters inputFilters;
  PM::DataPointsFilters mapPreFilters;
  PM::DataPointsFilters mapPostFilters;
  PM::DataPoints *mapPointCloud;
  PM::ICPSequence icp;
  unique_ptr<PM::Transformation> transformation;
  PM::DataPointsFilter *radiusFilter;

// multi-threading mapper
#if BOOST_VERSION >= 104100
  typedef boost::packaged_task<PM::DataPoints *> MapBuildingTask;
  typedef boost::unique_future<PM::DataPoints *> MapBuildingFuture;
  boost::thread mapBuildingThread;
  MapBuildingTask mapBuildingTask;
  MapBuildingFuture mapBuildingFuture;
  bool mapBuildingInProgress;
#endif // BOOST_VERSION >= 104100
  bool processingNewCloud;

  // Parameters
  bool publishMapTf;
  bool useConstMotionModel;
  bool localizing;
  bool mapping;
  int minReadingPointCount;
  int minMapPointCount;
  int inputQueueSize;
  double minOverlap;
  double maxOverlapToMerge;
  double tfRefreshPeriod; //!< if set to zero, tf will be publish at the rate of
                          //!the incoming point cloud messages
  string sensorFrame;
  string odomFrame;
  string robotFrame;
  string mapFrame;
  string finalMapName; //!< name of the final vtk map
  string filePath;     // path to save all files - or else it saves in .ros

  const double mapElevation; // initial correction on z-axis //FIXME: handle the
                             // full matrix

  // Parameters for dynamic filtering
  const float
      priorStatic; //!< ratio. Prior to be static when a new point is added
  const float
      priorDyn; //!< ratio. Prior to be dynamic when a new point is added
  const float maxAngle; //!< in rad. Openning angle of a laser beam
  const float eps_a;    //!< ratio. Error proportional to the laser distance
  const float eps_d;    //!< in meter. Fix error on the laser distance
  const float alpha;    //!< ratio. Propability of staying static given that the
                        //!point was dynamic
  const float beta;   //!< ratio. Propability of staying dynamic given that the
                      //!point was static
  const float maxDyn; //!< ratio. Threshold for which a point will stay dynamic
  const float maxDistNewPoint; //!< in meter. Distance at which a new point will
                               //!be added in the global map.
  const float sensorMaxRange;  //!< in meter. Maximum reading distance of the
                               //!laser. Used to cut the global map before
                               //!matching.

  PM::TransformationParameters T_odom_to_map;
  PM::TransformationParameters T_robot_to_map;
  PM::TransformationParameters T_cutMap_to_map;
  boost::thread publishThread;
  boost::mutex publishLock;
  boost::mutex icpMapLock;
  ros::Time publishStamp;

  tf::TransformListener tfListener;
  tf::TransformBroadcaster tfBroadcaster;

  const float eps;

  std::string bag_filename;
  rosbag::Bag bag;
  bool bag_file_mode;
  int bag_file_stepping;
  int bag_file_step = 0;
  int bag_file_interleave_step = 0;
  int bag_file_interleave;
  int bag_file_save_interval;
  int bag_file_save_interval_count = 0;
  int bag_file_save_interval_step = 0;

  int bag_file_skip_sec;
  std::string velodyne_topic;

public:
  Mapper(ros::NodeHandle &n, ros::NodeHandle &pn);
  ~Mapper();

protected:
  void processCloud(unique_ptr<DP> cloud, const std::string &scannerFrame,
                    const ros::Time &stamp, uint32_t seq);
  void processNewMapIfAvailable();
  void setMap(DP *newPointCloud);
  DP *updateMap(DP *newPointCloud,
                const PM::TransformationParameters T_updatedScanner_to_map,
                bool mapExists);
  void waitForMapBuildingCompleted();
  void updateIcpMap(const DP *newMapPointCloud);

  void publishLoop(double publishPeriod);
  void publishTransform();
  void loadExternalParameters();

  // Services
  bool getPointMap(map_msgs::GetPointMap::Request &req,
                   map_msgs::GetPointMap::Response &res);
  bool saveMap(map_msgs::SaveMap::Request &req,
               map_msgs::SaveMap::Response &res);
  bool loadMap(ethzasl_icp_mapper::LoadMap::Request &req,
               ethzasl_icp_mapper::LoadMap::Response &res);
  bool reset(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);
  bool correctPose(ethzasl_icp_mapper::CorrectPose::Request &req,
                   ethzasl_icp_mapper::CorrectPose::Response &res);
  bool setMode(ethzasl_icp_mapper::SetMode::Request &req,
               ethzasl_icp_mapper::SetMode::Response &res);
  bool getMode(ethzasl_icp_mapper::GetMode::Request &req,
               ethzasl_icp_mapper::GetMode::Response &res);
  bool getBoundedMap(ethzasl_icp_mapper::GetBoundedMap::Request &req,
                     ethzasl_icp_mapper::GetBoundedMap::Response &res);
  bool reloadallYaml(std_srvs::Empty::Request &req,
                     std_srvs::Empty::Response &res);
};

Mapper::Mapper(ros::NodeHandle &n, ros::NodeHandle &pn)
    : n(n), pn(pn), mapPointCloud(0),
      transformation(
          PM::get().REG(Transformation).create("RigidTransformation")),
#if BOOST_VERSION >= 104100
      mapBuildingInProgress(false),
#endif // BOOST_VERSION >= 104100
      processingNewCloud(false),
      publishMapTf(getParam<bool>("publishMapTf", true)),
      useConstMotionModel(getParam<bool>("useConstMotionModel", false)),
      localizing(getParam<bool>("localizing", true)),
      mapping(getParam<bool>("mapping", true)),
      minReadingPointCount(getParam<int>("minReadingPointCount", 2000)),
      minMapPointCount(getParam<int>("minMapPointCount", 500)),
      inputQueueSize(getParam<int>("inputQueueSize", 10)),
      minOverlap(getParam<double>("minOverlap", 0.5)),
      maxOverlapToMerge(getParam<double>("maxOverlapToMerge", 0.9)),
      tfRefreshPeriod(getParam<double>("tfRefreshPeriod", 0.01)),
      sensorFrame(getParam<string>("sensor_frame", "velodyne")),
      odomFrame(getParam<string>("odom_frame", "odom")),
      robotFrame(getParam<string>("robot_frame", "base_link")),
      mapFrame(getParam<string>("map_frame", "map")),
      finalMapName(getParam<string>("finalMapName", "finalMap.vtk")),
      filePath(getParam<string>("filePath", "")),
      mapElevation(getParam<double>("mapElevation", 0)),
      priorStatic(getParam<double>("priorStatic", 0.5)),
      priorDyn(getParam<double>("priorDyn", 0.5)),
      maxAngle(getParam<double>("maxAngle", 0.02)),
      eps_a(getParam<double>("eps_a", 0.05)),
      eps_d(getParam<double>("eps_d", 0.02)),
      alpha(getParam<double>("alpha", 0.99)),
      beta(getParam<double>("beta", 0.99)),
      maxDyn(getParam<double>("maxDyn", 0.95)),
      maxDistNewPoint(pow(getParam<double>("maxDistNewPoint", 0.1), 2)),
      sensorMaxRange(getParam<double>("sensorMaxRange", 80.0)),
      T_odom_to_map(PM::TransformationParameters::Identity(4, 4)),
      T_robot_to_map(PM::TransformationParameters::Identity(4, 4)),
      T_cutMap_to_map(PM::TransformationParameters::Identity(4, 4)),
      publishStamp(ros::Time::now()), tfListener(ros::Duration(30)),
      eps(0.0001) {

  // Ensure proper states
  if (localizing == false)
    mapping = false;
  if (mapping == true)
    localizing = true;

  // set logger
  if (getParam<bool>("useROSLogger", false))
    PointMatcherSupport::setLogger(new PointMatcherSupport::ROSLogger);

  // Load all parameters stored in external files
  loadExternalParameters();

  PM::Parameters params;
  params["dim"] = "-1";
  params["maxDist"] = toParam(sensorMaxRange);

  radiusFilter = PM::get().DataPointsFilterRegistrar.create(
      "MaxDistDataPointsFilter", params);

  mapPub = n.advertise<sensor_msgs::PointCloud2>("point_map", 2, true);
  outlierPub = n.advertise<sensor_msgs::PointCloud2>("outliers", 2, true);
  odomPub = n.advertise<nav_msgs::Odometry>("icp_odom_sensor", 50, true);
  robotPub = n.advertise<nav_msgs::Odometry>("icp_odom_robot", 50, true);
  odomErrorPub = n.advertise<nav_msgs::Odometry>("icp_error_odom", 50, true);

  // service initializations
  getPointMapSrv =
      n.advertiseService("dynamic_point_map", &Mapper::getPointMap, this);
  saveMapSrv = pn.advertiseService("save_map", &Mapper::saveMap, this);
  loadMapSrv = pn.advertiseService("load_map", &Mapper::loadMap, this);
  resetSrv = pn.advertiseService("reset", &Mapper::reset, this);
  correctPoseSrv =
      pn.advertiseService("correct_pose", &Mapper::correctPose, this);
  setModeSrv = pn.advertiseService("set_mode", &Mapper::setMode, this);
  getModeSrv = pn.advertiseService("get_mode", &Mapper::getMode, this);
  getBoundedMapSrv =
      pn.advertiseService("get_bounded_map", &Mapper::getBoundedMap, this);
  reloadAllYamlSrv =
      pn.advertiseService("reload_all_yaml", &Mapper::reloadallYaml, this);

  // refreshing tf transform thread
  publishThread =
      boost::thread(boost::bind(&Mapper::publishLoop, this, tfRefreshPeriod));

  string mapFileName;
  if (ros::param::get("~mapFileName", mapFileName)) {
    ROS_INFO_STREAM("[MAP] Loading map from config file: " << mapFileName);
    ethzasl_icp_mapper::LoadMap::Request req;
    req.filename.data = mapFileName;
    ethzasl_icp_mapper::LoadMap::Response res;
    // always returns true
    this->loadMap(req, res);
  }

  // bag_file_mode
  if (bag_file_mode || true) {

    int percent_complete = 0, temp_percent_complete, complete_secs;

    ROS_INFO_STREAM("Loading data from " << bag_filename);
    rosbag::Bag bag;
    bag.open(bag_filename, rosbag::bagmode::Read);
    rosbag::View view_temp(bag);
    ros::Time startTime = view_temp.getBeginTime();
    startTime.sec += bag_file_skip_sec;
    rosbag::View view(bag, rosbag::TopicQuery(velodyne_topic), startTime);

    const double duration = (view.getEndTime() - startTime).toSec();
    ROS_INFO_STREAM("Total Duration: " << duration << " seconds ");
    for (const rosbag::MessageInstance &msg : view) {
      // ROS_INFO_STREAM("bag_file_mode waiting for semForLoadingBag ");
      if (!ros::ok())
        break;
      // sem_wait(&semForLoadingBag);
      if (msg.isType<sensor_msgs::PointCloud2>()) // if (m.getTopic() ==
                                                  // velodyne_packet_topic)
      {
        // ROS_INFO_STREAM("bag_file_mode decode");
        sensor_msgs::PointCloud2ConstPtr velo_point_ptr =
            msg.instantiate<sensor_msgs::PointCloud2>();
        if (velo_point_ptr != NULL) {
          if (bag_file_stepping > 0 && ++bag_file_step >= bag_file_stepping) {
            bag_file_step = 0;
            // cin.ignore().get(); //Pause Command for Linux Terminal
            // cin.ignore(numeric_limits<streamsize>::max(), '\n');
          }
          if (++bag_file_interleave_step >= bag_file_interleave) {
            bag_file_interleave_step = 0;
            sensor_msgs::PointCloud2 velo_point = *velo_point_ptr;
            ROS_INFO_STREAM(percent_complete
                            << " % complete, " << complete_secs
                            << "seconds of Duration: " << duration
                            << " seconds | stamp = "
                            << velo_point.header.stamp.toSec());
            if (velo_point.width < 100) // even 15% random sample should give
                                        // more than 15 points..  360 degree
                                        // could shound have much more in
                                        // general
            {
              bag_file_interleave_step = bag_file_interleave;
              ROS_WARN_STREAM(
                  "skipping current point due to very small number of points "
                  << velo_point.width
                  << " at stamp = " << velo_point.header.stamp.toSec());
              continue;
            }

            unique_ptr<DP> cloud(
                new DP(PointMatcher_ros::rosMsgToPointMatcherCloud<float>(
                    velo_point)));
            // processCloud(move(cloud), velo_point.header.frame_id,
            // velo_point.header.stamp, velo_point.header.seq);
            processCloud(move(cloud), velo_point.header.frame_id,
                         ros::Time::now(), velo_point.header.seq);
          } // else ROS_INFO_STREAM("skipping bag_file_interleave_step
            // "<<bag_file_interleave_step <<" bag_file_interleave
            // "<<bag_file_interleave);

          if (bag_file_save_interval > 0 &&
              ++bag_file_save_interval_step >= bag_file_save_interval) {
            bag_file_save_interval_step = 0;
            bag_file_save_interval_count++;
            mapPointCloud->save(filePath + "/" +
                                std::to_string(bag_file_save_interval_count) +
                                finalMapName);
          }
          ros::spinOnce();
        }
        complete_secs = (msg.getTime() - startTime).toSec();
        temp_percent_complete = complete_secs * 100 / duration;
        if (temp_percent_complete >= percent_complete + 1) {
          percent_complete = temp_percent_complete;
          ROS_INFO_STREAM_THROTTLE(1.0, percent_complete
                                            << " % complete, " << complete_secs
                                            << "seconds of Duration: "
                                            << duration << " seconds ");
        }
      }
    }
    mapPointCloud->save(filePath + "/" + finalMapName);
  }
}

Mapper::~Mapper() {
#if BOOST_VERSION >= 104100
  // wait for map-building thread
  if (mapBuildingInProgress) {
    mapBuildingFuture.wait();
    if (mapBuildingFuture.has_value())
      delete mapBuildingFuture.get();
  }
#endif // BOOST_VERSION >= 104100
  // wait for publish thread
  publishThread.join();
  // save point cloud
  /*if (mapPointCloud && mapping)
  {
          mapPointCloud->save(filePath + "/" + finalMapName);
          delete mapPointCloud;
  }*/
}

struct BoolSetter {
public:
  bool toSetValue;
  BoolSetter(bool &target, bool toSetValue)
      : toSetValue(toSetValue), target(target) {}
  ~BoolSetter() { target = toSetValue; }

protected:
  bool &target;
};

void Mapper::processCloud(unique_ptr<DP> newPointCloud,
                          const std::string &scannerFrame,
                          const ros::Time &stamp, uint32_t seq) {
  processingNewCloud = true;
  BoolSetter stopProcessingSetter(processingNewCloud, false);
  this->sensorFrame = scannerFrame;

  // if the future has completed, use the new map

  // TODO: investigate if we need that
  processNewMapIfAvailable();

  // IMPORTANT:  We need to receive the point clouds in local coordinates
  // (scanner or robot)
  timer t;

  cerr << "newPointCloud->getNbPoints()" << newPointCloud->getNbPoints()
       << endl;
  cerr << "newPointCloud->times.cols()" << newPointCloud->times.cols() << endl;

  // Convert point cloud
  size_t goodCount;
  try {
    goodCount = newPointCloud->features.cols();
  } catch (std::exception e) {
    ROS_WARN_STREAM("[ICP] Got part of clouds has "
                    << newPointCloud->times.cols() << "Columns. " << e.what());
  }
  if (goodCount == 0) {
    ROS_ERROR("[ICP] I found no good points in the cloud");
    return;
  }

  // Dimension of the point cloud, important since we handle 2D and 3D
  const int dimp1(newPointCloud->features.rows());

  if (!(newPointCloud->descriptorExists("stamps_Msec") &&
        newPointCloud->descriptorExists("stamps_sec") &&
        newPointCloud->descriptorExists("stamps_nsec"))) {
    const float Msec = round(stamp.sec / 1e6);
    const float sec = round(stamp.sec - Msec * 1e6);
    const float nsec = round(stamp.nsec);

    const PM::Matrix desc_Msec = PM::Matrix::Constant(1, goodCount, Msec);
    const PM::Matrix desc_sec = PM::Matrix::Constant(1, goodCount, sec);
    const PM::Matrix desc_nsec = PM::Matrix::Constant(1, goodCount, nsec);
    newPointCloud->addDescriptor("stamps_Msec", desc_Msec);
    newPointCloud->addDescriptor("stamps_sec", desc_sec);
    newPointCloud->addDescriptor("stamps_nsec", desc_nsec);

    // cout << "Adding time" << endl;
  }

  {
    timer t; // Print how long take the algo

    // Apply filters to incoming cloud, in scanner coordinates
    inputFilters.apply(*newPointCloud);

    ROS_INFO_STREAM("[ICP] Input filters took " << t.elapsed() << " [s]");
  }
  cerr << "newPointCloud->getNbPoints()" << newPointCloud->getNbPoints()
       << endl;
  cerr << "newPointCloud->times.cols()" << newPointCloud->times.cols() << endl;

  string reason;
  // Initialize the transformation to identity if empty
  if (!icp.hasMap()) {
    // we need to know the dimensionality of the point cloud to initialize
    // properly
    publishLock.lock();
    T_odom_to_map = PM::TransformationParameters::Identity(dimp1, dimp1);
    T_robot_to_map = PM::TransformationParameters::Identity(dimp1, dimp1);
    T_cutMap_to_map = PM::TransformationParameters::Identity(dimp1, dimp1);
    // ISER
    // T_odom_to_map(2,3) = mapElevation;
    publishLock.unlock();
  }

  ros::Time stamp_latest = ros::Time(0);

  // Fetch transformation from scanner to odom
  // Note: we don't need to wait for transform. It is already called in
  // transformListenerToEigenMatrix()

  // Fetch transformation from scanner to robot
  // Note: we don't need to wait for transform. It is already called in
  // transformListenerToEigenMatrix()

  PM::TransformationParameters T_robot_to_scanner;
  try {
    T_robot_to_scanner = PointMatcher_ros::eigenMatrixToDim<float>(
        PointMatcher_ros::transformListenerToEigenMatrix<float>(
            tfListener,
            scannerFrame, // to
            robotFrame,   // from
            stamp_latest),
        dimp1);
  } catch (tf::ExtrapolationException e) {
    ROS_ERROR_STREAM("Extrapolation Exception. stamp = "
                     << stamp << " now = " << ros::Time::now()
                     << " delta = " << ros::Time::now() - stamp << endl
                     << e.what());
    return;
  }

  // ROS_DEBUG_STREAM("[ICP] T_odom_to_scanner(" << odomFrame<< " to " <<
  // scannerFrame << "):\n" << T_odom_to_scanner);
  ROS_DEBUG_STREAM("[ICP] T_odom_to_map(" << odomFrame << " to " << mapFrame
                                          << "):\n"
                                          << T_odom_to_map);

  ROS_DEBUG_STREAM("[ICP] T_robot_to_scanner(" << robotFrame << " to "
                                               << scannerFrame << "):\n"
                                               << T_robot_to_scanner);
  ROS_DEBUG_STREAM("[ICP] T_robot_to_map(" << robotFrame << " to " << mapFrame
                                           << "):\n"
                                           << T_robot_to_map);

  const PM::TransformationParameters T_scanner_to_map = T_odom_to_map;
  ROS_DEBUG_STREAM("[ICP] T_scanner_to_map (" << scannerFrame << " to "
                                              << mapFrame << "):\n"
                                              << T_scanner_to_map);

  const PM::TransformationParameters T_scanner_to_cutMap =
      T_cutMap_to_map.inverse() * T_scanner_to_map;

  // Ensure a minimum amount of point after filtering
  const int ptsCount = newPointCloud->getNbPoints();
  if (ptsCount < minReadingPointCount) {
    ROS_INFO_STREAM("[ICP] Not enough points in newPointCloud: only "
                    << ptsCount << " pts.");
    return;
  }

  // Initialize the map if empty
  if (!icp.hasMap()) {
    ROS_INFO_STREAM("[MAP] Creating an initial map");
    mapCreationTime = stamp;
    setMap(updateMap(newPointCloud.release(), T_scanner_to_map, false));
    // we must not delete newPointCloud because we just stored it in the
    // mapPointCloud
    return;
  }

  // Check dimension
  if (newPointCloud->getEuclideanDim() !=
      icp.getPrefilteredInternalMap().getEuclideanDim()) {
    ROS_ERROR_STREAM("[ICP] Dimensionality missmatch: incoming cloud is "
                     << newPointCloud->getEuclideanDim() << " while map is "
                     << icp.getPrefilteredInternalMap().getEuclideanDim());
    return;
  }

  try {
    // Apply ICP
    PM::TransformationParameters T_updatedScanner_to_map;
    PM::TransformationParameters T_updatedScanner_to_cutMap;

    ROS_INFO_STREAM("[ICP] Computing - reading: "
                    << newPointCloud->getNbPoints() << ", reference: "
                    << icp.getPrefilteredInternalMap().getNbPoints());
    // T_updatedScanner_to_map = icp(*newPointCloud, T_scanner_to_map);
    icpMapLock.lock();
    T_updatedScanner_to_cutMap = icp(*newPointCloud, T_scanner_to_cutMap);
    icpMapLock.unlock();
    T_updatedScanner_to_map = T_cutMap_to_map * T_updatedScanner_to_cutMap;

    // ISER
    // TODO: generalize that
    /*{
    // extract corrections
    PM::TransformationParameters Tdelta = T_updatedScanner_to_map *
    TscannerToMap.inverse();

    // remove roll and pitch
    Tdelta(2,0) = 0;
    Tdelta(2,1) = 0;
    Tdelta(2,2) = 1;
    Tdelta(0,2) = 0;
    Tdelta(1,2) = 0;
    Tdelta(2,3) = 0; //z
    Tdelta.block(0,0,3,3) =
    transformation->correctParameters(Tdelta.block(0,0,3,3));

    T_updatedScanner_to_map = Tdelta*TscannerToMap;

    }*/

    ROS_DEBUG_STREAM("[ICP] T_updatedScanner_to_map:\n"
                     << T_updatedScanner_to_map);

    // Ensure minimum overlap between scans
    const double estimatedOverlap = icp.errorMinimizer->getOverlap();
    ROS_INFO_STREAM("[ICP] Overlap: " << estimatedOverlap);
    if (estimatedOverlap < minOverlap) {
      ROS_ERROR_STREAM(
          "[ICP] Estimated overlap too small, ignoring ICP correction!");
      return;
    }

    // Compute tf
    publishStamp = stamp;
    publishLock.lock();

    // Update old transform
    T_odom_to_map = T_updatedScanner_to_map;
    T_robot_to_map = T_updatedScanner_to_map * T_robot_to_scanner;

    // Publish tf

    if (publishMapTf) {
      tfBroadcaster.sendTransform(
          PointMatcher_ros::eigenMatrixToStampedTransform<float>(
              T_odom_to_map, mapFrame, odomFrame, stamp));
    }

    publishLock.unlock();
    processingNewCloud = false;

    ROS_DEBUG_STREAM("[ICP] T_odom_to_map:\n" << T_odom_to_map);

    // Publish odometry
    if (odomPub.getNumSubscribers()) {
      odomPub.publish(PointMatcher_ros::eigenMatrixToOdomMsg<float>(
          T_updatedScanner_to_map, mapFrame, stamp, sensorFrame));
    }
    // Publish odometry ir robot frame
    if (robotPub.getNumSubscribers()) {
      robotPub.publish(PointMatcher_ros::eigenMatrixToOdomMsg<float>(
          T_robot_to_map, mapFrame, stamp, robotFrame)); // TODO
    }
    // Publish error on odometry
    if (odomErrorPub.getNumSubscribers())
      odomErrorPub.publish(PointMatcher_ros::eigenMatrixToOdomMsg<float>(
          T_odom_to_map, mapFrame, stamp, sensorFrame));

    // check if news points should be added to the map
    if (mapping && ((estimatedOverlap < maxOverlapToMerge) ||
                    (icp.getPrefilteredInternalMap().features.cols() <
                     minMapPointCount)) &&
        (!mapBuildingInProgress)) {
      // make sure we process the last available map
      mapCreationTime = stamp;

      ROS_INFO("[MAP] Adding new points in a separate thread");

      mapBuildingTask = MapBuildingTask(
          boost::bind(&Mapper::updateMap, this, newPointCloud.release(),
                      T_updatedScanner_to_map, true));
      mapBuildingFuture = mapBuildingTask.get_future();
      mapBuildingThread =
          boost::thread(boost::move(boost::ref(mapBuildingTask)));
      mapBuildingInProgress = true;
    }

  } catch (PM::ConvergenceError error) {
    ROS_ERROR_STREAM("[ICP] failed to converge: " << error.what());
    newPointCloud->save(filePath + "/" + "error_read.vtk");
    icp.getPrefilteredMap().save(filePath + "/" + "error_ref.vtk");
    return;
  }

  // Statistics about time and real-time capability
  int realTimeRatio =
      100 * t.elapsed() / (stamp.toSec() - lastPoinCloudTime.toSec());
  realTimeRatio *= seq - lastPointCloudSeq;

  ROS_INFO_STREAM("[TIME] Total ICP took: " << t.elapsed() << " [s]");

  if (realTimeRatio < 80)
    ROS_INFO_STREAM("[TIME] Real-time capability: " << realTimeRatio << "%");
  else
    ROS_WARN_STREAM("[TIME] Real-time capability: " << realTimeRatio << "%");

  lastPoinCloudTime = stamp;
  lastPointCloudSeq = seq;
}

void Mapper::processNewMapIfAvailable() {
#if BOOST_VERSION >= 104100
  if (mapBuildingInProgress && mapBuildingFuture.has_value()) {
    ROS_INFO_STREAM("[MAP] Computation in thread done");
    setMap(mapBuildingFuture.get());
    mapBuildingInProgress = false;
  }
#endif // BOOST_VERSION >= 104100
}

void Mapper::setMap(DP *newMapPointCloud) {

  // delete old map
  if (mapPointCloud && mapPointCloud != newMapPointCloud)
    delete mapPointCloud;

  // set new map
  mapPointCloud = newMapPointCloud;

  // update ICP map
  updateIcpMap(mapPointCloud);

  // Publish map point cloud
  // FIXME this crash when used without descriptor
  ROS_INFO_STREAM("[MAP] number of subs = " << mapPub.getNumSubscribers()
                                            << " mapFrame" << mapFrame);
  if (mapPub.getNumSubscribers()) {
    ROS_INFO_STREAM("[MAP] publishing " << mapPointCloud->getNbPoints()
                                        << " points");
    mapPub.publish(PointMatcher_ros::pointMatcherCloudToRosMsg<float>(
        *mapPointCloud, mapFrame, mapCreationTime));
  }
}

void Mapper::updateIcpMap(const DP *newMapPointCloud) {
  try {
    ros::Time time_current = ros::Time::now();
    // tfListener.waitForTransform(sensorFrame, odomFrame, time_current,
    // ros::Duration(3.0));

    const PM::TransformationParameters T_scanner_to_map = this->T_odom_to_map;
    DP localMap =
        transformation->compute(*newMapPointCloud, T_scanner_to_map.inverse());
    radiusFilter->inPlaceFilter(localMap);

    icpMapLock.lock();
    // Update the transformation to the cut map
    this->T_cutMap_to_map = T_scanner_to_map;

    icp.setMap(localMap);
    icpMapLock.unlock();

  } catch (DP::InvalidField e) {
    ROS_ERROR_STREAM(e.what());
    ROS_ERROR_STREAM("[updateIcpMap] Skipping Abort");
    return;
  } catch (const std::runtime_error &e) {
    ROS_ERROR_STREAM("[updateIcpMap] " << e.what());
    return;
  }
}

Mapper::DP *
Mapper::updateMap(DP *newPointCloud,
                  const PM::TransformationParameters T_updatedScanner_to_map,
                  bool mapExists) {
  timer t;

  try {
    // Prepare empty field if not existing
    if (newPointCloud->descriptorExists("probabilityStatic") == false) {
      // newPointCloud->addDescriptor("probabilityStatic", PM::Matrix::Zero(1,
      // newPointCloud->features.cols()));
      newPointCloud->addDescriptor(
          "probabilityStatic",
          PM::Matrix::Constant(1, newPointCloud->features.cols(), priorStatic));
    }

    if (newPointCloud->descriptorExists("probabilityDynamic") == false) {
      // newPointCloud->addDescriptor("probabilityDynamic", PM::Matrix::Zero(1,
      // newPointCloud->features.cols()));
      newPointCloud->addDescriptor(
          "probabilityDynamic",
          PM::Matrix::Constant(1, newPointCloud->features.cols(), priorDyn));
    }

    if (newPointCloud->descriptorExists("dynamic_ratio") == false) {
      newPointCloud->addDescriptor(
          "dynamic_ratio", PM::Matrix::Zero(1, newPointCloud->features.cols()));
    }

    if (!mapExists) {
      ROS_INFO_STREAM("[MAP] Initial map, only filtering points");
      *newPointCloud =
          transformation->compute(*newPointCloud, T_updatedScanner_to_map);
      mapPostFilters.apply(*newPointCloud);

      return newPointCloud;
    }

    // Early out if no map modification is wanted
    if (!mapping) {
      ROS_INFO_STREAM("[MAP] Skipping modification of the map");
      return mapPointCloud;
    }

    const int mapPtsCount(mapPointCloud->getNbPoints());
    const int readPtsCount(newPointCloud->getNbPoints());

    // Build a range image of the reading point cloud (local coordinates)
    PM::Matrix radius_reading =
        newPointCloud->features.topRows(3).colwise().norm();

    PM::Matrix angles_reading(2, readPtsCount); // 0=inclination, 1=azimuth

    // No atan in Eigen, so we are for to loop through it...
    for (int i = 0; i < readPtsCount; i++) {
      const float ratio = newPointCloud->features(2, i) / radius_reading(0, i);

      angles_reading(0, i) = acos(ratio);
      angles_reading(1, i) =
          atan2(newPointCloud->features(1, i), newPointCloud->features(0, i));
    }

    std::shared_ptr<NNS> featureNNS;
    featureNNS.reset(NNS::create(angles_reading));

    // Transform the global map in local coordinates
    DP mapLocalFrameCut = transformation->compute(
        *mapPointCloud, T_updatedScanner_to_map.inverse());

    // Remove points out of sensor range
    PM::Matrix globalId(1, mapPtsCount);

    int mapCutPtsCount = 0;
    for (int i = 0; i < mapPtsCount; i++) {
      if (mapLocalFrameCut.features.col(i).head(3).norm() < sensorMaxRange) {
        mapLocalFrameCut.setColFrom(mapCutPtsCount, mapLocalFrameCut, i);
        globalId(0, mapCutPtsCount) = i;
        mapCutPtsCount++;
      }
    }

    mapLocalFrameCut.conservativeResize(mapCutPtsCount);

    PM::Matrix radius_map =
        mapLocalFrameCut.features.topRows(3).colwise().norm();

    PM::Matrix angles_map(2, mapCutPtsCount); // 0=inclination, 1=azimuth

    // No atan in Eigen, so we need to loop through it...
    for (int i = 0; i < mapCutPtsCount; i++) {
      const float ratio = mapLocalFrameCut.features(2, i) / radius_map(0, i);

      angles_map(0, i) = acos(ratio);

      angles_map(1, i) = atan2(mapLocalFrameCut.features(1, i),
                               mapLocalFrameCut.features(0, i));
    }

    // Look for NN in spherical coordinates
    Matches::Dists dists(1, mapCutPtsCount);
    Matches::Ids ids(1, mapCutPtsCount);

    featureNNS->knn(angles_map, ids, dists, 1, 0, NNS::ALLOW_SELF_MATCH,
                    maxAngle);

    // Define views on descriptors
    DP::View viewOn_Msec_overlap =
        newPointCloud->getDescriptorViewByName("stamps_Msec");
    DP::View viewOn_sec_overlap =
        newPointCloud->getDescriptorViewByName("stamps_sec");
    DP::View viewOn_nsec_overlap =
        newPointCloud->getDescriptorViewByName("stamps_nsec");

    DP::View viewOnProbabilityStatic =
        mapPointCloud->getDescriptorViewByName("probabilityStatic");
    DP::View viewOnProbabilityDynamic =
        mapPointCloud->getDescriptorViewByName("probabilityDynamic");
    DP::View viewOnDynamicRatio =
        mapPointCloud->getDescriptorViewByName("dynamic_ratio");

    DP::View viewOn_normals_map =
        mapPointCloud->getDescriptorViewByName("normals");
    DP::View viewOn_Msec_map =
        mapPointCloud->getDescriptorViewByName("stamps_Msec");
    DP::View viewOn_sec_map =
        mapPointCloud->getDescriptorViewByName("stamps_sec");
    DP::View viewOn_nsec_map =
        mapPointCloud->getDescriptorViewByName("stamps_nsec");

    viewOnDynamicRatio = PM::Matrix::Zero(1, mapPtsCount);
    for (int i = 0; i < mapCutPtsCount; i++) {
      if (dists(i) != numeric_limits<float>::infinity()) {
        const int readId = ids(0, i);
        const int mapId = globalId(0, i);

        // in local coordinates
        const Eigen::Vector3f readPt =
            newPointCloud->features.col(readId).head(3);
        const Eigen::Vector3f mapPt = mapLocalFrameCut.features.col(i).head(3);
        const Eigen::Vector3f mapPt_n = mapPt.normalized();
        const float delta = (readPt - mapPt).norm();
        const float d_max = eps_a * readPt.norm();

        const Eigen::Vector3f normal_map = viewOn_normals_map.col(mapId);

        // Weight for dynamic elements
        const float w_v = eps + (1 - eps) * fabs(normal_map.dot(mapPt_n));
        const float w_d1 = eps + (1 - eps) * (1 - sqrt(dists(i)) / maxAngle);

        const float offset = delta - eps_d;
        float w_d2 = 1;
        if (delta < eps_d || mapPt.norm() > readPt.norm()) {
          w_d2 = eps;
        } else {
          if (offset < d_max) {
            w_d2 = eps + (1 - eps) * offset / d_max;
          }
        }

        float w_p2 = eps;
        if (delta < eps_d) {
          w_p2 = 1;
        } else {
          if (offset < d_max) {
            w_p2 = eps + (1 - eps) * (1 - offset / d_max);
          }
        }

        // We don't update point behind the reading
        if ((readPt.norm() + eps_d + d_max) >= mapPt.norm()) {
          const float lastDyn = viewOnProbabilityDynamic(0, mapId);
          const float lastStatic = viewOnProbabilityStatic(0, mapId);

          const float c1 = (1 - (w_v * (1 - w_d1)));
          const float c2 = w_v * (1 - w_d1);

          // Lock dynamic point to stay dynamic under a threshold
          if (lastDyn < maxDyn) {
            viewOnProbabilityDynamic(0, mapId) =
                c1 * lastDyn +
                c2 * w_d2 * ((1 - alpha) * lastStatic + beta * lastDyn);
            viewOnProbabilityStatic(0, mapId) =
                c1 * lastStatic +
                c2 * w_p2 * (alpha * lastStatic + (1 - beta) * lastDyn);
          } else {
            viewOnProbabilityStatic(0, mapId) = eps;
            viewOnProbabilityDynamic(0, mapId) = 1 - eps;
          }

          // normalization
          const float sumZ = viewOnProbabilityDynamic(0, mapId) +
                             viewOnProbabilityStatic(0, mapId);
          assert(sumZ >= eps);

          viewOnProbabilityDynamic(0, mapId) /= sumZ;
          viewOnProbabilityStatic(0, mapId) /= sumZ;

          viewOnDynamicRatio(0, mapId) = w_d2;

          // TODO use the new time structure
          // Refresh time
          viewOn_Msec_map(0, mapId) = viewOn_Msec_overlap(0, readId);
          viewOn_sec_map(0, mapId) = viewOn_sec_overlap(0, readId);
          viewOn_nsec_map(0, mapId) = viewOn_nsec_overlap(0, readId);
        }
      }
    }

    // Generate temporary map for density computation
    DP tmp_map = mapLocalFrameCut;
    tmp_map.concatenate(*newPointCloud);

    // build and populate NNS
    featureNNS.reset(NNS::create(tmp_map.features, tmp_map.features.rows() - 1,
                                 NNS::KDTREE_LINEAR_HEAP,
                                 NNS::TOUCH_STATISTICS));

    PM::Matches matches_overlap(Matches::Dists(1, readPtsCount),
                                Matches::Ids(1, readPtsCount));

    featureNNS->knn(newPointCloud->features, matches_overlap.ids,
                    matches_overlap.dists, 1, 0);

    DP overlap(newPointCloud->createSimilarEmpty());
    DP no_overlap(newPointCloud->createSimilarEmpty());

    int ptsOut = 0;
    int ptsIn = 0;
    for (int i = 0; i < readPtsCount; ++i) {
      if (matches_overlap.dists(i) > maxDistNewPoint) {
        no_overlap.setColFrom(ptsOut, *newPointCloud, i);
        ptsOut++;
      } else {
        overlap.setColFrom(ptsIn, *newPointCloud, i);
        ptsIn++;
      }
    }

    no_overlap.conservativeResize(ptsOut);
    overlap.conservativeResize(ptsIn);

    // Publish outliers
    // if (outlierPub.getNumSubscribers())
    //{
    // outlierPub.publish(PointMatcher_ros::pointMatcherCloudToRosMsg<float>(no_overlap,
    // mapFrame, mapCreationTime));
    //}

    // Initialize descriptors
    no_overlap.addDescriptor(
        "probabilityStatic",
        PM::Matrix::Constant(1, no_overlap.features.cols(), priorStatic));
    no_overlap.addDescriptor(
        "probabilityDynamic",
        PM::Matrix::Constant(1, no_overlap.features.cols(), priorDyn));
    no_overlap.addDescriptor("dynamic_ratio",
                             PM::Matrix::Zero(1, no_overlap.features.cols()));

    // shrink the newPointCloud to the new information
    *newPointCloud = no_overlap;

    // Correct new points using ICP result
    *newPointCloud =
        transformation->compute(*newPointCloud, T_updatedScanner_to_map);

    // Merge point clouds to map
    newPointCloud->concatenate(*mapPointCloud);
    mapPostFilters.apply(*newPointCloud);
  } catch (DP::InvalidField e) {
    ROS_ERROR_STREAM(e.what());
    ROS_ERROR_STREAM("[MAP] Skipping Abort");
    return mapPointCloud;
    // abort();
  } catch (const std::runtime_error &e) {
    ROS_ERROR_STREAM("[MAP] " << e.what());
    return mapPointCloud;
  }

  ROS_INFO_STREAM("[TIME][MAP] New map available ("
                  << newPointCloud->features.cols() << " pts), update took "
                  << t.elapsed() << " [s]");

  return newPointCloud;
}

void Mapper::waitForMapBuildingCompleted() {
#if BOOST_VERSION >= 104100
  if (mapBuildingInProgress) {
    // we wait for now, in future we should kill it
    mapBuildingFuture.wait();
    mapBuildingInProgress = false;
  }
#endif // BOOST_VERSION >= 104100
}

void Mapper::publishLoop(double publishPeriod) {
  if (publishPeriod == 0)
    return;
  ros::Rate r(1.0 / publishPeriod);
  while (ros::ok()) {
    publishTransform();
    r.sleep();
  }
}

void Mapper::publishTransform() {
  if (processingNewCloud == false && publishMapTf == true) {
    publishLock.lock();
    // Note: we use now as timestamp to refresh the tf and avoid other buffer to
    // be empty
    tfBroadcaster.sendTransform(
        PointMatcher_ros::eigenMatrixToStampedTransform<float>(
            T_odom_to_map, mapFrame, odomFrame, ros::Time::now()));
    publishLock.unlock();
  }
}

bool Mapper::getPointMap(map_msgs::GetPointMap::Request &req,
                         map_msgs::GetPointMap::Response &res) {
  if (!mapPointCloud)
    return false;

  // FIXME: do we need a mutex here?
  res.map = PointMatcher_ros::pointMatcherCloudToRosMsg<float>(
      *mapPointCloud, mapFrame, ros::Time::now());
  return true;
}

bool Mapper::saveMap(map_msgs::SaveMap::Request &req,
                     map_msgs::SaveMap::Response &res) {
  if (!mapPointCloud)
    return false;

  try {
    mapPointCloud->save(req.filename.data);
  } catch (const std::runtime_error &e) {
    ROS_ERROR_STREAM("Unable to save: " << e.what());
    return false;
  }

  ROS_INFO_STREAM("[MAP] saved at " << req.filename.data << " with "
                                    << mapPointCloud->features.cols()
                                    << " points.");
  return true;
}

bool Mapper::loadMap(ethzasl_icp_mapper::LoadMap::Request &req,
                     ethzasl_icp_mapper::LoadMap::Response &res) {
  waitForMapBuildingCompleted();

  // ROS_INFO_STREAM("[DEBUG] Map file name: " << req.filename.data);

  DP *cloud(new DP(DP::load(req.filename.data)));

  // Print new map information
  const int dim = cloud->features.rows();
  const int nbPts = cloud->features.cols();
  ROS_INFO_STREAM("[MAP] Loading " << dim - 1 << "D point cloud ("
                                   << req.filename.data << ") with " << nbPts
                                   << " points.");

  ROS_INFO_STREAM("  With descriptors:");
  for (int i = 0; i < cloud->descriptorLabels.size(); i++) {
    ROS_INFO_STREAM("    - " << cloud->descriptorLabels[i].text);
  }

  // reset transformation
  publishLock.lock();
  T_odom_to_map = PM::TransformationParameters::Identity(dim, dim);
  T_cutMap_to_map = PM::TransformationParameters::Identity(dim, dim);
  T_cutMap_to_map = PM::TransformationParameters::Identity(dim, dim);

  // ISER
  // T_odom_to_map(2,3) = mapElevation;
  publishLock.unlock();

  // setMap(cloud);
  setMap(updateMap(cloud, PM::TransformationParameters::Identity(dim, dim),
                   false));

  return true;
}

bool Mapper::reset(std_srvs::Empty::Request &req,
                   std_srvs::Empty::Response &res) {
  waitForMapBuildingCompleted();

  // note: no need for locking as we do ros::spin(), to update if we go for
  // multi-threading
  publishLock.lock();
  T_odom_to_map = PM::TransformationParameters::Identity(4, 4);
  T_cutMap_to_map = PM::TransformationParameters::Identity(4, 4);
  publishLock.unlock();

  icp.clearMap();

  return true;
}

bool Mapper::correctPose(ethzasl_icp_mapper::CorrectPose::Request &req,
                         ethzasl_icp_mapper::CorrectPose::Response &res) {
  publishLock.lock();
  T_odom_to_map = PointMatcher_ros::odomMsgToEigenMatrix<float>(req.odom);
  const PM::TransformationParameters T_scanner_to_map = T_odom_to_map;
  // T_cutMap_to_map = T_scanner_to_map;

  // update ICP map
  updateIcpMap(mapPointCloud);

  // ISER
  /*
  {
  // remove roll and pitch
  T_odom_to_map(2,0) = 0;
  T_odom_to_map(2,1) = 0;
  T_odom_to_map(2,2) = 1;
  T_odom_to_map(0,2) = 0;
  T_odom_to_map(1,2) = 0;
  T_odom_to_map(2,3) = mapElevation; //z
  }*/

  tfBroadcaster.sendTransform(
      PointMatcher_ros::eigenMatrixToStampedTransform<float>(
          T_odom_to_map, mapFrame, odomFrame, ros::Time::now()));
  publishLock.unlock();

  return true;
}

bool Mapper::setMode(ethzasl_icp_mapper::SetMode::Request &req,
                     ethzasl_icp_mapper::SetMode::Response &res) {
  // Impossible states
  if (req.localize == false && req.map == true)
    return false;

  localizing = req.localize;
  mapping = req.map;

  return true;
}

bool Mapper::getMode(ethzasl_icp_mapper::GetMode::Request &req,
                     ethzasl_icp_mapper::GetMode::Response &res) {
  res.localize = localizing;
  res.map = mapping;
  return true;
}

bool Mapper::getBoundedMap(ethzasl_icp_mapper::GetBoundedMap::Request &req,
                           ethzasl_icp_mapper::GetBoundedMap::Response &res) {
  if (!mapPointCloud)
    return false;

  const float max_x = req.topRightCorner.x;
  const float max_y = req.topRightCorner.y;
  const float max_z = req.topRightCorner.z;

  const float min_x = req.bottomLeftCorner.x;
  const float min_y = req.bottomLeftCorner.y;
  const float min_z = req.bottomLeftCorner.z;

  cerr << "min [" << min_x << ", " << min_y << ", " << min_z << "] " << endl;
  cerr << "max [" << max_x << ", " << max_y << ", " << max_z << "] " << endl;

  tf::StampedTransform stampedTr;

  Eigen::Affine3d eigenTr;
  tf::poseMsgToEigen(req.mapCenter, eigenTr);
  Eigen::MatrixXf T = eigenTr.matrix().inverse().cast<float>();
  // const Eigen::MatrixXf T = eigenTr.matrix().cast<float>();

  cerr << "T:" << endl << T << endl;
  T = transformation->correctParameters(T);

  // FIXME: do we need a mutex here?
  const DP centeredPointCloud = transformation->compute(*mapPointCloud, T);
  DP cutPointCloud = centeredPointCloud.createSimilarEmpty();

  cerr << centeredPointCloud.features.topLeftCorner(3, 10) << endl;
  cerr << T << endl;

  int newPtCount = 0;
  for (int i = 0; i < centeredPointCloud.features.cols(); i++) {
    const float x = centeredPointCloud.features(0, i);
    const float y = centeredPointCloud.features(1, i);
    const float z = centeredPointCloud.features(2, i);

    if (x < max_x && x > min_x && y < max_y && y > min_y && z < max_z &&
        z > min_z) {
      cutPointCloud.setColFrom(newPtCount, centeredPointCloud, i);
      newPtCount++;
    }
  }

  ROS_INFO_STREAM("Extract " << newPtCount << " points from the map");

  cutPointCloud.conservativeResize(newPtCount);
  cutPointCloud = transformation->compute(cutPointCloud, T.inverse());

  // Send the resulting point cloud in ROS format
  res.boundedMap = PointMatcher_ros::pointMatcherCloudToRosMsg<float>(
      cutPointCloud, mapFrame, ros::Time::now());
  return true;
}

void Mapper::loadExternalParameters() {
  // load configs
  string configFileName;

  pn.param<bool>("bag_file_mode", bag_file_mode, false);
  pn.param<std::string>("velodyne_topic", velodyne_topic, "velodyne_points");
  pn.param<std::string>("bag_filename", bag_filename, "");
  pn.param<int>("bag_file_stepping", bag_file_stepping, 0);
  pn.param<int>("bag_file_interleave", bag_file_interleave, 0);
  pn.param<int>("bag_file_save_interval", bag_file_save_interval, 0);

  pn.param<int>("bag_file_skip_sec", bag_file_skip_sec, 0);

  if (ros::param::get("~icpConfig", configFileName)) {
    ifstream ifs(configFileName.c_str());
    if (ifs.good()) {
      icp.loadFromYaml(ifs);
    } else {
      ROS_ERROR_STREAM("Cannot load ICP config from YAML file "
                       << configFileName);
      icp.setDefault();
    }
  } else {
    ROS_INFO_STREAM("No ICP config file given, using default");
    icp.setDefault();
  }

  if (ros::param::get("~inputFiltersConfig", configFileName)) {
    ifstream ifs(configFileName.c_str());
    if (ifs.good()) {
      inputFilters = PM::DataPointsFilters(ifs);
    } else {
      ROS_ERROR_STREAM("Cannot load input filters config from YAML file "
                       << configFileName);
    }
  } else {
    ROS_INFO_STREAM(
        "No input filters config file given, not using these filters");
  }

  if (ros::param::get("~mapPreFiltersConfig", configFileName)) {
    ifstream ifs(configFileName.c_str());
    if (ifs.good()) {
      mapPreFilters = PM::DataPointsFilters(ifs);
    } else {
      ROS_ERROR_STREAM("Cannot load map pre-filters config from YAML file "
                       << configFileName);
    }
  } else {
    ROS_INFO_STREAM(
        "No map pre-filters config file given, not using these filters");
  }

  if (ros::param::get("~mapPostFiltersConfig", configFileName)) {
    ifstream ifs(configFileName.c_str());
    if (ifs.good()) {
      mapPostFilters = PM::DataPointsFilters(ifs);
    } else {
      ROS_ERROR_STREAM("Cannot load map post-filters config from YAML file "
                       << configFileName);
    }
  } else {
    ROS_INFO_STREAM(
        "No map post-filters config file given, not using these filters");
  }
}

bool Mapper::reloadallYaml(std_srvs::Empty::Request &req,
                           std_srvs::Empty::Response &res) {
  loadExternalParameters();
  ROS_INFO_STREAM("Parameters reloaded");

  return true;
}

// Main function supporting the Mapper class
int main(int argc, char **argv) {
  ros::init(argc, argv, "mapper");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");
  Mapper mapper(n, pn);
  // ros::spin();

  return 0;
}
