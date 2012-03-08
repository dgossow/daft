roscore

roslaunch openni_launch openni.launch

rosrun dynamic_reconfigure reconfigure_gui

rosrun image_view image_view image:=/camera/depth_registered/image
rosrun image_view image_view image:=/camera/rgb/image_color

rosrun rgbd_evaluator test_rgbd_cv intensity_image:=/camera/rgb/image_rect depth_image:=/camera/depth_registered/image camera_info:=/camera/rgb/camera_info

