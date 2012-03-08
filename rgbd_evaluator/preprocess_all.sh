#!/bin/bash

bin/rgbd_preprocessing /work/gossow/bags/eval/poster_scale.bag 
bin/rgbd_preprocessing /work/gossow/bags/eval/poster_rotate.bag 
bin/rgbd_preprocessing /work/gossow/bags/eval/poster_viewpoint.bag -r
#bin/rgbd_preprocessing /work/gossow/bags/eval/hallway_viewpoint.bag -r

