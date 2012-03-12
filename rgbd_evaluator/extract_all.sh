#!/bin/bash

bin/extract_detector /work/gossow/bags/eval/poster_scale.bag 
bin/extract_detector /work/gossow/bags/eval/poster_rotate.bag 
bin/extract_detector /work/gossow/bags/eval/poster_viewpoint.bag -r
#bin/extract_detector /work/gossow/bags/eval/hallway_viewpoint.bag -r

