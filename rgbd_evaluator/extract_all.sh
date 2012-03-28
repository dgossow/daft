#!/bin/bash

bin/extract_detector /work/gossow/bags/tum_poster/viewpoint.bag
bin/extract_detector /work/gossow/bags/tum_poster/rotate.bag
bin/extract_detector /work/gossow/bags/tum_poster/scale.bag

bin/extract_detector /work/gossow/bags/world_map/*.bag
#bin/extract_detector /work/gossow/bags/ecai_poster/*.bag

