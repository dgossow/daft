#!/bin/bash

bin/extract_detector /work/gossow/bags/frosties/*.bag

bin/extract_detector /work/gossow/bags/granada/*.bag

bin/extract_detector /work/gossow/bags/tum_poster/viewpoint.bag -k 750
bin/extract_detector /work/gossow/bags/tum_poster/rotate.bag -k 750
bin/extract_detector /work/gossow/bags/tum_poster/scale.bag -k 750

bin/extract_detector /work/gossow/bags/world_map/*.bag -k 500
#bin/extract_detector /work/gossow/bags/ecai_poster/*.bag -k 500

