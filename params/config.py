import os
## Configuration file containing tunable parameters for fingerprinting and matching procedures

# Important directories
home_dir = os.path.expanduser("~") + "/Documents/GitHub/NonRepositories/westernBlotForensics/wb_forensics"
project_dir = home_dir + "/Projects"
dataset_dir = home_dir + "/Datasets"

## Segmentation Pipeline

# Pre-trained classifier files
clf_image_filename = home_dir + "/classifiers/saves_alexnet_images/dict.npy"
clf_blot_filename = home_dir + "/classifiers/saves_blots_2/dict.npy"

## Image cutter
fig_panel_threshold = 242
cnt_arc_len_approx = .04
cnt_area_upr = 3000
cnt_area_lwr = 30

# Blot cutter
SIFT_implement = False
# Preprocessing
inv_thresh_value = 60
se_kern_size = 11
mask_border_size = 0
thresh_type = "adaptive"
adaptive_thresh_block_size = (.16, .16)
adaptive_thresh_c = 10
# Segmentation
cut_T = 0
cut_order = (0.25, 0.015)
cut_min_vote = (0.15, 0.15)
cut_min_spacing = (.15, .05)
cut_line_decision = 0
dist_transform_weight = 1
# Postprocessing
ellipse_corr = 2
filter_cnt_pts = 5
pixel_min = 5
pixel_max = 500
ar_min = 1.0 / 20.0 
ar_max = 3
theta_lim = 20
variance_lim = 40
white_max = .1 #1 #.1
window_stretch = 10
max_descriptor_per_image = 40

## Fingerprint Pipeline
HOG_bin_shape = (2,4)
HOG_orientations = 8

## Matching Pipeline
# Reference labels (eBik dataset)
reference_loc_labels = home_dir + "/Datasets/eBik/annotations.npy"
reference_loc_min_distance = 30
# Matching method
match_type = "image" # pick {feature, image}
query_method = "ALL_PAIRS"
# Scoring
d_T = 1.0
r_T = None
score_T = 1.60
multiplier_num_sum = 2
accum_type = 'mean'
