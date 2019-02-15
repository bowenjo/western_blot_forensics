import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import argparse
import os

from generate_synthetic_dataset import GenerateSyntheticData
from analysis.HOG_analyze import plot_AUC_synthetic
import params.config_synthetic as config_synthetic
import params.config as config

parser = argparse.ArgumentParser(description='')

parser.add_argument('--project_name', dest='project_name', help='Name of the folder to save')
parser.add_argument('--t_name', dest='t_name', help='Name of transformation')

args = parser.parse_args()

name, t_args, labels = config_synthetic.get_args(args.t_name)

project = args.project_name  
out_dir = os.path.join(config.project_dir, project)

out_folders = ['/saves', '/vis']
all_dir = [out_dir] + [out_dir + dr for dr in out_folders]

for dr in all_dir:
	if not os.path.exists(dr):
		os.mkdir(dr)

save_file = os.path.join(all_dir[1], name + ".npy") 

# generate data
G = GenerateSyntheticData()
G.gather(config_synthetic.path_to_images, config_synthetic.x_lim, config_synthetic.y_lim, 
		 config_synthetic.padding_size)

data = []
if config_synthetic.pre_load:
	plot_AUC_synthetic(data, t_args, name, labels, save_file, config_synthetic.pre_load)
else:

	for arg in t_args:
		data.append(G.generate_copies(config_synthetic.copy_chance, name, arg))

	plot_AUC_synthetic(data, t_args, name, labels, save_file, config_synthetic.pre_load)
