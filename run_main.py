# import basic modules
import os
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import json

# import model-specific modules
import params.config as config
from wbFingerprint import collect_fingerprints
from wbForensicsHOG import wbForensicsHOG
from label_matches import AssignMatches


parser = argparse.ArgumentParser(description='')

## ====== arguments ==============
parser.add_argument('--project_name', dest='project_name', help='Name of the folder')
parser.add_argument('--dataset_name', dest='dataset_name', help='Name of the dataset to run')
parser.add_argument('--dataset_type', dest='dataset_type', help='specifies how to processes figure(s)')
parser.add_argument('--preload', dest='preload', default=False, help='Set to True if you want to pre-load a fingerprint file')

args = parser.parse_args()

# TODO: create filenames/ folder based on dataset_type
# create project in project directory
project = args.project_name + '_' + args.dataset_name  
out_dir = os.path.join(config.project_dir, project)

out_folders = ['/saves', '/log', '/vis']
all_dir = [out_dir] + [out_dir + dr for dr in out_folders]

for dr in all_dir:
	if not os.path.exists(dr):
		os.mkdir(dr)

with open(all_dir[2] + '/param_log.txt', 'w') as f:
  	f.write(' '.join(["config.%s = %s \n" % (k,v) for k,v in config.__dict__.items()]))

out_files = ["/fingerprint.npy", "/pairs.npy", "/dists.npy", "/ratios.npy", "/scores.npy", "/reference_analysis.npy"]
fingerprint_filename, pairs_filename, dists_filename, ratios_filename, scores_filename, reference_filename = \
	[all_dir[1] + file for file in out_files]

## =======================================================================
## ===========================Collect fingerprints========================
## =======================================================================
dataset = args.dataset_name + '/' + args.dataset_name + '.npy' 
to_dataset = os.path.join(config.dataset_dir, dataset)
to_folder = args.dataset_name
if not args.preload:
	collect_fingerprints(to_dataset, to_folder, fingerprint_filename, args.dataset_type)

## =======================================================================
## ========================== Reference Labeling =========================
## =======================================================================
Database = np.load(fingerprint_filename).item()
if args.dataset_type == "ref":
	Figures = np.load(to_dataset)
	reference_loc_labels = np.load(config.reference_loc_labels).item()
	AM = AssignMatches(Database, reference_loc_labels, Figures, to_folder)
	AM.assign(ASSIGNMENT=config.match_type)
	np.save(fingerprint_filename, AM.Database)

## =======================================================================
## ==========================Compute match scores ========================
## =======================================================================

Database = np.load(fingerprint_filename).item()
print("Fingerprints collected and saved:")
print("{} total descriptors computed".format(Database["feature_vectors"].shape[0]))

Forensics = wbForensicsHOG(Database=Database)
Forensics.KDTree_pairs(len(Database["feature_vectors"])+1)
Forensics.d_rank(pairs=Forensics.pairs, ratios=Forensics.ratios, distances=Forensics.dists)
	
np.save(pairs_filename, Forensics.pairs)
np.save(ratios_filename, Forensics.ratios)
np.save(dists_filename, Forensics.dists)
np.save(scores_filename, Forensics.Dist_Rank)

# # For feature pairs
if args.dataset_type == 'ref':
	if config.match_type == "feature":
		Forensics.assign_match_label(pairs=Forensics.pairs, ratios=Forensics.ratios, distances=Forensics.dists, 
			ASSIGNMENT = config.match_type)
	elif config.match_type == "image":
		Forensics.assign_match_label(pairs=Forensics.Dist_Rank["image_match_pairs"], 
									 distances = Forensics.Dist_Rank["image_match_dists"], 
									 ratios=Forensics.Dist_Rank["image_match_ratios"], 
									 ASSIGNMENT= config.match_type)
	else:
		raise NameError(config.match_type + " is not a recognized match type")
	np.save(reference_filename, Forensics.truth_values)

print("Matching pairs, and scores collected:")
print("{} total matches collected".format(Forensics.pair_counter))



	

