import numpy as np 
import cv2 
import tensorflow as tf
import os, glob
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import sys
import flask_app.utils_compute_flask as utils
import time

import params.config as config 
from wbFingerprint import WesternBlot
from wbForensicsHOG import wbForensicsHOG
from classifiers.clf import wb_clf

class DisplayFigure(object):
	"""Returns the chosen figure with labeled image pairs"""
	def __init__(self, filename, show_matches=True, threshold=None):
		self.figure = np.load(os.path.join('uploads', filename))
		#self.figure = cv2.imread(os.path.join('uploads', filename)) 
		self.compute_fingerprint()
		self.compute_matches()

		if show_matches:
			self.view()
		else:
			self.save_figure(self.figure)

	def compute_fingerprint(self):
		"""
		Wrapper to compute fingerprints using wbFingerprints module
		"""
		# initialize tensorflow session
		sess = tf.InteractiveSession()
		clf_image = wb_clf(config.clf_image_filename, reshape_size=None, input_dim=(100,200), 
							name="image", ALEXNET=True)
		clf_blot = wb_clf(config.clf_blot_filename, input_dim=(15,30), reshape_size=4*8*64, name="blot")
		sess.run(tf.global_variables_initializer())
		# create western blot fingerprinting object
		WB = WesternBlot(clf_image=clf_image, clf_blot=clf_blot)
		WB.figure = self.figure
		WB.figure_gray = cv2.cvtColor(self.figure, cv2.COLOR_BGR2GRAY)
		# compute fingerprint
		WB.westernBlotExtractor(VISUALIZE=False)
		self.local_database = WB.Fingerprint

	def compute_matches(self):
		"""
		Wrapper to compute matches using wbForensicsHOG module
		"""
		self.local_database["figure_number"] = [0] * len(self.local_database["feature_vectors"])
		Forensics = wbForensicsHOG(Database=self.local_database)
		Forensics.KDTree_pairs(leaf_size = len(self.local_database)+1)
		Forensics.d_rank(pairs=Forensics.pairs, distances=Forensics.dists, ratios=Forensics.ratios)

		self.local_matches = Forensics.Dist_Rank

	def view(self):
		"""	
		paints matches onto the figure
		"""
		figure_out = self.figure.copy()
		image_pairs = np.unique(self.local_matches["image_pairs"][0])
		for i in image_pairs:
			# draw bounding box
			i_loc = self.local_database["image_locs"][np.where(self.local_database["image_idx"] == i)[0][0]]
			cv2.rectangle(figure_out, (int(i_loc[0]), int(i_loc[1])), (int(i_loc[0]+i_loc[2]), int(i_loc[1]+i_loc[3])),
						  color = (255,0,0), thickness=5)
			# label matches text
			cv2.putText(figure_out, str(i), (int(i_loc[0]-50), int(i_loc[1] + 50)), cv2.FONT_HERSHEY_SIMPLEX, 2,
					    color=(255,0,0), thickness=7)
		self.save_figure(figure_out)

	def save_figure(self, data):
		"""save plotfile to server """

		sizes = np.shape(data)
		fig = plt.figure()
		fig.set_size_inches(1, 1. * sizes[0]/sizes[1], forward = False)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(data, "gray")

		#plt.show()
		self.plotfile = os.path.join('static', 'Figure' + '.png')
		plt.savefig(self.plotfile, dpi = sizes[1])




class AffineVisualizer(object):
	"""Returns affine matching components of a chosen image pair"""
	def __init__(self, Database, Matches, AFFINE=True, CONTRAST=True, HEATMAP="gray"):
		self.AFFINE = AFFINE
		self.CONTRAST = CONTRAST
		self.Database = Database
		self.Matches = Matches
		self.HEATMAP = HEATMAP

	def init_match_alignment(self, target):
		self.match_alignment[target] = []
		self.match_alignment_scores[target] = []

	def compute_transform(self, image_pair, figure):
		"""
		computes a matching transform between image indexed by i1 and another image indexed by i2 in self.Database. 
		"""	
		image_pair = [int(image_pair[1]), int(image_pair[3])]
		indices = np.where(np.all(np.abs(self.Matches["image_pairs"][0] - image_pair) == 0, axis=1)) 
		feature_pairs = self.Matches["image_match_pairs"][0][indices[0][0]]
		feature_pair_scores = self.Matches["image_feature_scores"][0][indices[0][0]] 

		i1_loc = self.Database["image_locs"][np.where(np.array(self.Database["image_idx"]) == image_pair[0])[0][0]]
		i2_loc = self.Database["image_locs"][np.where(np.array(self.Database["image_idx"]) == image_pair[1])[0][0]]

		self.image1 = cv2.cvtColor(extract_image(figure, i1_loc), cv2.COLOR_BGR2GRAY)
		self.image2 = cv2.cvtColor(extract_image(figure, i2_loc), cv2.COLOR_BGR2GRAY)
		# self.image1 = extract_image(figure, i1_loc)
		# self.image2 = extract_image(figure, i2_loc)

		self.match_alignment = {}; self.match_alignment_scores = {}
		for pair, score in zip(feature_pairs, feature_pair_scores):
			i1 = self.Database["image_idx"][pair[0]]
			i2 = self.Database["image_idx"][pair[1]]

			if i1 == image_pair[0]:
				target = pair[0] 
				start = pair[1]
				if target not in self.match_alignment.keys():
					self.init_match_alignment(target)
			else:
				target = pair[1]
				start = pair[0]				
				if target not in self.match_alignment.keys():
					self.init_match_alignment(target)

			# get ellipse and convert to three cardinal points
			target_ellipse = self.Database["orientation"][target]
			start_ellipse = self.Database["orientation"][start]
			target_points, target_params = utils.ellipse2points(target_ellipse)
			start_points, start_params = utils.ellipse2points(start_ellipse)

			# compute angle between Major axes
			rotation_angle = int(np.abs(target_params[0] - start_params[0]))
			reflection_truth = target_params[1] * start_params[1]
			if reflection_truth > 0:
				reflection_truth = "No"
			else:
				reflection_truth = "Yes"
			resolution_perc = round(((target_params[2] / start_params[2]) - 1), 1)

			# extract features for histogram matching
			target_bloc = self.Database["blot_locs"][target]
			start_bloc = self.Database["blot_locs"][start]
			target_feature = extract_image(self.image1, target_bloc, ex=5)
			final_feature = extract_image(self.image2, start_bloc, ex=5) 

			#find affine transform to match
			if self.AFFINE:
				T = utils.compute_affine(start_points, target_points)
				image2_warped = cv2.warpAffine(self.image2.copy(), T, (self.image1.shape[1], self.image1.shape[0]))
				final_feature = extract_image(image2_warped, target_bloc, ex=5)
			
			# match hist
			if self.CONTRAST:
				final_feature = utils.histogram_match(final_feature, target_feature)

			self.match_alignment[target].append((final_feature, start, rotation_angle, reflection_truth, \
												 resolution_perc, score))
			self.match_alignment_scores[target].append(score)

	def construct_aligned(self):
		image1 = self.image1.copy()
		image2 = self.image2.copy()

		# color assignment
		cmap = cm.get_cmap('rainbow')
		cl = np.linspace(0,1, len(self.match_alignment.keys()))
		cl_i = 0

		# initialize result storage 
		self.features = []; rects = {"target": [], "starter": []}
		for target in self.match_alignment.keys():
			color = np.random.randint(0,255, size=3)
			m_feature_final, m_i, m_angle, m_ref, m_scale, m_score = self.match_alignment[target][ \
			np.argmax(self.match_alignment_scores[target])]

			t_loc = self.Database["blot_locs"][target]
			m_loc = self.Database["blot_locs"][m_i]

			t_feature = extract_image(self.image1, t_loc, ex=5)
			m_feature = extract_image(self.image2, m_loc, ex=5) 

			stitched_features = stitch_image([t_feature, m_feature_final, m_feature], axis='v')

			# give a color-coded border
			stitched_features = cv2.copyMakeBorder(stitched_features,15,15,15,15,cv2.BORDER_CONSTANT,
												   value=(255,255,255))
			stitched_rect = mpatches.Rectangle((3,3), stitched_features.shape[1]-6, stitched_features.shape[0]-6, 
											   fill=False, edgecolor=cmap(cl[cl_i]), linestyle="dotted", lw=.75)

			# paint on image
			rect_t = mpatches.Rectangle((int(t_loc[0]-5), int(t_loc[1]-5)), int(t_loc[2]+10), int(t_loc[3]+10), 
										 fill=False, edgecolor=cmap(cl[cl_i]), linestyle="dotted", lw=.75)
			rect_m = mpatches.Rectangle((int(m_loc[0]-5), int(m_loc[1]-5)), int(m_loc[2]+10), int(m_loc[3]+10), 
										 fill=False, edgecolor=cmap(cl[cl_i]), linestyle="dotted", lw=.75)
			cl_i+=1
			rects["target"].append(rect_t)
			rects["starter"].append(rect_m)

			# save feature display results
			stitched_features_file = self.save_figure(stitched_features, name="feat"+str(target), rects=[stitched_rect])
			self.features.append((stitched_features_file, m_angle, m_ref, m_scale, round(m_score[0], 3)))

		# save image display results
		self.image1_file = self.save_figure(image1, "image1", rects["target"])
		self.image2_file = self.save_figure(image2, "image2", rects["starter"])

	def save_figure(self, data, name, rects=None):
		"""save plotfile to server """
		sizes = np.shape(data)
		fig = plt.figure()
		fig.set_size_inches(1, 1. * sizes[0]/sizes[1], forward = False)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(data, self.HEATMAP)

		if rects:
			for r in rects:
				ax.add_patch(r)

		plotfile = os.path.join('static', name + '_' + str(time.time()) + '.png')
		#plt.show()
		plt.savefig(plotfile, dpi = sizes[1])

		return plotfile

def draw_points(image, pnts):
	print(pnts)
	for pnt in pnts:
		s = image.shape
		#print(pnt, s)
		image[np.minimum(int(pnt[1]), s[0]), np.minimum(int(pnt[0]), s[1]-1), :] = (255,0,0)

		
def stitch_image(images, axis='v', sp = 10):
	shapes = [img.shape[:2] for img in images]

	if axis == 'v':
		row = np.max(shapes, axis=0)[0]
		col = np.sum(shapes, axis=0)[1] + sp*len(images)
	elif axis == 'h':
		row = np.sum(shapes, axis=0)[0] + sp*len(images)
		col = np.max(shapes, axis=0)[1]
	else:
		raise NameError(axis + " is not a recognized axis type")

	bckgrnd = 255*np.ones((row, col), dtype=np.uint8)

	idx = 0
	for img, shp in zip(images,shapes):
		if axis == 'v':
			bckgrnd[0:shp[0], idx:idx+shp[1]] = img
			idx += shp[1] + sp
		elif axis == 'h':
			bckgrnd[idx:idx+shp[0], 0:shp[1]]
			idx += shp[0] + sp

	return bckgrnd

def extract_image(img, l, ex=0):
	y1  = np.maximum(0, l[1] - ex)
	x1 = np.maximum(0, l[0] - ex)

	y2 = np.minimum(img.shape[0], l[1]+l[3]+ex)
	x2 = np.minimum(img.shape[1], l[0]+l[2]+ex)

	return img[y1:y2, x1:x2]



if __name__ == "__main__":
	#image_file = "testBlots/wb18.jpg"
	# DF = DisplayFigure(image_file)

	Database = np.load("static/Database.npy").item()
	Matches = np.load("static/Matches.npy").item()
	figure = np.load("static/figure.npy")

	AV = AffineVisualizer(Database, Matches)
	AV.compute_transform(image_pair=[0,1], figure=figure)
	AV.construct_aligned()





