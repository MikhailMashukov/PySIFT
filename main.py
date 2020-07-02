from skimage.io import imread
from sift import SIFT

import argparse
import pickle
import os
from os.path import isdir

import matplotlib.pyplot as plt

# if 0:
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PySIFT')
	parser.add_argument('--input', type=str, dest='input_fname')
	parser.add_argument('--output', type=str, dest='output_prefix', default='Unknown',
						help='The prefix for the kp_pyr and feat_pyr files generated')
	parser.add_argument('--output_format', type=str, choices=['m', 'p'], default='p',
						help="'m' - generate .sift.feat and .sift.desc files in meshroom format, 'p' - source pickle.dump format")
	args = parser.parse_args()

	im = imread(args.input_fname)

	sift_detector = SIFT(im, num_octave=4, t_c=0.03, min_keypoints=10)
	_ = sift_detector.calc_features()
	kp_pyr = sift_detector.kp_pyr

	if not isdir('results'):
		os.mkdir('results')

	if args.output_format != 'm':
		pickle.dump(sift_detector.kp_pyr, open('results/%s_kp_pyr.pkl' % args.output_prefix, 'wb'))
		pickle.dump(sift_detector.feats, open('results/%s_feat_pyr.pkl' % args.output_prefix, 'wb'))
	else:
		sift_detector.writeFeaturesInMeshroomFormat('results/%s.sift.feat' % args.output_prefix)
		# writeFeaturesInMeshroomFormat('results/%s.sift.feat' % args.output_prefix, sift_detector.kp_pyr)
		sift_detector.writeDescriptorsInMeshroomFormat('results/%s.sift.desc' % args.output_prefix)

	_, axs = plt.subplots(2, (sift_detector.num_octave + 1) // 2)
	
	for i in range(sift_detector.num_octave):
		ax = axs[i // 2, i % 2]
		ax.imshow(im)

		scaled_kps = kp_pyr[i] * (2**i)
		if scaled_kps.shape[0] > 0:
			ax.scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=2.5)

	plt.show()
	pass          # Nice place for breakpoint
