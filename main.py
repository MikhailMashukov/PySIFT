from skimage.io import imread
from sift import SIFT

import argparse
import pickle
import os
from os.path import isdir

import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PySIFT')
	parser.add_argument('--input', type=str, dest='input_fname')
	parser.add_argument('--output', type=str, dest='output_prefix', help='The prefix for the kp_pyr and feat_pyr files generated')
	args = parser.parse_args()

	im = imread(args.input_fname)

	sift_detector = SIFT(im, num_octave=4, min_keypoints=10)
	_ = sift_detector.get_features()
	kp_pyr = sift_detector.kp_pyr

	if not isdir('results'):
		os.mkdir('results')

	pickle.dump(sift_detector.kp_pyr, open('results/%s_kp_pyr.pkl' % args.output_prefix, 'wb'))
	pickle.dump(sift_detector.feats, open('results/%s_feat_pyr.pkl' % args.output_prefix, 'wb'))

	_, axs = plt.subplots(2, (sift_detector.num_octave + 1) // 2)
	
	for i in range(sift_detector.num_octave):
		ax = axs[i // 2, i % 2]
		ax.imshow(im)

		scaled_kps = kp_pyr[i] * (2**i)
		if scaled_kps.shape[0] > 0:
			ax.scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=2.5)

	plt.show()
	pass          # Nice place for breakpoint
