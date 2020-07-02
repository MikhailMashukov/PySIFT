import array
import math
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve

from gaussian_filter import gaussian_filter
from gaussian_pyramid import generate_gaussian_pyramid
from DoG_pyramid import generate_DoG_pyramid
from keypoints import get_keypoints
from orientation import assign_orientation
from descriptors import get_local_descriptors

class SIFT(object):
    def __init__(self, im, s=3, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16,
                 min_keypoints=0):
        self.im = convolve(rgb2gray(im), gaussian_filter(s0))
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w
        self.min_keypoints = min_keypoints

        self.kp_pyr = None

    def calc_features(self):
        gaussian_pyr = generate_gaussian_pyramid(self.im, self.num_octave, self.s, self.sigma)
        DoG_pyr = generate_DoG_pyramid(gaussian_pyr)
        kp_pyr = get_keypoints(DoG_pyr, self.R_th, self.t_c, self.w, self.min_keypoints)
        feats = []

        for i, DoG_octave in enumerate(DoG_pyr):
            kp_pyr[i] = assign_orientation(kp_pyr[i], DoG_octave)
            feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        self.gaussian_pyr = gaussian_pyr   # Added by Mikhail
        self.DoG_pyr = DoG_pyr

        return feats

    def writeFeaturesInMeshroomFormat(self, fileName):
        if self.kp_pyr is None:
            self.calc_features()
        features = self.kp_pyr[0]       # x, y, s, orientation (0-360), in Meshroom/AliceVision terms these are features
        scale_mult = 1
        for feat2 in self.kp_pyr[1:]:
            scale_mult *= 2
            newFeat = np.copy(feat2)
            newFeat[:, 2] *= scale_mult
            features = np.concatenate([features, newFeat], axis=0)
        features[features[:, 3] < 0, 3] += 360
        # assert np.sum(np.logical_or(features[:, 3] < 0, features[:, 3] >= 360)) == 0
        features[:, 3] *= math.pi / 180

        with open(fileName, 'w') as outF:
            for feat in features:
                outF.write('%.3f %.3f %8f %.6f\n' % tuple(feat))

    def writeDescriptorsInMeshroomFormat(self, fileName):
        if self.kp_pyr is None:
            self.calc_features()
        descs = np.concatenate(self.feats, axis=0)
        ucharDescs = np.array(descs * 512, dtype=np.uint8)

        with open(fileName, 'wb') as outF:
            # data = array.array('B')  # create array of bytes.
            # data.append(len(descs).to_bytes(4, byteorder='little', signed=False))
            ucharDescs.tofile(outF)
            # descs.tofile(outF)
        with open(fileName, 'wb') as outF:
            # data = array.array('B')  # create array of bytes.
            # data.append(len(descs).to_bytes(4, byteorder='little', signed=False))
            outF.write(len(descs).to_bytes(8, byteorder='little', signed=False))
            ucharDescs.tofile(outF)

            # for desc in descs:
            #     np.save(outF, desc)


