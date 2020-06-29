import numpy as np
import numpy.linalg as LA

def dot_app(m0, m1):
	if len(m0.shape) == 4:
		s0,s1,s2,s3 = m0.shape
		s4 = m1.shape[-1]
		m0.shape = s0,s1*s2,s3   # Get m0 as 3D for temporary usage
		r = np.empty((s0,s1*s2,s4))
		for i in range(s0):
			r[i] = m0[i].dot(m1[i])
		r.shape = s0,s1,s2,s4
		m0.shape = s0,s1,s2,s3  # Put m0 back to 4D
	else:
		size = list(m0.shape)[:-1]
		r = np.empty(size)
		for i in range(m0.shape[0]):
			r[i] = m0[i].dot(m1[i])
	return r

def get_candidate_keypoints(D, w=16):
	candidates = []

	''' Start '''
	# These 2 lines aren't specified in the paper but it makes it so the extrema
	# are found within the entire octave. They are always found in the first or
	# last layer so I probably have something wrong with my DoG pyramid construction.
	D[:,:,0] = 0
	D[:,:,-1] = 0
	''' End '''
	
	# have to start at w//2 so that when getting the local w x w descriptor, we don't fall off
	# for i in range(w//2+1, D.shape[0]-w//2-1):
		# for j in range(w//2+1, D.shape[1]-w//2-1):

	r = range(w//2+1 + D.shape[0] * 6 // 16, D.shape[0] * 62 // 160 - w//2 - 1)   #d_
	for i in r:
		for j in range(w//2+1 + D.shape[0] // 4, D.shape[0] * 5 // 16 - w//2 - 1):

			for k in range(1, D.shape[2]-1):
				patch = D[i-1:i+2, j-1:j+2, k-1:k+2]
				if np.argmax(patch) == 13 or np.argmin(patch) == 13:
					candidates.append([i, j, k])

	return candidates

def localize_keypoint_impl(D, x, y, s):
	dx = (D[y,x+1,s]-D[y,x-1,s])/2.
	dy = (D[y+1,x,s]-D[y-1,x,s])/2.
	ds = (D[y,x,s+1]-D[y,x,s-1])/2.

	dxx = D[y,x+1,s]-2*D[y,x,s]+D[y,x-1,s]
	dxy = ((D[y+1,x+1,s]-D[y+1,x-1,s]) - (D[y-1,x+1,s]-D[y-1,x-1,s]))/4.
	dxs = ((D[y,x+1,s+1]-D[y,x-1,s+1]) - (D[y,x+1,s-1]-D[y,x-1,s-1]))/4.
	dyy = D[y+1,x,s]-2*D[y,x,s]+D[y-1,x,s]
	dys = ((D[y+1,x,s+1]-D[y-1,x,s+1]) - (D[y+1,x,s-1]-D[y-1,x,s-1]))/4.
	dss = D[y,x,s+1]-2*D[y,x,s]+D[y,x,s-1]

	J = np.array([dx, dy, ds])
	HD = np.array([
		[dxx, dxy, dxs],
		[dxy, dyy, dys],
		[dxs, dys, dss]])
	return J, HD

def localize_keypoint(D, x, y, s):
	J, HD = localize_keypoint_impl(D, x, y, s)
	offset = -LA.inv(HD).dot(J)	# I know you're supposed to do something when an offset dimension is >0.5 but I couldn't get anything to work.
	return offset, J, HD[:2,:2], x, y, s

def localize_keypoint_array(D, xs, ys, ss):
	Js, HDs = localize_keypoint_impl(D, xs, ys, ss)
	Js = np.transpose(Js, (1, 0))
	HDs = np.transpose(HDs, (2, 0, 1))
	invs = -LA.inv(HDs)
	offsets = dot_app(invs, Js)
	return offsets, Js, HDs[:, :2, :2], xs, ys, ss

def calc_contrasts(D, candidates):
	# vals = []
	assert len(candidates) > 0
	if not isinstance(candidates, np.ndarray):
		candidates = np.array(candidates)
	ys, xs, ss = candidates[:, 0], candidates[:, 1], candidates[:, 2]
	vals1 = localize_keypoint_array(D, xs, ys, ss)
	offsets = vals1[0]
	Js = vals1[1]
	contrasts = D[ys, xs, ss] + .5 * dot_app(Js, offsets)
	vals = [abs(contrasts), vals1[3], vals1[4], offsets, Js, vals1[2], vals1[5]]
		# offset, J, H, x, y, s = localize_keypoint
		# vals.append((abs(contrast), x, y, offset, J, H, s))
	return vals

def find_keypoints_for_DoG_octave(D, R_th, t_c, w, min_keypoints):
	candidates = get_candidate_keypoints(D, w)
	#print('%d candidate keypoints found' % len(candidates))
	if not candidates:
		print('D %s: %d candidates' % \
		  	  (D.shape, len(candidates)))
		return np.empty((0, 3))

	keypoints = []
	if 1:
		vals = calc_contrasts(D, candidates)
		# Returns abs(contrast), x, y, offset, J, H, s

	if 0:
		vals2 = []
		for i, cand in enumerate(candidates):
			y0, x0, s = cand[0], cand[1], cand[2]
			offset, J, H, x, y, s = localize_keypoint(D, x0, y0, s)
				# Currently x, y == x0, y0
			# offsets.append(offset)
			# Hs.append(H)

			contrast = D[y,x,s] + .5*J.dot(offset)
			vals2.append((abs(contrast), x, y, offset, J, H, s))
			# if abs(contrast) < t_c:
			# 	continue
		# vals2 = np.array(vals2)
		# vals = np.split(vals, axis=0)   # Not tested. We need to convert to list of arrays

		# Comparing
		for i in range(len(vals)):
			for j in range(len(vals2)):
				if np.sum(vals[i][j] - vals2[j][i]):
					print('Mismatch at element %d, %d' % (i, j))
			# diff = vals[i] - vals2[:, i]
			# if sum(diff) != 0:
			# 	print('Mismatch in array %d')

	filteredInds = np.where(vals[0] >= t_c)[0]
	# if i < 100:  #d_
	print('D %s candidates: %s' % (D.shape, \
		  [(vals[0][i], vals[1][i], vals[2][i]) \
		   for i in range(min(100, len(vals[0])))]))

	if len(filteredInds) < min_keypoints:
		sortedByContrastInds = vals[0].argsort(axis=0)
		if 1:
			import matplotlib
			import matplotlib.pyplot as plt

			print('Top candidates contrasts: %s' % (str(list(reversed(vals[0][sortedByContrastInds[-5:]])))))
			i = 5
			while i < len(sortedByContrastInds):
				print('Candidates contrast %d: %f' % (i, vals[0][sortedByContrastInds[-i]]))
				i *= 2
			norm = matplotlib.colors.Normalize(vmin=0, vmax=vals[0][sortedByContrastInds[-1]] / 2, clip=False)
			plt.scatter(vals[1], vals[2], c=vals[0],
						cmap='inferno', norm=norm, s=4)
			plt.colorbar()

			# ax = plt.subplot()
			plt.show()

		filteredInds = sortedByContrastInds[-min_keypoints:]
	else:
		filteredInds = filteredInds

	# Vals: abs(contrast), x, y, offset, J, H, s
	Hs = vals[5]
	ws, vs = LA.eig(Hs)
	rs = ws[:, 1] / ws[:, 0]
	Rs = (rs+1)**2 / rs
	kps = np.stack([vals[1], vals[2], vals[-1]], axis=1) + vals[3]

	for i in filteredInds:
		if Rs[i] > R_th: continue

		if kps[i][1] >= D.shape[0] or kps[i][0] >= D.shape[1]: continue # throw out boundary points because I don't want to deal with them

		keypoints.append(kps[i])

	# for i in filteredInds:
	# 	_, x, y, offset, J, H, s = [arr[i] for arr in vals]
	# 	w, v = LA.eig(H)
	# 	r = w[1]/w[0]
	# 	R = (r+1)**2 / r
	# 	if R > R_th: continue
    #
	# 	kp = np.array([x, y, s]) + offset
	# 	if kp[1] >= D.shape[0] or kp[0] >= D.shape[1]: continue # throw out boundary points because I don't want to deal with them
    #
	# 	keypoints.append(kp)

	print('D %s: %d candidates, %d after filter by t_c, %d keypoints' % \
		  	(D.shape, len(candidates), len(filteredInds), len(keypoints)))
	return np.array(keypoints)

def get_keypoints(DoG_pyr, R_th, t_c, w, min_keypoints):
    kps = []

    for D in DoG_pyr:
        kps.append(find_keypoints_for_DoG_octave(D, R_th, t_c, w, min_keypoints))

    return kps