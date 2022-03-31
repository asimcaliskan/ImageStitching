import numpy as np

def calculate_homography_matrix(combined_matches):
	"""
	This function calculates homography matrix by using SVD approach.
	It returns 3x3 homography matrix.
	|w*x_l|         |x_r|
	|w*y_l| = |H| . |y_r|
	|  w  |         | 1 |

	_r means right, _l means left
	"""

	A = []
	for x_l, y_l, x_r, y_r in combined_matches:
		A.append([x_r, y_r, 1, 0, 0, 0, -x_l * x_r, -x_l * y_r, -x_l])
		A.append([0, 0, 0, x_r, y_r, 1, -y_l * x_r, -y_l * y_r, -y_l])
	A = np.array(A)

	#singular value decomposition 
	(U, S, V) = np.linalg.svd(A)

	#V.shape = (9, 9), The last element of the V is the smallest eigenvector.
	H = np.reshape(V[-1], (3, 3))

	#normalization
	H = (1 / H.item(8)) * H

	return H

def euclidean_distance(combined_matches, H):
	"""
	p1_estimate = H.p2
	This function calculates euclidean distance between p1_estimate and p1.
	if this distance is too small. that means H matrix did it job.
	it returns distance
	"""

	#homogeneous coordinates
	p1 = np.array([combined_matches[0], combined_matches[1], 1])#from left image
	p2 = np.array([combined_matches[2], combined_matches[3], 1])#from right image

	p1_estimate = np.dot(H, np.transpose(p2))
	if p1_estimate[2] == 0:
		p1_estimate = (1 / 0.000001) * p1_estimate
	else:
		p1_estimate = (1 / p1_estimate[2]) * p1_estimate
	
	return np.sqrt((p1[0] - p1_estimate[0])** 2 + (p1[1] - p1_estimate[1]) ** 2) 

def ransac(combined_matches, MAXIMUM_ITERATION, MAX_DISTANCE):
	"""
	This function:
		-selects random 4 point in combined_matches
		-builds a homography matrix from these random points
		-applies this homography matrix on combined_matches
		-measure the performance based on the number of inliers
		-tries to find best homography matrix and returns it 
	"""
	best_number_of_inliers = 0
	best_H = None
	for _ in range(MAXIMUM_ITERATION):
		#choose 4 random points in combined_matches
		pairs = [combined_matches[i] for i in np.random.choice(len(combined_matches), 4)]

		H = calculate_homography_matrix(pairs)
		inliers = {(c[0], c[1], c[2], c[3]) for c in combined_matches if euclidean_distance(c, H) < MAX_DISTANCE}

		if len(inliers) > best_number_of_inliers:
			best_number_of_inliers = len(inliers)
			best_H = H

	return best_H
