from random import randint
import numpy as np
import cv2

DEBUG = False

ass = np.array([2, 3, 1])
print((1 / ass[2]) * ass)


def line_point_distance(p1, p2, point):
	return np.linalg.norm(np.cross((p1[0] - p2[0], p1[1] - p2[1]), (point[0] - p2[0], point[1] - p2[1]))) / np.linalg.norm((p1[0] - p2[0], p1[1] - p2[1]))


def ransac(train_points: np.ndarray, query_points: np.ndarray):
	#translate train_points to separate them from query_points
	MAX_DISTANCE = 500
	train_points[:, :] += 500

	if DEBUG:
		empty = np.zeros((1000, 1200, 3))
		for dot in train_points:
			empty = cv2.circle(empty, (dot[0], dot[1]), 2, (255, 0, 255), 2)

		for dot in query_points:
			empty = cv2.circle(empty, (dot[0], dot[1]), 2, (0, 0, 255), 2)
		cv2.imshow("---", empty)
		cv2.waitKey(0)

	number_of_train_pts = len(train_points)
	number_of_query_pts = len(query_points)



	train_counter = 0
	success = 0
	max_success = 0
	detected_q_pt = None
	detected_t_pt = None
	while train_counter < 100:
		index = randint(0, min(number_of_train_pts, number_of_query_pts) - 1)

		train_pt = train_points[index]
		query_pt = query_points[index]
		
		for q_pt in query_points:
			distance = line_point_distance(query_pt, train_pt, q_pt)
			if distance < MAX_DISTANCE:
				success += 1
		for t_pt in train_points:
			distance = line_point_distance(query_pt, train_pt, t_pt)
			if distance < MAX_DISTANCE:
				success += 1
		
		if success > max_success:
			max_success = success
			detected_t_pt = train_pt
			detected_q_pt = query_pt	
		
		train_counter += 1


	if DEBUG:
		empyt = cv2.line(empty,tuple(detected_q_pt), tuple(detected_t_pt), (255, 0, 0), 1)
		cv2.imshow("---", empty)
		cv2.waitKey(0)
		
	detected_t_pt = [detected_t_pt[0] - 500, detected_t_pt[1] - 500]
	return (tuple(detected_q_pt), tuple(detected_t_pt[:]))

"""
#LINE EQUATION BEGIN
#y = mx + b
#m = (y_t - y_q) / (x_t - x_q)
m = (train_pt[1] - query_pt[1]) / (train_pt[0] - query_pt[0])

#b = mx - y
b = m * train_pt[0] - train_pt[1]
#LINE EQUATION END

#Distance = (| a*x1 + b*y1 + c |) / (sqrt( a*a + b*b))
"""