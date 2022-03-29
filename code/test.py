from sift import SIFT
from orb import ORB
from surf import SURF
import cv2


if __name__ == "__main__":
	image_array = []
	gray_image_array = []
	#TODO: This code must be changed based on max and min image id.
	#TODO: Add a console UI to control the sw
	for ix in range(6, 30):
		image_file_name = "dataset/cyl_image" + ( "0" + str(ix)  if ix < 10 else str(ix)) + ".png"
		image_array.append(cv2.imread(image_file_name))  

	stitchy = cv2.createStitcher()
	(dummy,output)=stitchy.stitch(image_array)

	if dummy != cv2.STITCHER_OK:
	# checking if the stitching procedure is successful
	# .stitch() function returns a true value if stitching is
	# done successfully
		print("stitching ain't successful")
	else:
		print('Your Panorama is ready!!!')

	# final output
	cv2.imshow('final result',output)

	cv2.waitKey(0)



