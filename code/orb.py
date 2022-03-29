import numpy as np
import cv2

class ORB:
    def __init__(self, image_array, gray_image_array):
        self.image_array = image_array
        self.gray_image_array = gray_image_array
        self.orb = cv2.ORB_create()
        self.brute_force_matcher = cv2.BFMatcher()

    def show_feature_points(self):
        for image_index in range(len(self.gray_image_array) - 1):
            sub_image1 = self.gray_image_array[image_index]
            sub_image2 = self.gray_image_array[image_index + 1]

            sub_image1_key_points = self.orb.detect(sub_image1, None)
            sub_image2_key_points = self.orb.detect(sub_image2, None)
            
            sub_image1 = cv2.drawKeypoints(sub_image1, sub_image1_key_points, sub_image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            sub_image2 = cv2.drawKeypoints(sub_image2, sub_image2_key_points, sub_image2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow('ORB FEATURES ' + str(image_index) + "-" + str(image_index + 1), np.concatenate((sub_image1, sub_image2), axis=1))
            cv2.waitKey(0)

    def show_feature_matches(self):
        for image_index in range(len(self.gray_image_array) - 1):
            sub_image1 = self.gray_image_array[image_index]
            sub_image2 = self.gray_image_array[image_index + 1]

            sub_image1_key_points, sub_image1_descriptor = self.orb.detectAndCompute(sub_image1, None)
            sub_image2_key_points, sub_image2_descriptor = self.orb.detectAndCompute(sub_image2, None)

            matches = self.brute_force_matcher.knnMatch(sub_image1_descriptor, sub_image2_descriptor, k=2)            
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])

            result_image = cv2.drawMatchesKnn(sub_image1, sub_image1_key_points,
                                     sub_image2, sub_image2_key_points,
                                     good_matches, 
                                     None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.imshow('ORB FEATURE MATCHES ' + str(image_index) + "-" + str(image_index + 1), result_image)
            cv2.waitKey(0)

    