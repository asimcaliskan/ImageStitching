from utils import ransac
import numpy as np
import cv2

MAXIMUM_ITERATION = 1000
MAX_DISTANCE = 2

class SIFT:
    def __init__(self, image_array, gray_image_array):
        self.image_array = image_array
        self.gray_image_array = gray_image_array
        self.number_of_images = len(self.gray_image_array)
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def show_feature_points(self):
        for image_index in range(len(self.gray_image_array) - 1):
            sub_image1 = self.gray_image_array[image_index]
            sub_image2 = self.gray_image_array[image_index + 1]

            sub_image1_key_points = self.sift.detect(sub_image1, None)
            sub_image2_key_points = self.sift.detect(sub_image2, None)
            
            sub_image1 = cv2.drawKeypoints(sub_image1, sub_image1_key_points, sub_image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            sub_image2 = cv2.drawKeypoints(sub_image2, sub_image2_key_points, sub_image2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow('SIFT FEATURES ' + str(image_index) + "-" + str(image_index + 1), np.concatenate((sub_image1, sub_image2), axis=1))
            cv2.waitKey(0)

    def show_feature_matches(self):
        for image_index in range(len(self.gray_image_array) - 1):
            sub_image1 = self.gray_image_array[image_index]
            sub_image2 = self.gray_image_array[image_index + 1]

            sub_image1_key_points, sub_image1_descriptor = self.sift.detectAndCompute(sub_image1, None)
            sub_image2_key_points, sub_image2_descriptor = self.sift.detectAndCompute(sub_image2, None)

            matches = self.brute_force_matcher.knnMatch(sub_image1_descriptor, sub_image2_descriptor, k=2)            
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])

            result_image = cv2.drawMatchesKnn(sub_image1, sub_image1_key_points,
                                     sub_image2, sub_image2_key_points,
                                     good_matches, 
                                     None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.imshow('SIFT FEATURE MATCHES ' + str(image_index) + "-" + str(image_index + 1), result_image)
            cv2.waitKey(0)

 

    def stitch_images(self):
        """
        this function stitch images from left to right.
        it uses:
            opencv.detectAndCompute
            opencv.BFmatcher.match
            opencv.warpPerspective
            homography matrix with RANSAC algorithm

        How does it work?
            ->assigns last image as stitched_image
            ->finds match points for left_image and right_image
            ->calculates homography matrix
            ->applies homography matrix on stitched_image
            ->stitchs left_image and stitched_image and this image would be new stitched image
        """


        stitched_image = self.gray_image_array[-1]
        for image_index in range(self.number_of_images - 1, 0, -1):
     
            left_image = self.gray_image_array[image_index - 1]
            right_image = self.gray_image_array[image_index]

            #KEY POINT DETECTION BEGIN
            left_image_kps, left_image_descs = self.sift.detectAndCompute(left_image, None)
            right_image_kps, right_image_descs = self.sift.detectAndCompute(right_image, None)
            #KEY POINT DETECTION END

            #KEY POINT MATCHING BEGIN
            raw_matches = self.brute_force_matcher.match(left_image_descs, right_image_descs)   
            #KEY POINT MATCHING END
            
            #HOMOGRAPHY MATRIX BEGIN
            combined_matches = np.array([[
                left_image_kps[match.queryIdx].pt[0],
                left_image_kps[match.queryIdx].pt[1],
                right_image_kps[match.trainIdx].pt[0],
                right_image_kps[match.trainIdx].pt[1]] for match in raw_matches])
            
            H = ransac(combined_matches, MAXIMUM_ITERATION, MAX_DISTANCE)
            #HOMOGRAPHY MATRIX END

            #IMAGE WARPING BEGIN
            stitched_image = cv2.warpPerspective(stitched_image, H, (stitched_image.shape[1] + left_image.shape[1], stitched_image.shape[0]))
            stitched_image[0: left_image.shape[0], 0: left_image.shape[1]] = left_image
            #IMAGE WARPING END

        cv2.imshow("SIFT-Stitched Image", stitched_image)
        cv2.waitKey(0)
