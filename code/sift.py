from audioop import cross
from pickletools import uint8
from turtle import right
import ransac
import numpy as np
import cv2

THRESHOLD = 0.9
NUM_ITERS = 1500

class SIFT:
    def __init__(self, image_array, gray_image_array):
        self.image_array = image_array
        self.gray_image_array = gray_image_array
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.distance_ratio = 0.75
        self.ransac_reproj_threshold = 5

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

    def calculate_homography_matrix(self, pairs):
        """
        w*x_d        x_s
        w*y_d =  H . y_s
        w            1
        
        """
        A = []
        for x_l, y_l, x_r, y_r in pairs:
            A.append([x_r, y_r, 1, 0, 0, 0, -x_l * x_r, -x_l * y_r, -x_l])
            A.append([0, 0, 0, x_r, y_r, 1, -y_l * x_r, -y_l * y_r, -y_l])
        A = np.array(A)

        #singular value decomposition 
        (U, S, V) = np.linalg.svd(A)

        #V.shape = (9, 9), The last element of the V is the smallest eigenvector.
        H = np.reshape(V[-1], (3, 3))

        #use w value
        H = (1 / H.item(8)) * H

        return H

    def dist(self, pair, H):
        """ Returns the geometric distance between a pair of points given the
        homography H. """
        # points in homogeneous coordinates
        p1 = np.array([pair[0], pair[1], 1])#from left image
        p2 = np.array([pair[2], pair[3], 1])#from right image

        p2_estimate = np.dot(H, np.transpose(p2))
        p2_estimate = (1 / p2_estimate[2]) * p2_estimate
        
        return np.sqrt((p1[0] - p2_estimate[0])** 2 + (p1[1] - p2_estimate[1]) ** 2) 

    def RANSAC(self, point_map, threshold=THRESHOLD, verbose=True):
        bestInliers = set()
        homography = None
        p = None
        for i in range(NUM_ITERS):
            # randomly choose 4 points from the matrix to compute the homography
            pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

            H = self.calculate_homography_matrix(pairs)
            inliers = {(c[0], c[1], c[2], c[3]) for c in point_map if self.dist(c, H) < 2}

            if len(inliers) > len(bestInliers):
                print(len(inliers))
                bestInliers = inliers
                homography = H
                p = pairs
                #if len(bestInliers) > (len(point_map) * threshold): break
        return homography, pairs


    def stitch_images(self):
        #the algorithm starts stitching from last image to first image
        #[query_image] <-stitch-> [train_image]
        stitched_image = self.gray_image_array[0]
        for image_index in range(len(self.gray_image_array) - 1):
            #train image will be changed based on request_image by homography matrix
            left_image = self.gray_image_array[image_index]
            right_image = self.gray_image_array[image_index + 1]

            #KEY POINT DETECTION BEGIN
            left_image_kps, left_image_descs = self.sift.detectAndCompute(left_image, None)
            right_image_kps, right_image_descs = self.sift.detectAndCompute(right_image, None)
            #KEY POINT DETECTION END

            #KEY POINT MATCHING BEGIN
            #match: query_image -> train_image
            raw_matches = self.brute_force_matcher.match(left_image_descs, right_image_descs)   
            """
            result_image = cv2.drawMatches(left_image, left_image_kps,
                                     right_image, right_image_kps,
                                     raw_matches, 
                                     None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            cv2.imshow('SIFT FEATURE MATCHES ', result_image)
            cv2.waitKey(0)
            """
            
            #KEY POINT MATCHING END
            
            #HOMOGRAPHY MATRIX BEGIN
            #ransac.ransac(train_points.copy(), query_points.copy())

            #(H, _) = cv2.findHomography(train_points, query_points, cv2.RANSAC, self.ransac_reproj_threshold)
            #print(H)
            combined_matches = np.array([[
                left_image_kps[match.queryIdx].pt[0],
                left_image_kps[match.queryIdx].pt[1],
                right_image_kps[match.trainIdx].pt[0],
                right_image_kps[match.trainIdx].pt[1]] for match in raw_matches])
            
            H, pairs = self.RANSAC(combined_matches)

            for c1, c2, c3, c4 in pairs:
                ss = cv2.circle(left_image, (int(c1), int(c2)), 3, (0, 0, 0), 4)
                cv2.imshow("--aa", ss)
                cv2.waitKey(0)
                ss = cv2.circle(right_image, (int(c3), int(c4)), 3, (0, 0, 0), 4)
                cv2.imshow("--ab", ss)
                cv2.waitKey(0)

            #H = self.calculate_homography_matrix(train_points[10: 14], query_points[10: 14])#, cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold)
            #HOMOGRAPHY MATRIX END

            #IMAGE WARPING BEGIN
            stitched_image = cv2.warpPerspective(right_image, H, (right_image.shape[1] + left_image.shape[1], right_image.shape[0]))
            cv2.imshow("a-", stitched_image)
            cv2.waitKey(0)
            stitched_image[0: left_image.shape[0], 0: left_image.shape[1]] = left_image
    
            #IMAGE WARPING END

        cv2.imshow("--", stitched_image)
        cv2.waitKey(0)
