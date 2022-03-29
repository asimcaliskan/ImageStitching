from audioop import cross
from pickletools import uint8
from turtle import right
import numpy as np
import cv2

class SIFT:
    def __init__(self, image_array, gray_image_array):
        self.image_array = image_array
        self.gray_image_array = gray_image_array
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
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

    def calculate_homography_matrix(self, train_points, query_points):

        #p = [src_points, dst_points]
        x_1 = [train_points[0][0],query_points[0][0]]
        y_1 = [train_points[0][1],query_points[0][1]]
        x_2 = [train_points[1][0],query_points[1][0]]
        y_2 = [train_points[1][1],query_points[1][1]]
        x_3 = [train_points[2][0],query_points[2][0]]
        y_3 = [train_points[2][1],query_points[2][1]]
        x_4 = [train_points[3][0],query_points[3][0]]
        y_4 = [train_points[3][1],query_points[3][1]]

        P = np.array([
            [-x_1[0], -y_1[0], -1, 0, 0, 0, x_1[0]*x_1[1], y_1[0]*x_1[1], x_1[1]],
            [0, 0, 0, -x_1[0], -y_1[0], -1, x_1[0]*y_1[1], y_1[0]*y_1[1], y_1[1]],
            [-x_2[0], -y_2[0], -1, 0, 0, 0, x_2[0]*x_2[1], y_2[0]*x_2[1], x_2[1]],
            [0, 0, 0, -x_2[0], -y_2[0], -1, x_2[0]*y_2[1], y_2[0]*y_2[1], y_2[1]],
            [-x_3[0], -y_3[0], -1, 0, 0, 0, x_3[0]*x_3[1], y_3[0]*x_3[1], x_3[1]],
            [0, 0, 0, -x_3[0], -y_3[0], -1, x_3[0]*y_3[1], y_3[0]*y_3[1], y_3[1]],
            [-x_4[0], -y_4[0], -1, 0, 0, 0, x_4[0]*x_4[1], y_4[0]*x_4[1], x_4[1]],
            [0, 0, 0, -x_4[0], -y_4[0], -1, x_4[0]*y_4[1], y_4[0]*y_4[1], y_4[1]],
            ])
        [U, S, Vt] = np.linalg.svd(P)
        return Vt[-1].reshape(3, 3)

    def show_image(im):
        cv2.imshow("asd", im)
        cv2.waitKey(0)

    def stitch(self, left_image, right_image):
        left_image_key_points, left_image_descriptor = self.sift.detectAndCompute(left_image, None)
        right_image_key_points, right_image_descriptor = self.sift.detectAndCompute(right_image, None)

        matches = self.brute_force_matcher.knnMatch(left_image_descriptor, right_image_descriptor, k=2)            
        
        good = []
        for m in matches:
            if (m[0].distance < 0.5*m[1].distance):
                good.append(m)
        matches = np.asarray(good)

        if (len(matches[:,0]) >= 4):
            src = np.float32([ left_image_key_points[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            dst = np.float32([ right_image_key_points[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        else:
            raise AssertionError('Cant find enough keypoints.')

        homography_matrix, masked = cv2.findHomography(src, dst, cv2.RANSAC)
        
        cv2.imshow("l",  left_image)
        cv2.waitKey(0)

        cv2.imshow("r",  right_image)
        cv2.waitKey(0)

        destination = cv2.warpIma(left_image, homography_matrix, (left_image.shape[1] + right_image.shape[1], right_image.shape[0]))
        cv2.imshow("d",  destination)
        cv2.waitKey(0)

        destination[0: right_image.shape[0], 0: right_image.shape[1]] = right_image

        return destination


    def stitch_images(self):
        #the algorithm starts stitching from last image to first image
        stitched_image = self.gray_image_array[-1]
        for image_index in range(len(self.gray_image_array) - 1, 1, -1):
            #train image will be changed based on request_image by homography matrix
            train_image = self.gray_image_array[image_index]
            query_image = self.gray_image_array[image_index - 1]

            #KEY POINT DETECTION BEGIN
            train_image_kps, train_image_descs = self.sift.detectAndCompute(train_image, None)
            query_image_kps, query_image_descs = self.sift.detectAndCompute(query_image, None)
            #KEY POINT DETECTION END

            #KEY POINT MATCHING BEGIN
            raw_matches = self.brute_force_matcher.knnMatch(query_image_descs, train_image_descs, k=2)     
            matches = []
            for m1, m2 in raw_matches:
                if m1.distance < m2.distance * self.distance_ratio:
                    matches.append(m1)
            
            #number of matches must be bigger than 4 to create homography matrix
            if len(matches) < 4:
                raise ("Number of matches is smaller than 4!")
            #KEY POINT MATCHING END
            
            #HOMOGRAPHY MATRIX BEGIN
            train_image_kps_list = np.float32([kp.pt for kp in train_image_kps])
            query_image_kps_list = np.float32([kp.pt for kp in query_image_kps])

            train_points = np.float32([train_image_kps_list[match.trainIdx] for match in matches]) 
            query_points = np.float32([query_image_kps_list[match.queryIdx] for match in matches])
            
            H = self.calculate_homography_matrix(train_points, query_points)#, cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold)
            #(H, _) = cv2.findHomography(train_points, query_points, cv2.RANSAC, self.ransac_reproj_threshold)
            #HOMOGRAPHY MATRIX END

            #IMAGE WARPING BEGIN
            stitched_image = cv2.warpPerspective(stitched_image, H, (query_image.shape[1] + stitched_image.shape[1], query_image.shape[0]))
            stitched_image[0: query_image.shape[0], 0: query_image.shape[1]] = query_image
            #IMAGE WARPING END

        cv2.imshow("--", stitched_image)
        cv2.waitKey(0)
