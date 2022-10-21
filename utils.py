# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""

import numpy as np
import cv2
import pyqtgraph as pg
import pyqtgraph.opengl as pg3

class CameraPoses:
    def __init__(self, intrinsic, orbs=1000, draw_matches=True):
        self.K = intrinsic
        self.extrinsic = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)))
        self.P = self.K @ self.extrinsic
        self.orb = cv2.ORB_create(orbs)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.world_points = []
        self.camera_poses = np.zeros((1000, 4, 4))
        self.n_camera_poses = 0
        self.current_pose = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1)
        self.current_frame = None
        self.current_keypoints = None
        self.current_descriptors = None

        self.draw_matches = draw_matches

        self.track_plot_view = pg3.GLViewWidget()
        self.track_plot_view.show()
        xgrid = pg3.GLGridItem()
        self.camera_track = pg3.GLLinePlotItem()
        self.track_plot_view.addItem(self.camera_track)
        self.track_plot_view.addItem(xgrid)
        xgrid.rotate(90, 0, 1, 0)

    def track(self, frame):
        # Find the keypoints and descriptors with ORB
        kp, des = self.orb.detectAndCompute(frame, None)
        q1, q2 = self.get_matches(frame, kp, des)
        if q1 is not None:
            if len(q1) > 20 and len(q2) > 20:
                try:
                    self.update_pose(q1, q2)
                    self.current_frame = frame
                    self.current_keypoints = kp
                    self.current_descriptors = des
                except FloatingPointError:
                    pass

        if self.current_frame is None:
            self.current_frame = frame
            self.current_keypoints = kp
            self.current_descriptors = des

    @staticmethod
    def _form_transf(R, t):

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def get_world_points(self):
        return np.array(self.world_points)

    def get_matches(self, frame, kp, des):
        # Find matches
        if self.current_frame is not None and len(kp) > 6 and len(self.current_keypoints) > 6:
            matches = self.flann.knnMatch(self.current_descriptors, des, k=2)
            # We could also have used cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Find the matches which do not have a too high distance
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass

            if self.draw_matches:
                matching_result = cv2.drawMatches(self.current_frame, self.current_keypoints, frame, kp, good_matches,
                                                  None, flags=2)
                cv2.imshow('Matching features', matching_result)

            q1 = np.float32([self.current_keypoints[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
        else:
            q1 = q2 = None
        return q1, q2

    def update_pose(self, q1, q2):
        transf = self.get_pose(q1, q2)
        with np.errstate(invalid='raise'):
            self.current_pose = self.current_pose@transf
        hom_camera_pose = np.concatenate((self.current_pose, np.array([[0, 0, 0, 1]])), axis=0)
        self._append_to_camera_poses(hom_camera_pose)


    def _append_to_camera_poses(self, hom_camera_pose: np.ndarray):
        if self.n_camera_poses == self.camera_poses.shape[0]:
            self.camera_poses = np.concatenate((self.camera_poses, np.zeros((1000, 4, 4))), axis=0)
        self.camera_poses[self.n_camera_poses] = hom_camera_pose
        self.n_camera_poses += 1
        self.camera_track.setData(pos=self.camera_poses[:self.n_camera_poses, :3, 3])

    def get_pose(self, q1, q2):

        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.K, maxIters=10)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))

        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):

        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1, np.ndarray.flatten(t))
        T2 = self._form_transf(R2, np.ndarray.flatten(t))
        T3 = self._form_transf(R1, np.ndarray.flatten(-t))
        T4 = self._form_transf(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        # Homogenize K
        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)

    def decomp_essential_mat_old(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # self.world_points.append(Q1)

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        T = self._form_transf(R1, t)
        # Make the projection matrix
        P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        self.world_points.append(Q1)

        return [R1, t]
