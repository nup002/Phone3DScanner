# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple
import pyqtgraph.opengl as pg3
import cv2
import sys
import random as rand


class Cells:
    def __init__(self):
        self.pts = np.zeros((0, 2, 2), dtype=np.float32)

    def rand_pt(self):
        return self.pts[np.random.randint(0, self.pts.shape[0])]


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
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None

        self.draw_matches = draw_matches

        self.track_plot_view = pg3.GLViewWidget()
        self.track_plot_view.show()
        xgrid = pg3.GLGridItem()
        self.camera_track = pg3.GLLinePlotItem()
        self.track_plot_view.addItem(self.camera_track)
        self.track_plot_view.addItem(xgrid)
        xgrid.rotate(90, 0, 1, 0)

    def track(self, frame):
        y_bar, x_bar = np.array(frame.shape[:-1]) / 8

        # Find the keypoints and descriptors with ORB
        kp, des = self.orb.detectAndCompute(frame, None)
        q1, q2 = self.get_matches(frame, kp, des)

        if q1 is not None:
            if len(q1) > 20 and len(q2) > 20:
                try:
                    # -----> Initialise the grids and points array variables <----- #
                    grid = np.empty((8, 8), dtype=object)

                    # Place the points q1 and q2 into their respective cells
                    q1_regularized = np.copy(q1)
                    q1_regularized[:, 0] /= x_bar
                    q1_regularized[:, 1] /= y_bar
                    q1_regularized = q1_regularized.astype(np.int32)
                    for grid_idx in np.arange(8*8):
                        h = grid_idx % 8
                        w = grid_idx // 8
                        q1_in_grid = np.logical_and(q1_regularized[:, 0] == h, q1_regularized[:, 1] == w)
                        cell = Cells()
                        cell.pts = np.stack((q1[q1_in_grid], q2[q1_in_grid]), axis=-1)
                        grid[h, w] = cell

                    F = self.estimate_fundamental_matrix_RANSAC(q1, q2, grid, 0.05)
                    E = self.estimate_essential_matrix(self.K, F)
                    transf = self.transformation_from_essential_mat(E, q1, q2)
                    #transf = self.get_pose(q1, q2)
                    self.update_pose(transf)
                    self.previous_frame = frame
                    self.previous_keypoints = kp
                    self.previous_descriptors = des
                except FloatingPointError:
                    pass

        if self.previous_frame is None:
            self.previous_frame = frame
            self.previous_keypoints = kp
            self.previous_descriptors = des

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
        if self.previous_frame is not None and len(kp) > 6 and len(self.previous_keypoints) > 6:
            matches = self.flann.knnMatch(des, self.previous_descriptors, k=2)
            # We could also have used cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Only keep the descriptor matches which has a single good match and a single bad match.
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass

            if self.draw_matches:
                matching_result = cv2.drawMatches(frame, kp, self.previous_frame, self.previous_keypoints, good_matches,
                                                  None, flags=2)
                original_shape = matching_result.shape
                new_shape = (int(original_shape[1] * 0.3), int(original_shape[0] * 0.3))
                matching_result_downsized = cv2.resize(matching_result, new_shape)
                cv2.imshow('Matching features', matching_result_downsized)

            q1 = np.array([self.previous_keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float32)
            q2 = np.array([kp[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        else:
            q1 = q2 = None
        return q1, q2

    def update_pose(self, transf):
        with np.errstate(invalid='raise'):
            self.current_pose = self.current_pose @ transf
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
        q1 = np.float32(q1)
        q2 = np.float32(q2)
        E, mask = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.99)
        transf = self.transformation_from_essential_mat(E, q1, q2, mask)
        return transf

    def transformation_from_essential_mat(self, E, q1, q2, mask=None):
        # Decompose the Essential matrix into R and t
        # R, t = self.decomp_essential_mat(E, q1, q2)
        retval, R, t, mask = cv2.recoverPose(E, q1, q2, self.K, mask)
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

    @staticmethod
    def estimate_essential_matrix(K: np.array, F: np.array) -> np.array:
        E = K.T @ F @ K
        U,S,V = np.linalg.svd(E)
        S = [[1,0,0],[0,1,0],[0,0,0]]
        E = U @ S @ V
        return E

    def estimate_fundamental_matrix_RANSAC(self, q1, q2, grid, epsilon=0.05) -> list:
        max_inliers = 0
        fund_mat_best = []
        confidence = 0.99
        N = sys.maxsize
        count = 0
        ones = np.ones((len(q1), 1))
        x = np.hstack((q1, ones))
        x_ = np.hstack((q2, ones))
        while N > count:
            x_1, x_2 = self.get_rand8(grid)
            if x_1 is not None:
                fund_mat, _ = self.calculate_fundamental_matrix(np.array(x_1), np.array(x_2))
                e, e_ = x @ fund_mat.T, x_ @ fund_mat
                error = np.sum(e_ * x, axis=1, keepdims=True) ** 2 / np.sum(np.hstack((e[:, :-1], e_[:, :-1])) ** 2, axis=1,
                                                                            keepdims=True)
                inliers = error <= epsilon
                counter = np.sum(inliers)
                if max_inliers < counter:
                    max_inliers = counter
                    fund_mat_best = fund_mat
            I_O_ratio = counter / len(q1)
            if np.log(1 - (I_O_ratio ** 8)) == 0:
                continue
            N = np.log(1 - confidence) / np.log(1 - (I_O_ratio ** 8))
            count += 1
        return fund_mat_best

    @staticmethod
    def calculate_fundamental_matrix(pts_cf: np.array, pts_nf: np.array):
        F_CV, _ = cv2.findFundamentalMat(pts_cf, pts_nf, cv2.FM_8POINT)
        origin = np.mean(pts_cf, axis=0)
        origin_ = np.mean(pts_nf, axis=0)
        k = np.mean(np.sum((pts_cf - origin) ** 2, axis=1, keepdims=True) ** .5)
        k_ = np.mean(np.sum((pts_nf - origin_) ** 2, axis=1, keepdims=True) ** .5)
        k = np.sqrt(2.) / k
        k_ = np.sqrt(2.) / k_
        x = (pts_cf[:, 0].reshape((-1, 1)) - origin[0]) * k
        y = (pts_cf[:, 1].reshape((-1, 1)) - origin[1]) * k
        x_ = (pts_nf[:, 0].reshape((-1, 1)) - origin_[0]) * k_
        y_ = (pts_nf[:, 1].reshape((-1, 1)) - origin_[1]) * k_
        A = np.hstack((x_ * x, x_ * y, x_, y_ * x, y_ * y, y_, x, y, np.ones((len(x), 1))))
        U, S, V = np.linalg.svd(A)
        F = V[-1]
        F = np.reshape(F, (3, 3))
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ V
        T1 = np.array([[k, 0, -k * origin[0]], [0, k, -k * origin[1]], [0, 0, 1]])
        T2 = np.array([[k_, 0, -k_ * origin_[0]], [0, k_, -k_ * origin_[1]], [0, 0, 1]])
        F = T2.T @ F @ T1
        F = F / F[-1, -1]
        return F, F_CV

    @staticmethod
    def get_rand8(grid: npt.NDArray[Cells]):
        selected_grid_indexes = np.random.choice(8*8, 8, replace=False)
        unselected_grid_indexes = np.delete(np.array(np.arange(8*8)), selected_grid_indexes)

        rand8 = np.ndarray(shape=(8, 2), dtype=np.float32)
        rand8_ = np.ndarray(shape=(8, 2), dtype=np.float32)
        for n, index in enumerate(selected_grid_indexes):
            grid_cell = grid[index % 8, index // 8]
            if grid_cell.pts.shape[0] > 0:
                pt = grid_cell.rand_pt()
            else:
                rand_idx = np.random.randint(0, unselected_grid_indexes.shape[0])
                index = unselected_grid_indexes[rand_idx]
                grid_cell = grid[index % 8, index // 8]
                while grid_cell.pts.shape[0] == 0:
                    if unselected_grid_indexes.shape[0] > 0:
                        unselected_grid_indexes = np.delete(unselected_grid_indexes, rand_idx)
                        rand_idx = np.random.randint(0, unselected_grid_indexes.shape[0])
                        index = unselected_grid_indexes[rand_idx]
                        grid_cell = grid[index % 8, index // 8]
                    else:
                        rand_idx = np.random.randint(0, selected_grid_indexes.shape[0])
                        index = selected_grid_indexes[rand_idx]
                        grid_cell = grid[index % 8, index // 8]

                pt = grid_cell.rand_pt()
            rand8[n] = pt[:, 0]
            rand8_[n] = pt[:, 1]
        return rand8, rand8_
