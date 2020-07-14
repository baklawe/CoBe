from __future__ import print_function

import pickle
import cv2 as cv
import numpy as np
import scipy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import urllib
import bz2
import os
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares


class RelativePose:
    # Defines the pose between c2 and c1 (p_c2 -> p_c1)
    def __init__(self, R, t, c1, c2):
        self.R = R
        self.t = t
        self.c1 = c1
        self.c2 = c2

    def append_alignment(self, new_relativePose):
        if new_relativePose.c1 == self.c2:
            self.t = self.t + self.R @ new_relativePose.t
            self.R = self.R @ new_relativePose.R
            self.c2 = new_relativePose.c2
        elif new_relativePose.c2 == self.c1:
            self.t = new_relativePose.t + new_relativePose.R @ self.t
            self.R = new_relativePose.R @ self.R
            self.c1 = new_relativePose.c1
        else:
            print("Unrelated new relative pose !")

    def update_scale(self, scale):
        self.t = self.t * scale


class LEDInfo:
    def __init__(self):
        self.loc = []
        self.frame = []

    def add(self, loc, frame_id):
        self.loc.append(loc)
        self.frame.append(frame_id)


class PC:
    def __init__(self, pc, codes):
        self.pc = pc
        self.codes = codes

    def get_3D(self, led_codes):
        #index = self.codes.index(led_codes)
        # If led_codes is an int, put it inside a list
        if type(led_codes) is int:
            led_codes = [led_codes]

        # Get the indices of led_codes from the stored indices (self.codes). returned indices are sorted according to the
        # order given in led_codes
        indices = []
        for code in led_codes:
            indices += [self.codes.index(code)]
        # print(f'self.codes: {self.codes}, led_codes: {led_codes}, indices:{indices} ')
        return self.pc[:,indices]

    def update_scale(self, scale):
        print("Updating scale...")
        self.pc = self.pc * scale

    # def apply_alignment(self, rp:RelativePose):
    #     self.pc = rp.R @ self.pc - rp.t

    def apply_alignment(self, R, t, new_pc, s=1):
        #TODO: verify this function
        mu_pc = np.mean(new_pc, axis=1).reshape(-1, 1)
        # mu_pc = np.mean(self.pc, axis=1).reshape(-1, 1)
        pc_scaled = (self.pc - mu_pc) * s + mu_pc
        self.pc = R @ pc_scaled - t


    def plot_pc(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        # ax.scatter(triangulatedPoints[0, :], triangulatedPoints[1, :], triangulatedPoints[2, :])
        for i in range(self.pc.shape[1]):  # plot each point + it's index as text above
            ax.scatter(self.pc[0, i], self.pc[1, i], self.pc[2, i], color='b')
            ax.text(self.pc[0, i], self.pc[1, i], self.pc[2, i], '%s' %
                    (str(self.codes[i])), size=10, zorder=1, color='k')
        # ax.set_xlim3d(-30, 30)
        # ax.set_ylim3d(-30, 30)
        # ax.set_zlim3d(80, 140)
        # ax.set_aspect('equal')
        plt.show()


class SFM:
    def __init__(self, led_code_list=None, data_path=None, calib_path=None):
        if led_code_list is not None:
            self.led_code_list = led_code_list
        else:
            self.led_code_list = [7, 51, 85, 127, 265, 335, 467, 741, 959, 1207, 1753, 2007, 2211, 2293,2733,2875]

        if data_path is not None:
            self.data_path = data_path
        else:
            self.data_path = '/home/roman/PycharmProjects/psEyeTesting/code/sfm/'

        if calib_path is not None:
            self.calib_path = calib_path
        else:
            self.calib_path = 'calib_12.11.19.npz'

        with np.load(self.calib_path) as X:
            self.mtx, self.distortion, self.rvecs, self.tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

        # Dict. that wil store all the occurences (pixels) of every LED code
        self.first_frame = True
        self.led_graph = {}
        for i in self.led_code_list:
            self.led_graph[i] = LEDInfo()

        # Dict. that will store the relative pose for every new frame (relative to the point cloud's world)
        self.frame_graph = {}

        # The final point cloud
        self.pc = None

    # Store two led codes + actual distace between them. This is used to fix the final scale of the final PC
    def known_scale(self, led1, led2, dist):
        if led1 not in self.led_code_list or led2 not in self.led_code_list:
            print("Bad led number...")
        else:
            self.scale_led1 = led1
            self.scale_led2 = led2
            self.dist = dist

    def inverse_Rt(self,R,t):
        k = np.concatenate([R, t], axis=1)
        k = np.concatenate([k, np.array([[0,0,0,1]])], axis=0)
        inv_k = np.linalg.inv(k)
        return inv_k[:3,:3], inv_k[:3,3].reshape(3,1)


    # Compute optimal rotation, translation and scale between new_pc and self.pc.
    # common_codes contains the codes of the leds in common between new_pc and self.pc
    def comp_alignment_scale(self, new_pc:PC, common_codes):
        # print(f'common codes:{common_codes}')
        P = new_pc.get_3D(common_codes)     # 3 X N
        Q = self.pc.get_3D(common_codes)    # 3 X N

        # Compute optimal translation t* = mu(P) - mu(Q)
        mu_P = np.mean(P, axis=1).reshape(-1, 1)
        mu_Q = np.mean(Q, axis=1).reshape(-1, 1)
        P_centered = P - mu_P
        Q_centered = Q - mu_Q

        # Compute optimal rotation matrix
        # PQ^T = UDV^T -> R = V*(1,0,0;0,1,0;0,0,det(VU^T))*U
        U, D, V = LA.svd(P_centered @ Q_centered.transpose())
        V = V.transpose()
        H = np.eye(3)
        H[-1, -1] = np.linalg.det(V @ U.transpose())
        # print(f'H is :{H}')
        R = V @ H @ U.transpose()
        t = (R @ mu_P) - mu_Q

        # print(f't:{t}')

        # Compute optimal scale s = sum {p_i^Tq_i} / sum ||p_i||^2 on the aligned and centered P
        P_aligned = R @ P - t
        P_aligned_centered = P_aligned - np.mean(P_aligned,axis=1).reshape(-1,1)
        s = np.trace(P_aligned_centered.transpose() @ Q_centered) / np.sum(np.linalg.norm(P_aligned_centered, axis=0) ** 2)
        return R, t, s

    # Given a new computer point cloud with a new_pose (between the last frame and the new frame), append the new point
    # cloud to the existing one (self.pc). This includes fixing the scale + alignment of the new PC to match those of the
    # existing PC.
    def merge_point_cloud(self, new_pc:PC, new_pose:RelativePose):
        if self.pc is None:
            self.pc = new_pc
            self.relative_pose = new_pose
            self.frame_graph[new_pose.c1] = [np.eye(3), np.zeros((3, 1))]
            self.frame_graph[new_pose.c2] = [new_pose.R,new_pose.t]
        else:
            # Compute the scale of the new pc by finding a pair of led codes that appear both in the existing pc and the
            # new point cloud, and take the scale from the existing point cloud
            common_pc_codes = [i for i in self.pc.codes if i in new_pc.codes]
            common_index = [new_pc.codes.index(i) for i in common_pc_codes]
            if len(common_pc_codes) < 2:
                print("Not enough common leds !!!!!!!")
                return
            else:
                common_led1 = common_pc_codes[0]
                common_led2 = common_pc_codes[1]

                # new_pc_scale = np.linalg.norm(self.pc.get_3D(common_led1)-self.pc.get_3D(common_led2))/np.linalg.norm(new_pc.get_3D(common_led1)-new_pc.get_3D(common_led2))
                # new_pc.update_scale(new_pc_scale)
                # new_pose.update_scale(new_pc_scale)
                # new_pc.apply_alignment(self.relative_pose)

                # Compute the relative R,t and scale s between the new_pc and self.pc
                R, t, s = self.comp_alignment_scale(new_pc,common_pc_codes)
                # Align the new_pc with the global self.pc
                new_pc.apply_alignment(R, t, new_pc.get_3D(common_pc_codes), s)

                # plot merging the point clouds
                fig = plt.figure()
                ax = Axes3D(fig)
                # ax.scatter(triangulatedPoints[0, :], triangulatedPoints[1, :], triangulatedPoints[2, :])
                for i in range(new_pc.pc.shape[1]):  # plot each point + it's index as text above
                    ax.scatter(new_pc.pc[0, i], new_pc.pc[1, i], new_pc.pc[2, i], color='r')
                    ax.text(new_pc.pc[0, i], new_pc.pc[1, i], new_pc.pc[2, i], '%s' %
                            (str(new_pc.codes[i])), size=10, zorder=1, color='r')

                for i in range(self.pc.pc.shape[1]):  # plot each point + it's index as text above
                    ax.scatter(self.pc.pc[0, i], self.pc.pc[1, i], self.pc.pc[2, i], color='b')
                    ax.text(self.pc.pc[0, i], self.pc.pc[1, i], self.pc.pc[2, i], '%s' %
                            (str(self.pc.codes[i])), size=10, zorder=1, color='k')
                ax.set_title(f'frame {new_pose.c1} and {new_pose.c2} vs frame {self.relative_pose.c1} to {self.relative_pose.c2}')
                plt.show()

                #TODO: Take the R,t from the point cloud alignment, and compute the inverse transformation:
                # M = [R, t; [0,0,0,1]], M2 = inv(M), R2 = M2[:3,:3], t2 = M2[:3,3]
                # Use R2 t2 to initialize the camera transformation, and not R,t
                # R2 should be pretty similar to the relative_pose.R from the firs frame, and to the concatination of relative_pose.R in the other frames

                # Apply the scale to the new relative t ???????????????
                # self.relative_pose.R = -R  # new_pose (of the new PC) is taken from the PC alignment and not from the essential decomposition
                # self.relative_pose.t = t
                invR, invt = self.inverse_Rt(R,t)
                # print(f"self.relative_pose.R: {self.relative_pose.R}, invR: {invR}")
                # print(f"self.relative_pose.R: {self.relative_pose.t}, invR: {invt}")
                self.frame_graph[new_pose.c1] = [self.relative_pose.R, self.relative_pose.t]
                # self.frame_graph[new_pose.c1] = [invR, invt]

                new_pose.update_scale(s) # make sure if this is needed
                self.relative_pose.append_alignment(new_pose)
                self.frame_graph[new_pose.c2] = [self.relative_pose.R,self.relative_pose.t]  # override this for every camera, except for last one




                new_pc.codes = [new_pc.codes[i] for i in range(len(new_pc.codes)) if new_pc.codes[i] not in common_pc_codes]
                # new_pc.codes = [x for x in new_pc.codes if x not in common_pc_codes]
                self.pc.codes.extend(new_pc.codes)
                self.pc.pc = np.concatenate([self.pc.pc, np.delete(new_pc.pc,common_index,1)],1)

    # Build the data structure for the optimization function as follows:
    # 4 structures: frame_idx_list, code_idx_list, code_2d_list, code_3d_list
    # code_idx_list[i] - the led number of some observed led (every LED_CODE = running counter)
    # frame_idx_list[i] - contains the frame index of the observed led code_idx_list[i]
    # code_2d_list - the 2D pixels of the observed led code_idx_list[i]
    # code_3d_list - a list of 3D coordinates (one 3D for each led number)
    def build_opt_struct(self):
        frame_idx_list = []
        code_idx_list = []
        code_2d_list = []
        for counter,code in enumerate(self.pc.codes):
            for idx in range(len(self.led_graph[code].loc)):
                code_idx_list.append(counter)
                frame_idx_list.append(self.led_graph[code].frame[idx])
                code_2d_list.append(self.led_graph[code].loc[idx])
        code_3d_list = self.pc.pc

        camera_params = np.zeros([len(self.frame_graph), 6])
        for idx, frame in enumerate(self.frame_graph):
            camera_params[idx, :3] = cv.Rodrigues(self.frame_graph[frame][0])[0].reshape(3)
            camera_params[idx, 3:] = self.frame_graph[frame][1].reshape(3)

        return camera_params, np.asarray(code_3d_list).transpose(), np.asarray(frame_idx_list), np.asarray(code_idx_list), np.asarray(code_2d_list)


    # Takes 2 point list (from 2 image frames) an extracts 3d points and camera pose using opencv's RecoverPose function
    def two_frame_sfm(self, name1, name2, id1, id2):
        with open(self.data_path + name1, "rb") as fp:   # Unpickling
            point_list1 = pickle.load(fp)
        with open(self.data_path + name2, "rb") as fp:   # Unpickling
            point_list2 = pickle.load(fp)

        points1_0 = np.asarray(point_list1)
        points2_0 = np.asarray(point_list2)

        bad_indices1 = np.where(~points1_0.any(axis=1))[0]
        bad_indices2 = np.where(~points2_0.any(axis=1))[0]

        # Save for every led all its locations (pixels)
        if self.first_frame:
            self.first_frame = False
            for i in np.where(points1_0.any(axis=1))[0]:
                self.led_graph[self.led_code_list[i]].add((points1_0[i, 0],points1_0[i,1]), id1)
        for i in np.where(points2_0.any(axis=1))[0]:
            self.led_graph[self.led_code_list[i]].add((points2_0[i, 0], points2_0[i, 1]), id2)

        bad_indices = list(bad_indices1)
        bad_indices.extend(x for x in bad_indices2 if x not in bad_indices)
        good_indices = [i for i in range(points1_0.shape[0]) if i not in bad_indices]

        good_code_names = [self.led_code_list[i] for i in good_indices]
        print("good_code_names: ",good_code_names)
        points1 = points1_0[good_indices, :]
        points2 = points2_0[good_indices, :]
        E, mask = cv.findEssentialMat(points1, points2, self.mtx, method=cv.RANSAC, prob=0.999, threshold=1.0)
        # E = E[:3,:3]
        print(E)
        retval, R, t, mask, triangulatedPoints = cv.recoverPose(E, points1, points2, self.mtx, distanceThresh=50)
        camera_two = -R @ t

        triangulatedPoints = triangulatedPoints[0:3,:]/triangulatedPoints[3, :]
        frame_pc = PC(triangulatedPoints, good_code_names)
        frame_pose = RelativePose(R,t,id1,id2)

        if np.linalg.det(R)-1 > 1e-7:
            print(f'error in R matrix:{R}')

        return frame_pc, frame_pose


class Optimization:

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def project(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        # points_proj = self.rotate(points, camera_params[:, :3])
        # points_proj -= camera_params[:, 3:6]
        # points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        #
        # f = self.mtx[0,0]
        # print("f", f)
        # k1 = 0#self.distortion[0][0]
        # k2 = 0#self.distortion[0][1]
        # n = np.sum(points_proj ** 2, axis=1)
        # r = 1 + k1 * n + k2 * n ** 2
        # points_proj *= (r * f)[:, np.newaxis]

        points_proj = np.zeros((points.shape[0],2))
        for i in range(points.shape[0]):
            curr_proj_point = cv.projectPoints(points[i,:].reshape(1,3), camera_params[i, :3].reshape(3,1), camera_params[i, 3:].reshape(3,1), self.mtx, None)
            points_proj[i, :] = curr_proj_point[0]

        return points_proj

    def fun(self,params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * self.num_cam_params].reshape((n_cameras, self.num_cam_params))
        points_3d = params[n_cameras * self.num_cam_params:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices,:])

        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self,n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * self.num_cam_params + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(self.num_cam_params):
            print(camera_indices)
            A[2 * i, camera_indices * self.num_cam_params + s] = 1
            A[2 * i + 1, camera_indices * self.num_cam_params + s] = 1

        for s in range(3):
            print(point_indices.shape)
            A[2 * i, n_cameras * self.num_cam_params + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * self.num_cam_params + point_indices * 3 + s] = 1

        return A

    def __init__(self, camera_params, points_3d, camera_indices, point_indices, points_2d, mtx, dist):

        self.num_cam_params = 6
        self.mtx = mtx
        self.distortion = dist


        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]

        n = self.num_cam_params * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]

        print("n_cameras: {}".format(n_cameras))
        print("n_points: {}".format(n_points))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))

        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        f0 = self.fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
        plt.plot(f0)
        plt.show()
        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        # print(A.shape)
        # plt.spy(A)
        # plt.show()
        t0 = time.time()
        self.res = least_squares(self.fun, x0, verbose=2, x_scale='jac', ftol=1e-7, method='trf',
                            args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
        t1 = time.time()

        print("Optimization took {0:.0f} seconds".format(t1 - t0))
        plt.plot(self.res.fun)
        plt.show()


def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))*180/np.pi


sfm = SFM(data_path='sfm/')

frame_list = ['img1.pkl','img2.pkl','img3.pkl']

for i in range(len(frame_list)-1):
    pc, pose = sfm.two_frame_sfm(frame_list[i], frame_list[i+1], i, i+1)
    sfm.merge_point_cloud(new_pc=pc, new_pose=pose)

    # final_scale = sfm.dist / (np.linalg.norm(sfm.pc.get_3D(sfm.scale_led1) - sfm.pc.get_3D(sfm.scale_led2)))
    # sfm.pc.update_scale(final_scale)
    # print(np.linalg.norm(sfm.pc.get_3D(265) - sfm.pc.get_3D(1415)))
    # sfm.pc.plot_pc()
# print(f'sfm.led_graph: {sfm.led_graph[85].loc}')
# print(f'sfm.frame_graph: {sfm.frame_graph[1].t}')
# sfm.known_scale(335,1207,90.5)
# print(sfm.pc.pc.shape)
# final_scale = sfm.dist/(np.linalg.norm(sfm.pc.get_3D(sfm.scale_led1) - sfm.pc.get_3D(sfm.scale_led2)))
# print(final_scale)
# sfm.pc.update_scale(final_scale)
# print(np.linalg.norm(sfm.pc.get_3D(51) - sfm.pc.get_3D(741)))
# sfm.pc.plot_pc()
# print(f'camera_params: {camera_params}, \npoints_3d: {points_3d}, \ncamera_indices: {camera_indices}, \npoint_indices: {point_indices}, \nLED_CODES: {sfm.pc.codes}')
# print("camera_params: ",camera_params.shape)
# print("points_3d: ",points_3d.shape)
# print("camera_indices: ",camera_indices.shape)
# print("point_indices: ",point_indices.shape)
# print("points_2d: ",points_2d.shape)

# use known scale from measurements and update the point cloud
led1 = 7
led2 = 1207
scale = 130
sfm.known_scale(led1,led2,scale)
final_scale = sfm.dist/(np.linalg.norm(sfm.pc.get_3D(sfm.scale_led1) - sfm.pc.get_3D(sfm.scale_led2)))
sfm.pc.update_scale(final_scale)

camera_params, points_3d, camera_indices, point_indices, points_2d = sfm.build_opt_struct()
optimized_pc = PC(points_3d.transpose(),sfm.pc.codes)
optimized_pc.plot_pc()
#points_3d[10,0] += 5
# optimized_pc = PC(points_3d.transpose(),sfm.pc.codes)
# optimized_pc.plot_pc()

OPT = Optimization(camera_params, points_3d, camera_indices, point_indices, points_2d,sfm.mtx,sfm.distortion)
optimized_3d_points = OPT.res.x[camera_params.size:].reshape(points_3d.shape[0], 3)
optimized_pc = PC(optimized_3d_points.transpose(),sfm.pc.codes)
final_scale = scale/(np.linalg.norm(optimized_pc.get_3D(led1) - optimized_pc.get_3D(led2)))
optimized_pc.update_scale(final_scale)
optimized_pc.plot_pc()

#106.5
check_led1 = 51
check_led2 = 741
print(np.linalg.norm(sfm.pc.get_3D(check_led1) - sfm.pc.get_3D(check_led2)))
print(np.linalg.norm(optimized_pc.get_3D(check_led1) - optimized_pc.get_3D(check_led2)))
np.savez('sfm/presentation.npz', pc_list=optimized_pc.pc, pc_codes=optimized_pc.codes)

# f = open(f'sfm/PC.pkl', "wb")
# pickle.dump(optimized_pc.pc, f)
# f.close()
