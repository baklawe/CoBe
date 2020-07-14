import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist
import scipy.io as spio
import subprocess
from collections import defaultdict
import time
from scipy.optimize import linear_sum_assignment
import socket
import pickle

# communication to PC over UDP
send_udp = False
UDP_IP = "192.168.43.203"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

# Camera calibration params
with np.load('calib_12.11.19.npz') as X: mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist,None,None,size = (640,480),m1type = cv.CV_32FC1)  # create a distortion map
# LED 3D map - need to add num bits to the file
with np.load('PC_27.11.npz') as X: pc_list, pc_codes = [X[i] for i in ('pc_list', 'pc_codes')]
led_world_cord_dict = {7: 0, 51: 1, 85: 2, 127: 3, 265: 4, 335: 5, 467: 6, 741: 7, 959: 8, 1207: 9, 1753: 10, 2007: 11, 2211:12, 2293:13, 2733:14, 2874:15}
# led_world_cord_dict = {}
# for i in range(len(pc_codes)):
#     led_world_cord_dict[pc_codes[i]] = tuple(pc_list[:,i])
# Codebook
num_bits = 15
code_dict = spio.loadmat(f'CodeCyclic/CodeCyclic{num_bits}.mat', squeeze_me=True)
code_list = code_dict['dict']
codes = code_dict['codes']


# Camera parameters
gain = 35
exposure = 150
fps = 30
cam_props = {'horizontal_flip': 0, 'gain_automatic': 0, 'auto_exposure': 1, 'gain': gain, 'exposure': exposure}
for key in cam_props:
    subprocess.call(['v4l2-ctl -d /dev/video2 -c {}={}'.format(key, str(cam_props[key]))], shell=True)

# cap = cv.VideoCapture('videos/our_normal.avi')  # Capture from file
cap = cv.VideoCapture(2) # Capture live
cap.set(cv.CAP_PROP_FPS, 60)  # set max rate for PSeye camera


# Additional parameters
min_radius = 0  # minimal radius size of circle in pixels
max_radius = 50  # maximal radius size of circle in pixels
max_count = 100  # maximal count for a code name for a LED
MATCH_TH = 4000  # threshold distance for the optimal matching
low_thresh = 150  # Lower threshold for the thresholding function

# Initialization for variables in the loop
pre_contour_list = np.asarray([])
contour_dict = defaultdict(list)  # keeps history only of LEDs area that we currently see
area_dict = defaultdict(list)  # keeps history of all LEDs area for all frames until no LED is seen, then we clear
location_dict = defaultdict(list)  # corresponds to "contour_dict" with pixel location
name_decoder_dict = {}
PNPTH = 10
prev_tvec = None
start = 0
frame_counter = 0
first_name = 1
print_counter = 1

# # Initialization for plotting the pose using Plot_pose
# initial_tmat = np.eye(4,4)
# tracker_plot = Plot_Pose(initial_tmat, size=10, show_animation=True)


#  Find matching between current and previous frame
def find_optimal_match(prev_points, curr_points):
    m = curr_points.shape[0]
    n = prev_points.shape[0]
    dist_mat = cdist(curr_points, prev_points)
    diff = np.abs(m - n)
    if m > n:
        padding = ((0, 0), (0, diff))
        dist_mat = np.pad(dist_mat, padding, mode='constant', constant_values=0)
        smaller = n
    else:
        padding = ((0, diff), (0, 0))
        dist_mat = np.pad(dist_mat, padding, mode='constant', constant_values=0)
        smaller = m
    row_ind, col_ind = linear_sum_assignment(dist_mat)

    avg_cost = dist_mat[row_ind[:smaller], col_ind[:smaller]].sum() / smaller

    # Problem: an old point might disappear and a new point might take its place -> we think all points are old points.
    # Solution: If the average cost is big, then we probably have a new point which ruins our matching.
    # -> add padding to allow matching 1 of the curr_points with cost 0 to some "joker".
    i = 1
    while avg_cost > MATCH_TH:
        padding = ((0, 1), (0, 1))
        dist_mat = np.pad(dist_mat, padding, mode='constant', constant_values=0)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        avg_cost = dist_mat[row_ind[:smaller], col_ind[:smaller]].sum() / (smaller-i)
        i += 1

    # Find indices of current (new) points which have been matched to the newly added padding.
    #unmatched = col_ind[col_ind >= np.max([m,n])]
    #unmatched_idx = [np.where(col_ind == i)[0][0] for i in unmatched]

    return col_ind  # current[i] is matched to to pre[indexes[i]]


#  Convert radius data to binary data for code construction
def convert_to_binary_avg(vector):
    max_val = max(vector)
    min_val = min(vector)
    avg_val = (max_val+min_val) / 2
    binary_list = [1 if i > avg_val else 0 for i in vector]
    binary_str = ''.join(str(e) for e in binary_list)
    return binary_str


while True:
    # Capture frame-by-frame
    frame_counter = frame_counter + 1
    while (time.time()-start) < 1/fps:
        pass
    end = time.time()
    seconds = end - start
    # print(seconds)
    start = time.time()
    ret, im = cap.read()  # get current frame
    im = cv.remap(im, mapx, mapy,cv.INTER_LINEAR)  # Undistort the image
    im2 = np.copy(im)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  # Convert to gray scale
    im_gauss = cv.GaussianBlur(imgray, (5, 5), 0)  # Blur the image for better stability
    _, thresh = cv.threshold(im_gauss, low_thresh, 255, 0)  # create a threshold map to accelerate findContours

    # get contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_list = []

    for contour in contours:
        ((x, y), radius) = cv.minEnclosingCircle(contour)
        area = np.pi * (radius**2)
        center = (int(x), int(y))
        if min_radius < radius < max_radius:
            contour_list.append((x, y, area, None))
            cv.circle(im2, center, int(radius), (0, 0, 255), -1)
            im3 = np.copy(im2)

    contour_list = np.asarray(contour_list)
    curr_num_cont = contour_list.shape[0]
    prev_num_cont = pre_contour_list.shape[0]

    if prev_num_cont is not 0:
        if curr_num_cont is not 0:
            matches = find_optimal_match(pre_contour_list[:, 0:2], contour_list[:, 0:2])

            if curr_num_cont <= prev_num_cont:
                for i in range(curr_num_cont):
                    # if i not in unmatched_idx:
                    if matches[i] < prev_num_cont:
                        contour_list[i, 3] = pre_contour_list[matches[i], 3]
            else:
                for i in range(prev_num_cont):
                    index = np.where(matches == i)[0][0]
                    if index < curr_num_cont:
                        contour_list[index, 3] = pre_contour_list[i, 3]

            for i in range(curr_num_cont):
                if contour_list[i, 3] is None:
                    contour_list[i, 3] = first_name
                    first_name = first_name + 1
                # cv.putText(im, str(contour_list[i, 3]), (int(contour_list[i, 0]+10), int(contour_list[i, 1]+10)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv.LINE_AA)
    else:
        first_name = 1
        area_dict.clear()
        name_decoder_dict = {}

    world_pts = []
    world_pts2 = []
    img_pts = []
    img_pts2 = []
    larger_num_bits = []
    contour_dict.clear()
    location_dict.clear()
    code_current_list = []
    potential_bad = []
    sfm_list = np.zeros((len(led_world_cord_dict), 2))

    for src in contour_list:
        if src[3] in area_dict:
            contour_dict[src[3]].extend(area_dict[src[3]])
        area_dict[src[3]].append(src[2])
        contour_dict[src[3]].append(src[2])
        location_dict[src[3]].append([src[0], src[1]])
        if (len(contour_dict[src[3]]) > num_bits - 1) and src[3] is not None:
            larger_num_bits.append(src[3])

        if src[3] in larger_num_bits:
            bin_vec = convert_to_binary_avg(contour_dict[src[3]][-(num_bits):])
            led_code = code_list[int(bin_vec, 2)] - 1

            # name_decoder_dist: [fake name] -> [code name, # times we have seen this code name consecutively]
            # if diff code name seen with this fake name -> counter --. If counter is 0 change code name.
            if src[3] not in name_decoder_dict:
                name_decoder_dict[src[3]] = [led_code, 1]
            else:
                if name_decoder_dict[src[3]][0] == led_code and name_decoder_dict[src[3]][0] != -1:
                    name_decoder_dict[src[3]][1] = name_decoder_dict[src[3]][1] + 1
                    if name_decoder_dict[src[3]][1] >= max_count:
                        name_decoder_dict[src[3]][1] = max_count

                else:
                    name_decoder_dict[src[3]][1] = name_decoder_dict[src[3]][1] - 1
                    if name_decoder_dict[src[3]][1] <= 0:
                        name_decoder_dict[src[3]] = [led_code, 1]
            led_code = name_decoder_dict[src[3]][0]

            # Check if there is any other led with the same led_code in the same frame. If there is, remove the one with the
            # smallest counter. If none are found, we just append it.
            # code_current_list contains the fake names of all the leds in the current frame, each occuring only once!
            flag1 = False
            for idx in code_current_list:
                if name_decoder_dict[idx][0] == led_code and idx != src[3]:
                    flag1 = True
                    if name_decoder_dict[idx][1] > name_decoder_dict[src[3]][1]:
                        del name_decoder_dict[src[3]]
                        if src[3] in code_current_list:
                            code_current_list.remove(src[3])
                    else:
                        del name_decoder_dict[idx]
                        code_current_list.remove(idx)
                        code_current_list.append(src[3])
            if not flag1:
                code_current_list.append(src[3])

            if led_code in led_world_cord_dict:
                sfm_list[led_world_cord_dict[led_code]][0] = location_dict[src[3]][0][0]
                sfm_list[led_world_cord_dict[led_code]][1] = location_dict[src[3]][0][1]


        elif src[3] in name_decoder_dict:
            code_current_list.remove(src[3])
            del name_decoder_dict[src[3]]

    for src in code_current_list:
        cv.putText(im, str(name_decoder_dict[src][0]), (int(location_dict[src][0][0]), int(location_dict[src][0][1])),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv.LINE_AA)
        if name_decoder_dict[src][0] in led_world_cord_dict:
            world_pts.append(led_world_cord_dict[name_decoder_dict[src][0]])
            img_pts.append((location_dict[src][0][0], location_dict[src][0][1]))
            if name_decoder_dict[src][1] > 1:

                world_pts2.append(led_world_cord_dict[name_decoder_dict[src][0]])
                img_pts2.append((location_dict[src][0][0], location_dict[src][0][1]))
            else:
                potential_bad.append(src)


    if cv.waitKey(10) & 0xFF == ord('p'):
        print("saving data")

        spio.savemat(f'sfm/img{print_counter}.mat', mdict={f'img{print_counter}': sfm_list})
        f = open(f'sfm/img{print_counter}.pkl', "wb")
        pickle.dump(sfm_list, f)
        f.close()
        cv.imwrite(f'sfm/img{print_counter}.png', im)

        # value = [255, 255, 255]
        # border = 5
        # im = cv.copyMakeBorder(im, border, border, border, border, cv.BORDER_CONSTANT, None, value)
        # im2 = cv.copyMakeBorder(im2, border, border, border, border, cv.BORDER_CONSTANT, None, value)
        # im3 = cv.copyMakeBorder(im3, border, border, border, border, cv.BORDER_CONSTANT, None, value)
        # numpy_horizontal = np.hstack((im, im2,im3))

        # cv.imwrite(f'photos/horizontal.png', numpy_horizontal)
        # cv.imwrite(f'photos/imgclean{print_counter}.png', im)
        # cv.imwrite(f'photos/imgcircles{print_counter}.png', im2)
        # cv.imwrite(f'photos/imgcodes{print_counter}.png', im3)
        print_counter = print_counter + 1
        # plt.pause(0.0001)

    # print("end of frame")
    pre_contour_list = contour_list
    tmp = pre_contour_list

    # Display the resulting frame
    # if ret:
    cv.imshow('frame', im)
    #if cv.waitKey(15) & 0xFF == ord('q'):
     #   break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
