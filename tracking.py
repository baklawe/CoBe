import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist
import scipy.io as spio
from scipy.optimize import linear_sum_assignment
import subprocess
from collections import defaultdict
import time
import socket
import random
from plot_pose import Plot_Pose

# communication to PC over UDP
send_udp = False
UDP_IP = "192.168.43.203"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

def colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r,g,b))
  return ret

calibration_file = 'calib_12.11.19.npz'
map_file = 'robohead_test.npz'
load_video_file = '/home/roman/CIRLT/videos/1703/cam1_yaw1.avi'
pose_data_file = '/home/roman/CIRLT/videos/cam1_head.txt'
save_video_file = '/home/roman/CIRLT/videos/1703/cam1_yaw1_letters.avi'
live_camport = 2
live = 0
plot_tracking = 0
write_file = 0
vid_record = 1
map_file = 'PC_18.02.npz'
load_video_file = '/home/roman/CIRLT/videos/final/ours_good.avi'
pose_data_file = '/home/roman/CIRLT/videos/final/ours_good_pose.txt'
save_video_file = f'videos/final/ours_good_markers.avi'
live_camport = 0
live = 1
plot_tracking = 0
write_file = 0
vid_record = 0

# Camera calibration params
with np.load(calibration_file) as X: mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist,None,None,size = (640,480),m1type = cv.CV_32FC1)  # create a distortion map
# LED 3D map - need to add num bits to the file
with np.load(map_file) as X: pc_list, pc_codes = [X[i] for i in ('pc_list', 'pc_codes')]
led_world_cord_dict = {}
for i in range(len(pc_codes)):
    led_world_cord_dict[pc_codes[i]] = tuple(pc_list[:,i])
# Codebook
num_bits = 15
code_dict = spio.loadmat(f'CodeCyclic/CodeCyclic{num_bits}.mat', squeeze_me=True)
code_list = code_dict['dict']
codes = code_dict['codes']

import string
letters_list = string.ascii_uppercase
codes_colors = {}
# uniqe_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,255,0),(255,0,255)]
# uniqe_colors = colors(len(pc_codes))
for num,code in enumerate(pc_codes):
    # codes_colors[code] = uniqe_colors[num]
    codes_colors[code] = letters_list[num]

# Camera parameters

gain = 35
exposure = 100
gain = 15
exposure = 50
fps = 30
if live:
    cam_props = {'horizontal_flip': 0, 'gain_automatic': 0, 'auto_exposure': 1, 'gain': gain, 'exposure': exposure}
    for key in cam_props:
        subprocess.call(['v4l2-ctl -d /dev/video{} -c {}={}'.format(live_camport, key, str(cam_props[key]))], shell=True)
    cap = cv.VideoCapture(live_camport) # Capture live
else:
    cap = cv.VideoCapture(load_video_file)  # Capture from file
#cap.set(cv.CAP_PROP_FRAME_WIDTH,320);
#cap.set(cv.CAP_PROP_FRAME_HEIGHT,240);
cap.set(cv.CAP_PROP_FPS, 50)

# Additional parameters
if write_file:
    pose_data = open(pose_data_file, "w")
min_radius = 0  # minimal radius size of circle in pixels
max_radius = 15  # maximal radius size of circle in pixels
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
t_mat = None
start = 0
frame_counter = 0
first_name = 1

# # Initialization for plotting the pose using Plot_pose
if plot_tracking:
    initial_tmat = np.eye(4,4)
    tracker_plot = Plot_Pose(initial_tmat, size=10, show_animation=True)

## Video capture
if vid_record:
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv.VideoWriter(save_video_file, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))


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
    while (time.time()-start) < 1/fps and live == 1:
        pass
    end = time.time()
    seconds = end - start
    print(seconds)
    start = time.time()
    ret, im = cap.read()  # get current frame
#    im = cv.remap(im, mapx, mapy,cv.INTER_LINEAR)  # Undistort the image
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  # Convert to gray scale
    im_gauss = cv.GaussianBlur(imgray, (5, 5), 0)  # Blur the image for better stability
    _, thresh = cv.threshold(im_gauss, low_thresh, 255, 0)  # create a threshold map to accelerate findContours

    # get contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_list = []

    for contour in contours:
        ((x, y), radius) = cv.minEnclosingCircle(contour)
        area = np.pi * (radius**2)
        center = (int(x), int(y))
        if min_radius < radius < max_radius:
            contour_list.append((x, y, area, None))
            cv.circle(im, center, int(radius), (0, 0, 255), 2)

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
        elif src[3] in name_decoder_dict:
            code_current_list.remove(src[3])
            del name_decoder_dict[src[3]]

    for src in code_current_list:
        if name_decoder_dict[src][0] in codes_colors:
            # cv.circle(im, (int(location_dict[src][0][0]), int(location_dict[src][0][1])),
            #           int(np.sqrt(contour_dict[src][-1])), (0,0,255), -1)
            cv.putText(im, codes_colors[name_decoder_dict[src][0]], (int(location_dict[src][0][0]), int(location_dict[src][0][1])),
            # cv.putText(im, str(name_decoder_dict[src][0]), (int(location_dict[src][0][0]), int(location_dict[src][0][1])),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv.LINE_AA)
        if name_decoder_dict[src][0] in led_world_cord_dict:
            world_pts.append(led_world_cord_dict[name_decoder_dict[src][0]])
            img_pts.append((location_dict[src][0][0], location_dict[src][0][1]))
            if name_decoder_dict[src][1] > 1:

                world_pts2.append(led_world_cord_dict[name_decoder_dict[src][0]])
                img_pts2.append((location_dict[src][0][0], location_dict[src][0][1]))
            else:
                potential_bad.append(src)

    if len(world_pts) >= 4:

        ret, rvecs, tvecs, inliners = cv.solvePnPRansac(np.asarray(world_pts, dtype=np.float32).reshape(len(world_pts), 1, 3),
                                        np.asarray(img_pts, dtype=np.float32).reshape(len(world_pts),1,2),
                                        mtx, None, reprojectionError = 8.0,flags=cv.SOLVEPNP_EPNP  ) #see if Undistord is needed or not

        # ret, rvecs, tvecs = cv.solvePnP(np.asarray(world_pts, dtype=np.float32).reshape(len(world_pts), 1, 3),
        #                                 np.asarray(img_pts, dtype=np.float32).reshape(len(world_pts),1,2),
        #                                 mtx, None,flags=cv.SOLVEPNP_EPNP) #see if Undistord is needed or not
        r_mat, _ = cv.Rodrigues(rvecs)
        tvecs = -r_mat.T @ tvecs
        # tvecs = np.zeros_like(tvecs)

        if prev_tvec is not None:
            pnp_dist = np.linalg.norm(prev_tvec - tvecs)

        t_mat = np.concatenate((r_mat, tvecs), axis=1)
        t_mat = np.concatenate((t_mat, np.array([[0, 0, 0, 1]])), axis=0)
        if write_file:
            np.savetxt(pose_data,t_mat.flatten().reshape(1,-1))

#        print("tmat",t_mat)
        if send_udp is True:
            MESSAGE = t_mat.tostring()
            sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
        if plot_tracking:
            tracker_plot.update_pose(t_mat,True)
        # if prev_tvec is not None and pnp_dist < PNPTH:
        prev_tvec = tvecs
        prev_rvec = rvecs
        prev_r_mat = r_mat
    else:
        if write_file:
            np.savetxt(pose_data, np.zeros((1,16)))
    # print("end of frame")
    pre_contour_list = contour_list
    tmp = pre_contour_list

    # Display the resulting frame
    # if ret:
    if vid_record:
        out.write(im)
    cv.imshow('frame', im)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
