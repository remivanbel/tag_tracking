import streamlit as st
import cv2
import cv2.aruco as aruco
import tempfile
import numpy as np
import math
from bokeh.plotting import figure, show

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

def get_speed(array, dt):
    # Step 1: Calculate Displacement
    displacement = np.diff(array, axis=0)

    # Step 2: Calculate Euclidean Distance
    distance = np.linalg.norm(displacement, axis=1)

    # Step 4: Calculate Speed
    speed = distance / dt

    # Note: The first time interval might be different; you can handle it accordingly.

    # Print the resulting speed array
    return speed

def get_distance(point1, point2):
    return math.dist(point1, point2)

class Tag():
    def __init__(self, corners, id):
        self.corners = corners[0]
        self.id = id

    def get_position(self):
        return np.mean(self.corners, axis=0).astype(int)

    def get_resolution(self):
        # Calculate the distances between consecutive corners
        known_marker_size = self.id*100

        side_lengths = [np.linalg.norm(self.corners[i] - self.corners[(i + 1) % 4]) for i in range(4)]

        # Calculate the average distance
        average_pixel_distance = np.mean(side_lengths)

        # Calculate the resolution in real-world units
        resolution = known_marker_size / average_pixel_distance

        return resolution

def detect_aruco_markers(frame):

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    print(rejected_img_points)
    tag_list = list()
    if ids is not None:
        for i in range(len(ids)):
            tag = Tag(corners[i], ids[i])
            tag_list.append(tag)
            cv2.circle(frame, tuple(tag.get_position()), 5, (0, 0, 255), -1)
        return frame, tag_list
    else:
        return frame, None

f = st.file_uploader("Upload file")

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())


    vf = cv2.VideoCapture(tfile.name)
    fps = vf.get(cv2.CAP_PROP_FPS)
    dt = 1/fps
    st.write(f'Frames per second = {fps}')

    stframe = st.empty()
    position_list = list()
    resolution_list = list()
    position_time = list()

    speed_lol = list()
    time_lol = list()
    current_time = 0

    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        else:
            current_time += dt

        gray, tag_list = detect_aruco_markers(frame)

        if tag_list is not None:
            position_list.append(tag_list[0].get_position())
            resolution_list.append(tag_list[0].get_resolution())
            position_time.append(current_time)

        elif position_list:
            p = figure(width=400, height=400)
            print(position_list)
            position_matrix = np.vstack(position_list)

            speed = get_speed(position_matrix, dt)
            speed = speed*np.median(np.asarray(resolution_list))
            speed_lol.append(list(speed))
            time_lol.append(position_time[1::])

            position_time = list()
            position_list = list()

        if len(position_list) > 1:
            cv2.polylines(gray,[np.array(position_list)], isClosed=False, color=(255, 0, 0), thickness=2)
        stframe.image(gray)

    p = figure(
        title='simple line example',
        x_axis_label='Time [s]',
        y_axis_label='Speed [mm/s]')

    # add a circle renderer with a size, color, and alpha
    for time_list, speed_list in zip(time_lol, speed_lol):
        p.line(time_list, speed_list)

    st.bokeh_chart(p)

# TODO: test in production, how many images processed per time