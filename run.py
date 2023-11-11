import streamlit as st
import cv2
import cv2.aruco as aruco
import tempfile
import numpy as np

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

def detect_aruco_markers(frame):

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
                # Calculate the center of the marker
                center = np.mean(corners[i][0], axis=0).astype(int)
                position = [int(center[0]), int(center[1])]
                # Draw a circle at the center
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
    else:
        position = None
    return frame, position


f = st.file_uploader("Upload file")

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())


    vf = cv2.VideoCapture(tfile.name)
    fps = vf.get(cv2.CAP_PROP_FPS)
    st.write(f'Frames per second = {fps}')

    stframe = st.empty()
    position_list = list()
    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        gray, position = detect_aruco_markers(frame)
        
        if position is not None:
            position_list.append(position)
        else:
            position_list = list()

        print(position_list)
        if len(position_list) > 1:
            cv2.polylines(gray,[np.array(position_list)], isClosed=False, color=(255, 0, 0), thickness=2)
        stframe.image(gray)