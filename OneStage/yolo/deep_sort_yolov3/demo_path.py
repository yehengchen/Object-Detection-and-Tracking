#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
#list = [[] for _ in range(100)]

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.9 #0.9
    nn_budget = None
    nms_max_overlap = 0.3
    counter = []
   # deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_path = "../../yolo_dataset/t1_video/test_video/det_t1_video_00045_test.avi"
    video_capture = cv2.VideoCapture(video_path)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)

       # print("box_num",len(boxs))
        features = encoder(frame,boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        #print(tracker.predict())
        #tracks = tracker.update(detections)
        i = int(0)
        k = 0
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            #print(det)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
            #cv2.putText(frame, "person - "+str(track.track_id),(int(bbox[0]), int(bbox[1] -5)),0, 5e-3 * 150, (color),2)
            i = i+1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            #print(pts[track.track_id])
            #print(pts[track.track_id][0])
            #print(pts[track.track_id])
            #print("*****************************************")
            #print(track.track_id)
            thickness = 5
            cv2.circle(frame,  (center), 1, color, thickness)
            #cv2.line(frame,pts[i-1], pts[i],(color),thickness)
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
        
        # object counter
        count = len(set(counter))
        cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO3_Deep_SORT", 0);
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
        cv2.imshow('YOLO3_Deep_SORT', frame)


        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print(set(counter))
        #print("fps= %f"%(fps))
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("************* Finish *************")
    print(str(count) +' object found ')
    
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
