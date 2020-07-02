from face_detect import face
from facial_landmarks import facial
from gaze import gaze
from head_pose import head_pose
from input_feeder import InputFeeder
from mouse_controller import MouseController
import time
import os
import cv2
import argparse
import sys
import logging as log


def main(args):
    fmodel = args.FM
    flmodel = args.FLM
    hmodel = args.HM
    gmodel = args.GM
    device = args.D
    video_file = args.V
    face_detect = args.FD
    eye_detect = args.ED
    gaze_v = args.GD
    head = args.HD 
    threshold = 0.6

    start_model_load_time = time.time()

    #initailizing models
    
    face_model = face(fmodel, face_detect, device, threshold)
    facial_landmarks = facial(flmodel,eye_detect, device, threshold)
    head_pose_est = head_pose(hmodel,head, device, threshold)
    gaze_est = gaze(gmodel, gaze_v, device, threshold)

    # Loading models

    face_model.load_model()
    facial_landmarks.load_model()
    head_pose_est.load_model()
    gaze_est.load_model()
    total_model_load_time = time.time() - start_model_load_time

    if video_file != "cam":
        vtype = 'video'
    else:
        vtype = 'cam'

    counter = 0
    start_inference_time = time.time()
        
    try:
        feed=InputFeeder(input_type = vtype, input_file = video_file)
        feed.load_data()
        for batch in feed.next_batch():
            if batch is None:
                log.error('The input frame is not being read, The file is corrupted')
                exit()
            counter += 1
            frame,face_crop,detections = face_model.predict(batch)
            limg,rimg = facial_landmarks.predict(face_crop)
            angles,frame = head_pose_est.predict(face_crop,detections,frame)
            x,y = gaze_est.predict(limg,rimg,angles)
            
            if eye_detect == "eye_detect":
                cv2.imshow('frame', face_crop)
            elif head == "head_pose":
                cv2.imshow('frame',frame) 
            else:
                cv2.imshow('frame',frame)
                        
            if vtype != 'video':
                t=1
            else:
                t=500
                
            if cv2.waitKey(t) & 0xFF == ord('q'):
                break
            mouse = MouseController('low','fast')
            mouse.move(x,y)
                
        feed.close()
        cv2.destroyAllWindows()

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time

        print("The total time to load all the models is :"+str(total_model_load_time)+"sec")
        print("The total inference time of the models is :"+str(total_inference_time)+"sec")
        print("The total number of frames per second is :"+str(fps)+"fps")
        
    except Exception as e:
        print("Could not run Inference: ", e)



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-V', default=None,
                        help = "Input file, User can enter path of video file or enter cam for webcam.")
    
    parser.add_argument('-FM', required=True,
                        help = "Path to xml file of Face detection model.")
    
    parser.add_argument('-FLM', required=True,
                        help = "Path to xml file of Facial Landmarks model.")
    
    parser.add_argument('-HM', required=True,
                        help = "Path to xml file of Head pose estimation model.")
    
    parser.add_argument('-GM', required=True,
                        help = "Path to xml file of Gaze Estimation model.")
    
    parser.add_argument('-D', default='CPU',
                        help = "specifying device like CPU,GPU,VPU,FPGA to run inference.")
    
    parser.add_argument('-FD',default=None,
                        help = "Visualizing face detection model output, Enter face_detect .")
    
    parser.add_argument('-ED',default=None,
                        help = "Visualizing left and right eye from facial landmark detection,Enter eye_detect.")
    
    parser.add_argument('-HD',default=None,
                        help = "Visualizing head pose angles from head pose estimation,Enter head_pose.")

    parser.add_argument('-GD',default=None,
                        help = "Visualizing gaze vector,Enter gaze_vector.")
    
    args=parser.parse_args()
    main(args)
