from openvino.inference_engine import IENetwork, IECore
import cv2
import math
import numpy as np


class head_pose:

    def __init__(self, model_name, head, device='CPU', threshold=0.6):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.head=head
        self.threshold=threshold

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device,num_requests=1)

    def predict(self, image,fd_result,frame):
        self.processed_image=self.preprocess_input(image)
        results = self.net.infer(inputs={self.input_name:self.processed_image})
        p,r,y = results['angle_p_fc'],results['angle_r_fc'],results['angle_y_fc']
        detections=[]
        detections.append((p,r,y))
        if self.head == "head_pose":
            focal_length = 950.0
            scale = 50
            frame_h, frame_w = frame.shape[:2]
            faces = fd_result[0][:, np.where(fd_result[0][0][:, 2] > 0.1)]
            for face in faces[0][0]:
                box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            (xmin, ymin, xmax, ymax) = box.astype("int")
            face_frame = frame[ymin:ymax, xmin:xmax]
            center_of_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0)
            frame = self.draw_axes(frame, center_of_face, y, p, r, scale, focal_length)
        return detections,frame
    
    def draw_axes(self,frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                       [0, 1, 0],
                       [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0],
                       [0, 0, 1]])
        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame

    def build_camera_matrix(self, center_of_face, focal_length):
        self.cx = int(center_of_face[0])
        self.cy = int(center_of_face[1])
        self.camera_matrix = np.zeros((3, 3), dtype='float32')
        self.camera_matrix[0][0] = focal_length
        self.camera_matrix[0][2] = self.cx
        self.camera_matrix[1][1] = focal_length
        self.camera_matrix[1][2] = self.cy
        self.camera_matrix[2][2] = 1
        return self.camera_matrix        

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        self.image = self.image.transpose((2,0,1))
        self.image = self.image.reshape(1, *self.image.shape)
        return self.image

