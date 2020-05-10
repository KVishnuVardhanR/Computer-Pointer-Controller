import cv2
from openvino.inference_engine import IENetwork, IECore
import logging as log

class facial:

    def __init__(self, model_name,eye_detect, device='CPU', threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.eye = eye_detect

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


    def predict(self, image):
        self.processed_image=self.preprocess_input(image)
        results= self.net.infer(inputs={self.input_name:self.processed_image})
        detection = results[self.output_name]
        list3=[]
        rxmin = int(detection[0][4] * image.shape[1])-20
        rymin = int(detection[0][5] * image.shape[0])-20
        rxmax = int(detection[0][6] * image.shape[1])+30
        rymax = int(detection[0][7] * image.shape[0])+40
        list3.append((rxmin, rymin, rxmax, rymax))
        try:
            rcrpimg=image[list3[0][1]:list3[0][3],list3[0][0]:list3[0][2]]
            if self.eye == "eye_detect":
                cv2.rectangle(image, (rxmin,rymin), (rxmax,rymax), (0,0,255),1)
        except Exception as e:
            log.error('Could not detect right eye in the frame, Index is out of range')
            exit()
        try:
            left_eye = image[int(image.shape[0] * detection[0][1]) - 45:int(image.shape[0] * detection[0][3]) + 45, int(image.shape[1] * detection[0][0]) - 45:int(image.shape[1] * detection[0][2]) + 45]
            if self.eye == "eye_detect":
                cv2.rectangle(image, (int(image.shape[1] * detection[0][0]) - 45, int(image.shape[0] * detection[0][1]) - 45), (int(image.shape[1] * detection[0][2]) + 45, int(image.shape[0] * detection[0][3]) + 45), (0,0,255),1)
        except Exception as e:
            log.error('Could not detect left eye in the frame, Index is out of range')
            exit()                    
            
        return left_eye, rcrpimg
        

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        self.image = self.image.transpose((2,0,1))
        self.image = self.image.reshape(1, *self.image.shape)
        return self.image

