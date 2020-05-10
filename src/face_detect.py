from openvino.inference_engine import IENetwork, IECore
import cv2
import logging as log

class face:

    def __init__(self, model_name, face_detect, device='CPU', threshold=0.60):

        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.face=face_detect

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
        try:
            self.processed_image=self.preprocess_input(image)
            results= self.net.infer(inputs={self.input_name:self.processed_image})
            detections = results[self.output_name]
            self.coords, self.image = self.draw_outputs(detections, image)
            self.crpimg = image[self.coords[0][1]:self.coords[0][3], self.coords[0][0]:self.coords[0][2]]
            return image, self.crpimg,detections
        except Exception as e:
            log.error('Face detection failed, Could not detect any face in the frame')
            exit()
            
    def draw_outputs(self, detections, image):
        coords=[]
        for box in detections[0][0]: 
            conf = box[2]
            if conf >= self.threshold :
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                coords.append((xmin, ymin, xmax, ymax))
                if self.face == "face_detect":
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)       
        return coords ,image

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        self.image = self.image.transpose((2,0,1))
        self.image = self.image.reshape(1, *self.image.shape)
        return self.image

