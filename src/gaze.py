from openvino.inference_engine import IENetwork, IECore
import cv2

class gaze:

    def __init__(self, model_name, gaze_v, device='CPU', threshold=0.6):

        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.gaze=gaze_v

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

        
    def predict(self, limage,rimage,angles):
        self.processed_rimage=self.preprocess_input(rimage)
        self.processed_limage=self.preprocess_input(limage)
        results= self.net.infer(inputs={"head_pose_angles": angles,"left_eye_image":self.processed_limage,
                "right_eye_image":self.processed_rimage})
        x,y = results['gaze_vector'][0][0],results['gaze_vector'][0][1]
        if self.gaze == "gaze_vector":
            print(results)
        return x,y

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        self.image = cv2.resize(image, (60,60))
        self.image = self.image.transpose((2,0,1))
        self.image = self.image.reshape(1, *self.image.shape)
        return self.image

