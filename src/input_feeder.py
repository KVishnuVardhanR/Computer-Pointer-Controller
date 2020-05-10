import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)
        
        # if stream is not open exit program 
        # opencv print error message
        if not self.cap.isOpened():
            exit(0)
            
        
    def get_dimensions(self):
        return self.cap.get(3), self.cap.get(4)

    def next_batch(self):
        # if isinstance(self.cap, ndarray):
        #     while True:
        #         yield self.cap
        # else:
        while True:
            self.cap.set(3, 700)
            self.cap.set(4, 700)
            ret, frame=self.cap.read()

            if not ret:
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #self.cap.release()
            yield frame

    def close(self):
        if not self.input_type=='image':
            self.cap.release()
'''

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)

    def next_batch(self):
        while True:
            for _ in range(10):
                self.cap.set(3, 700)
                self.cap.set(4, 700)
                _, frame=self.cap.read()
            yield frame


    def close(self):
        if not self.input_type=='image':
            self.cap.release()
'''
        
