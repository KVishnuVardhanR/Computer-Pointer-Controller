# Computer Pointer Controller

A Computer Pointer Controller is a **Human-Computer Interaction Project**, where the user can control the mouse movements with his\her **eye-gaze** which will be captured through a webcam or even a video, with the help of **OpenVINO Toolkit** along with its Pre-Trained Models which helps deploying AI at Edge.


## Project Set Up and Installation

- Setup your local environment:
  - Download and install the **OpenVINO Toolkit**. The installations directions for OpenVINO can be found [here](https://docs.openvinotoolkit.org/latest/index.html)
  - Run the **Verification Scripts to verify your installation**. This is a very important step to be done before you proceed further.  
- The project directory is structured as follows:
```
					project
					|  
					|_ bin
					|  |_demo.mp4
					|      
					|_ README.md    
					|   
					|_ requirements.txt   
					|    
					|_src
					   |_ main.py
					   |_ input_feeder.py
					   |_ mouse_controller.py
					   |_ face_detect.py
					   |_ head_pose.py
					   |_ facial_landmarks.py
					   |_ gaze.py
	
```
  - The project directory contains a ```bin``` folder which has an .mp4 file, can be used as the input file for the project.
  - It has requirements.txt file which contains all the necessary dependencies to be installed before running the project.
  - The ```src``` folder in project directory contains the following python files:
    - The input_feeder.py is used to take the input file such as a video file or a webcam and yeilds the frames for running inference.
	- The mouse_controller.py takes the x,y co-ordinates from the gaze.py to move the mouse.
	- The face_detect,head_pose,facial_landmarks,gaze .py files contains each class functions to preprocess the inputs and run inference on those inputs and sent it to mouse_controller to move the mouse position.
	
- The Pre-Trained models you will need to download from OpenVINO Open model zoo using the ```model downloader``` are:
  - Face Detection model
  - Head Pose Estimation model
  - Facial Landmark Detection model
  - Gaze Estimation model
- Create a folder named **models** in the project directory, These models are to be downloaded and stored in models folder.
- The Ensembling of models to be done for controlling mouse pointer can be seen in the pipeline.png above.

**Note: This project has been tested only in Windows 10 Operating System environment with Intel core i3-7100 processor which has an Intel Integrated GPU HD Graphics 630.**  

## Demo

- First, initialize the OpenVINO environment:
  - Open command prompt and ```cd C:\Program Files (x86)\IntelSWTools\openvino\bin```
  - type ```setupvars.bat``` command and press *Enter* to initialize OpenVINO environment.
- Next, ```cd``` to src folder in the project directory:
- Now, run the following command to run our application:
  - ```python main.py -V "C:\<path_to_project_directory>\bin\demo.mp4" -FM "C:\<path_to_project_directory>\models\face-detection-retail-0004\FP32\face-detection-retail-0004" -D CPU -FLM "C:\<path_to_project_directory>\models\facial-landmarks-35-adas-0002\FP32\facial-landmarks-35-adas-0002" -HM "C:\<path_to_project_directory>\models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001" -GM "C:\<path_to_project_directory>\models\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002"```

**Note: Enter the path of the project directory inplace of <path_to_project_directory>, while executing the above command.**
 
## Documentation
- The ```python main.py -h``` command displays the commands which are supported by project:
  - ```-V``` argument takes the input video file or a webcam, for accessing video file the command is ```-V "<path of video file>"``` whereas for accessing webcam ```-V "cam"```.
  - ```-D``` argument specifies the devices such as **CPU,GPU,VPU,FPGA** to run inference on.
  - ```-FM``` argument takes in the face detection model, ```-FLM``` argument takes in the facial_landmarks model,```-HM``` argument takes in the head_pose estimation and ```-GM``` argument takes in the gaze estimation model.
  - ```-FD``` argument takes in to visualize outputs of face detection model, The flag is set by passing the argument as ```-FD "face_detect"``` to visualize the face detection model.
  - ```-ED``` argument takes in to visualize outputs of facial_landmarks detection model, The flag is set by passing the argument as ```-ED "eye_detect"``` to visualize left and right eye from the model.
  - ```-HD``` argument takes in to visualize outputs of head_pose estimation model, The flag is set by passing the argument as ```-HD "head_pose"``` to visualize the head positions from the model.
  - ```-GD``` argument takes in to print outputs of gaze estimation model, The flag is set by passing the argument as ```-GD "gaze_vector"``` to print the gaze vector from the model.

## Benchmarks

The benchmark result of running my model on **CPU** with multiple model precisions are :
- FP32:
  - The total model loading time is : 3.361sec
  - The total inference time is : 11.4sec
  - The total FPS is : 0.35fps
  
- FP16:
  - The total model loading time is : 1.77sec
  - The total inference time is : 8.7sec
  - The total FPS is : 0.45fps
  
- INT8:
  - The total model loading time is : 6.03sec
  - The total inference time is : 8.7sec
  - The total FPS is : 0.45fps 

The benchmark result of running my model on **IGPU[Intel HD Graphics 630]** with multiple model precisions are :
- FP32:
  - The total model loading time is : 65.697sec
  - The total inference time is : 9.0sec
  - The total FPS is : 0.4444fps
  
- FP16:
  - The total model loading time is : 66.4sec
  - The total inference time is : 9.4sec
  - The total FPS is : 0.425fps
 
  
## Results

- As we can see the above benchmark results, we can say that by using less precision model gives us faster inference.
- And also by reducing the precision, the usage of memory is less and its less computationally expensive when compared to higher precision models.
- By comparing the results between FP16 and INT8, the inference is same but the model loading time was more.
- Hence, by reducing the precision from FP32 to FP16 the model was able to run inference faster with more number of frames per second and with less model loading time.

## Stand Out Performances
- I've build an inference pipeline for both video file and webcam feed as input. Allowing the user to select their input option in the command line arguments:
  - ```-V``` argument takes the input video file or a webcam, for accessing video file the command is ```-V "<path of video file>"``` whereas for accessing webcam ```-V "cam"```. 
- I improved my model inference time by changing the precisions of the models, the following precisions been used on the models are: 
  - FP16 precision for **Face detection, Head Pose Estimation, Gaze Estimation** and FP32 precision for **Facial Landmarks Detection**.
  - The total model loading time is : 1.81sec
  - The total inference time is : 8.1sec
  - The total FPS is : 0.49fps

### Edge Cases
- Case 1) **Lighting**:
  - Lighting is considered to be one of the most important specification for both video file and webcam:
    - For webcam, if there is no enough lighting for the webcam to capture our face then, the models cannot detect our face and throws an exception error : **Could not run inference:...**.
	- For video file, before passing it to models, we have to check whether the video file which is being passed has the video of a person and the face of the person is clearer, if not then the program throws the exception error as above.
- Case 2) **Multiple persons in a frame**:
  - The mouse movements can be controlled by a single persons eye-gaze, If multiple persons are detected in the frame, The model immediately throws an exception error because, It causes an ambiguity while performing gaze estimation.
- To avoid such edge cases, we have to make sure that there is enough lighting and only a single person in the frame to run the project more robustly. 
