# RC_Car_HandGesture_Control_RPi
Is a project using a Raspberry pi to control an RC Car using hand gestures.

Important Libraries used:
>> openCV               #open source computer vision functions
>> Adafruit_PCA9685     #(16-channel, 12bit-resolution) PWM module adafruit libraries
>> picamera             #For setting Camera parameters



#To run the code: make sure you have an access to the streaming openCV GUI, otherwise the code will give an error unless you comment the cv2.imshow() fuction, but the project won't actually work without displaying the streamed frames on the GUI, as it requires some inputs from the user, for more info read the detailed technical documentation.
#one simple solution for the above discussed problem is to use vncserver to connect the pi's GUI to your laptop as a remote desktop X.

#command to run the code:
>> python CSE489_Project_3.py


#References:
>> https://github.com/amarlearning/Finger-Detection-and-Tracking
>> https://learn.adafruit.com/adafruit-16-channel-servo-driver-with-raspberry-pi/hooking-it-up?fbclid=IwAR2_nDJPFG5PvKZnEwYcxaXUJwGf06BaMI7vc7fQvG9bYqBDBnnAjmnSEhQ
