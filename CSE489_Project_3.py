from __future__ import division
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import math
import sys
#import RPi.GPIO as GPIO
import Adafruit_PCA9685

##################################################################################################
#GPIO initialization
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(17, GPIO.OUT)            #GPIO_17: is the main motors pin
#GPIO.setup(27, GPIO.OUT)            #GPIO_27: is the steering direction pin

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()
# Set frequency to 60hz
pwm.set_pwm_freq(60)

##################################################################################################
hand_hist = None
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


##################################################################################################
#Define Vertices
A = [320, 240]
B = [320, 0]
C = [320, 480-1]
D = [640-1, 240]
E = [0, 240]
F = [640-1, 0]
G = [0, 0]
H = [0, 480-1]
I = [640-1, 480-1]

#Define Quads for control
quad_F    = np.array([ [B[0]-20, B[1]],
		       [B[0]+20, B[1]], 
		       [A[0]+20, A[1]-20],
		       [A[0]-20, A[1]-20] ], np.int32)
#quad_F = quad_F.reshape((6))

quad_B    = np.array([ [A[0]-20, A[1]+20],
		       [A[0]+20, A[1]+20],
		       [C[0]+20, C[1]],
		       [C[0]-20, C[1]] ], np.int32)
#quad_B = quad_B.reshape((6))

quad_1    = np.array([ [B[0]+20, B[1]],
		       F,
		       [D[0], D[1]-20],
		       [A[0]+20, A[1]-20] ], np.int32)
#quad_1 = quad_1.reshape((6))

quad_2    = np.array([ G,
		       [B[0]-20, B[1]],
		       [A[0]-20, A[1]-20],
		       [E[0], E[1]-20] ], np.int32)
#quad_2 = quad_2.reshape((6))

quad_3 	  = np.array([ [E[0], E[1]+20],
		       [A[0]-20, A[1]+20],
		       [C[0]-20, C[1]],
		       H ], np.int32)
#quad_3 = quad_3.reshape((6))

quad_4	  = np.array([ [A[0]+20, A[1]+20],
		       [D[0], D[1]+20],
		       I,
		       [C[0]+20, C[1]] ], np.int32)
#quad_4 = quad_4.reshape((6))

quad_Idle = np.array([ [E[0], E[1]-20],
		       [D[0], D[1]-20],
		       [D[0], D[1]+20],
		       [E[0], E[1]+20] ], np.int32)
#quad_Idle = quad_Idle.reshape((6))

##################################################################################################
#functions:

#def rescale_frame(frame, wpercent=100, hpercent=100):
#    width = int(frame.shape[1] * wpercent / 100)
#    height = int(frame.shape[0] * hpercent / 100)
#    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 255), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
	cx=320		#image centre x
	cy=240		#image centre y
        return cx, cy

##################################################################################################
#Motor Control Pins diagram using PWM channels (0-5) from 16 channel PWM module
EN0 = 0		# channel_0 -> Driving motors -> Note: (0 <> 2^12) PWM range for speed control
EN1 = 5		# channel_1 -> Steering Motor -> Note: (0 <> 2^12) but used only 50%
#
#Direction Control pins on the motor driver, Note: always max PWM 2^12, thus used as digital pins
N1 = 1		# channel_2 -> N1
N2 = 2		# channel_3 -> N2
N3 = 3		# channel_4 -> N3
N4 = 4		# channel_5 -> N4
#
max_pulseW = 4095               #100% duty cycle
min_pulseW = 0			#  0% duty cycle
pwm_redu = 500
#
##################################################################################################


def dir_Control(state):                 #RC-Car direction control
    
    if   state == 'F':                  #forward
        #GPIO.output(17, GPIO.HIGH)
        pwm.set_pwm(EN0, 0, max_pulseW-pwm_redu)	#50% duty cycle
	pwm.set_pwm(N1,  0, max_pulseW)		#N1: HIGH
	pwm.set_pwm(N2,  0, min_pulseW)		#N2: LOW
	
        pwm.set_pwm(EN1, 0, min_pulseW)		# 0% duty cycle
	pwm.set_pwm(N3,  0, min_pulseW)		#N1: LOW
	pwm.set_pwm(N4,  0, min_pulseW)		#N1: LOW

    elif state == 'B':                  #backward
        #GPIO.output(17, GPIO.LOW)
        pwm.set_pwm(EN0, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N1,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N2,  0, max_pulseW)         #N2: HIGH
        
        pwm.set_pwm(EN1, 0, min_pulseW)         # 0% duty cycle 
        pwm.set_pwm(N3,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N4,  0, min_pulseW)         #N1: LOW

    elif state == '1':                  #forward and right
        #GPIO.output(17, GPIO.HIGH)
        pwm.set_pwm(EN0, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N1,  0, max_pulseW)         #N1: HIGH
        pwm.set_pwm(N2,  0, min_pulseW)         #N2: LOW

        pwm.set_pwm(EN1, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N3,  0, max_pulseW)         #N1: HIGH
        pwm.set_pwm(N4,  0, min_pulseW)         #N1: LOW

    elif state == '2':                  #forward and left
        #GPIO.output(17, GPIO.HIGH)
        pwm.set_pwm(EN0, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N1,  0, max_pulseW)         #N1: HIGH
        pwm.set_pwm(N2,  0, min_pulseW)         #N2: LOW

        pwm.set_pwm(EN1, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N3,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N4,  0, max_pulseW)         #N1: HIGH

    elif state == '3':                  #backward and left
        #GPIO.output(17, GPIO.LOW)
        pwm.set_pwm(EN0, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N1,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N2,  0, max_pulseW)         #N2: HIGH

        pwm.set_pwm(EN1, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N3,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N4,  0, max_pulseW)         #N1: HIGH

    elif state == '4':                  #backward and right
        #GPIO.output(17, GPIO.LOW)
        pwm.set_pwm(EN0, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N1,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N2,  0, max_pulseW)         #N2: HIGH

        pwm.set_pwm(EN1, 0, max_pulseW-pwm_redu)    #50% duty cycle
        pwm.set_pwm(N3,  0, max_pulseW)         #N1: HIGH
        pwm.set_pwm(N4,  0, min_pulseW)         #N1: LOW

    elif state == 'X':                  #Idle Zone
        pwm.set_pwm(EN0, 0, min_pulseW)         # 0% duty cycle
        pwm.set_pwm(N1,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N2,  0, min_pulseW)         #N2: LOW

        pwm.set_pwm(EN1, 0, min_pulseW)         # 0% duty cycle
        pwm.set_pwm(N3,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N4,  0, min_pulseW)         #N1: LOW

    else:				#Default Brake Zone
        pwm.set_pwm(EN0, 0, min_pulseW)		# 0% duty cycle
        pwm.set_pwm(N1,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N2,  0, min_pulseW)         #N2: LOW

        pwm.set_pwm(EN1, 0, min_pulseW)         # 0% duty cycle
        pwm.set_pwm(N3,  0, min_pulseW)         #N1: LOW
        pwm.set_pwm(N4,  0, min_pulseW)         #N1: LOW

##################################################################################################


def d_angle(x1, y1, x2, y2):

    theta_1 = np.arctan2(y1, x1)
    theta_2 = np.arctan2(y2, x2)

    d_theta = np.subtract(theta_1, theta_2)

    while (d_theta > np.pi):
	d_theta = d_theta - 2*np.pi
    while (d_theta < -np.pi):
	d_theta = d_theta + 2*np.pi

    return d_theta


def insideQuad(quad, target):

    angle = 0.0
    p1 = [0,0]
    p2 = [0,0]

    for i in range(4):
	for j in range(2):
	    p1[j] = quad[i,j] - target[j]
	    p2[j] = quad[(i+1)%3,j] - target[j]

	angle = angle + d_angle(p1[0],p1[1],p2[0],p2[1])


    if (abs(angle) < np.pi):
	return False
    else:
	return True



def control(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    xy_centroid = centroid(max_cont)
    cv2.circle(frame, xy_centroid, 5, [0, 0, 255], -1)
    #print "(x,y)= ", xy_centroid

    #direction control
    if insideQuad(quad_F, xy_centroid) == True:
	dir_Control('F')
	print "Forward"

    elif insideQuad(quad_B, xy_centroid) == True:
	dir_Control('B')
	print "Backward"

    elif insideQuad(quad_1, xy_centroid) == True:
	dir_Control('1')
	print "1st Quad"

    elif insideQuad(quad_2, xy_centroid) == True:
        dir_Control('2')
	print "2nd Quad"

    elif insideQuad(quad_3, xy_centroid) == True:
        dir_Control('3')
	print "3rd Quad"

    elif insideQuad(quad_4, xy_centroid) == True:
        dir_Control('4')
	print "4th Quad"

    elif insideQuad(quad_Idle, xy_centroid) == True:
	dir_Control('X')
	print "Idle Region"

    else:
	dir_Control('S')
	print "Stop"


###################################################################################################

def main():

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=(640, 480))

    #Global Variables

    global hand_hist
    global quad_1
    quad_1_draw = quad_1.reshape((-1,1,2))

    global quad_2
    quad_2_draw = quad_2.reshape((-1,1,2))

    global quad_3
    quad_3_draw = quad_3.reshape((-1,1,2))

    global quad_4
    quad_4_draw = quad_4.reshape((-1,1,2))

    is_hand_hist_created = False

    #Main loop

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

	pressed_key = cv2.waitKey(1)
        image = frame.array

        if pressed_key & 0xFF == ord('x'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(image)

        if is_hand_hist_created:
            control(image, hand_hist)
	    #Quads Drawing
            cv2.polylines(image,[quad_1_draw],True,(0,255,255), 2)
            cv2.polylines(image,[quad_2_draw],True,(0,255,255), 2)
            cv2.polylines(image,[quad_3_draw],True,(0,255,255), 2)
            cv2.polylines(image,[quad_4_draw],True,(0,255,255), 2)


        else:
            image = draw_rect(image)


	#image = cv2.flip(image, -1)
        cv2.imshow("Car Controller", image)
	
	# clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        if pressed_key == 27:
            break



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print 'Interrupted'
    #finally:
	#GPIO.cleanup()
