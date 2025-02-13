"""
get_frame node

Requirements
1) cv2.videocapture in initialization
2) get frame & calculate coordinates using cv in method
3) subscribe angle from controller
4) publish State topic
"""


import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray #Float32MultiArray
from rclpy.qos import QoSProfile


#red box detection
#return 4x2 matrix for edge position

def detect_red_box(frame):
    # Load image and convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color ranges in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create red color masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check for quadrilateral
        if len(approx) == 4:
            # Order vertices clockwise
            vertices = approx.reshape(4, 2).tolist()
            out = sorted(vertices, key=lambda x: (x[0], x[1]))
            print(out)
            return out

    return None



#green dot - robot tip detection
#return vector of position

def detect_green_dot(image):
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for green color in HSV
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the green dot)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return [cX, cY]
    
    return None


class GET_FRAME(Node):
    def __init__(self):
        super().__init__('get_frame')
        qos_profile = QoSProfile(depth=10)

        #cv video capture
        #self.cap = cv2.VideoCapture(0) # '/dev/video1' #*********************************************************************************
        #alter: image of red box and green dot
        self.image = cv2.imread('/home/kisang/Arm_control/src/arm/redbox_greendot.jpg')

        #subscriber
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/angle', # length 3 vector of integer angles
            self.record_angle,
            qos_profile
        )
        self.subscription #prevent unused variable warning

        #publisher
        self.publisher = self.create_publisher(Float32MultiArray,
        'state', qos_profile)

        #set timer
        self.timer = self.create_timer(0.1, self.get_frame)


    def record_angle(self, msg):
        self.get_logger().info('angle received from controller')
        self.servo_angle = msg.data
        #record angle from subscription


    def get_frame(self):
        #read camera
        try:
            frame = self.image
            #ret, frame = self.cap.read()   #*********************************************************************************
            #cv2.imshow('test', frame) #works only when run separately
            #cv2.waitKey(1)
            #process image -> get coordinate of vertices, 4x2 matrix
            vertex_list = detect_red_box(frame) # ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!
            #get coordinate of arm tip, 1x2 vector
            arm_tip = detect_green_dot(frame)

            self.get_logger().info("got box and dot")

            # append all
            state = np.array(vertex_list).flatten().tolist()
            state += arm_tip
            try:
                state += self.servo_angle #angle info received
                self.get_logger().info("servo angle appended")
            except:
                state += np.zeros(3).tolist()

        except:
            self.get_logger().info("cap not opened")
            state = np.zeros(13)

        #publish
        state_array = Float32MultiArray()
        state_array.data = np.float32(state).tolist()
        #self.get_logger().info('successfully transitioned, publishing...')
        self.publisher.publish(state_array)



def main(args=None):
    #main function call
    rclpy.init(args=args)
    node = GET_FRAME()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    """main function"""
    main()