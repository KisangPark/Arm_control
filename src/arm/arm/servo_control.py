"""
servo_control node

Requirements
1) spinning, send request for action index
2) when index, appropriate GPIO servo action
3) spinning -> every spin, send request, publish servo position
"""

joint_names = ['bodyjoint_1', 'bodyjoint_2', 'bodyjoint_3']
length = len(joint_names)
print(joint_names)


import RPi.GPIO as GPIO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile

servoPin1          = 8
servoPin2          = 10
servoPin3          = 12

SERVO_MAX_DUTY    = 12   # duty for 180 degree
SERVO_MIN_DUTY    = 3    # duty for 0 degree


def calc_duty(angle):
    #receive angle, calculate duty for gpio
    duty = SERVO_MIN_DUTY+(angle*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
    return duty

def select_action(index):
    """
    total 27 action space
    000, 001, 002, 010, 011, 012, 020, 021, 022
    100, 101, 102, 110, 111, 112, 120, 121, 122
    200, 201, 202, 210, 211, 212, 220, 221, 222
    -> 1st joint by very first bit -> compare with size
    -> 2nd joint by 3으로 나눈 몫이 짝수이면면
    -> 3rd joint by dividing 3
    """
    action_set = [(0,0,0), (0,0,1), (0,0,-1), (0,1,0), (0,1,1), (0,1,-1), (0,-1,0), (0,-1,1), (0,-1,-1),
                (1,0,0), (1,0,1), (1,0,-1), (1,1,0), (1,1,1), (1,1,-1), (1,-1,0), (1,-1,1), (1,-1,-1),
                (-1,0,0), (-1,0,1), (-1,0,-1), (-1,1,0), (-1,1,1), (-1,1,-1), (-1,-1,0), (-1,-1,1), (-1,-1,-1)]
    
    return action_set[index]



class SERVO_CONTROL(Node):
    def __init__(self):
        super().__init__('servo_control')
        
        qos_profile = QoSProfile(depth=10)

        #setup GPIO
        #GPIO.setmode(GPIO.BOARD)
        #GPIO.setup(servoPin1, GPIO.OUT)
        #GPIO.setup(servoPin2, GPIO.OUT)
        #GPIO.setup(servoPin3, GPIO.OUT)

        #servo motor definition
        #servo1 = GPIO.PWM(servoPin1, 50)
        #servo2 = GPIO.PWM(servoPin2, 50)
        #servo3 = GPIO.PWM(servoPin3, 50)
        #servo1.start(0)
        #servo2.start(0)
        #servo3.start(0)

        #subscriber
        self.subscription = self.create_subscription(
            Int32,
            '/action', # integer of action index
            self.action_callback,
            qos_profile
        )
        self.subscription
        
        #publish angle value of servos, float array
        self.publisher_sai = self.create_publisher(Float32MultiArray, 'angle', qos_profile)

        #publish JointState for rviz & robot state publisher
        self.publisher_js = self.create_publisher(JointState, 'joint_states', qos_profile)


        #initialize angle with 90
        # important: servo value is modified as 0~180 on code,
        #              but in xacro it's -90~90
        self.servo_angle = [90, 90, 90]
        self.joint_state = [0,0,0]

        self.sai = Float32MultiArray()#servo angle info initialize
        self.js = JointState()#joint state initialize
        self.js.header.frame_id = "joint_states"
        self.js.name = joint_names
        self.js.position = np.zeros(length).tolist()

    def action_callback(self,msg):

        #1) calculate self angle
        action_list = select_action(msg.data)
        for i, value in enumerate(action_list):
            self.servo_angle[i] = max(0, min(180, self.servo_angle[i] + value)) #cat
            self.joint_state[i] = max(-90, min(90, self.joint_state[i] + value)) #cat

        #2) call gpio control method
        #self.servo_signal()

        #3) create message
        self.sai.data = self.servo_angle
        #publish angle
        self.publisher_sai.publish(self.sai)

        #4) create JointState
        self.js.position = self.joint_state
        self.js.effort = []
        self.js.velocity = []
        self.publisher_js.publish(self.js)


    def servo_signal(self):
        #get self.angle, change into duty, servo control
        duty_list = calc_duty(self.angle)
        
        for i, servo in enumerate(servo1, servo2, servo3):
            servo.ChangeDutyCycle(duty_list[i])

 



def main(args=None):
    #main function call
    rclpy.init(args=args)
    node = SERVO_CONTROL()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    """main function"""
    main()