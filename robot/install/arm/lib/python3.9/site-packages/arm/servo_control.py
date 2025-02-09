"""
servo_control node

Requirements
1) spinning, send request for action index
2) when index, appropriate GPIO servo action
3) spinning -> every spin, send request, publish servo position
"""

import RPi.GPIO as GPIO

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile

servoPin1          = 12
servoPin2          = 13
servoPin3          = 14

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



class MAKE_ACTION(Node):
    def __init__(self):
        super().__init__('make_action')
        
        qos_profile = QoSProfile(depth=10)

        #setup GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(servoPin1, GPIO.OUT)
        GPIO.setup(servoPin2, GPIO.OUT)
        GPIO.setup(servoPin3, GPIO.OUT)

        #servo motor definition
        servo1 = GPIO.PWM(servoPin1, 50)
        servo2 = GPIO.PWM(servoPin2, 50)
        servo3 = GPIO.PWM(servoPin3, 50)
        servo1.start(0)
        servo2.start(0)
        servo3.start(0)

        #service client
        #?

        #publish servo angle, float array
        #self.publisher = 

        #initialize angle with 90
        self.angle = [90, 90, 90]


    def action_client(self,msg):
        #request for action index
        #send request

        #calculate self angle
        action_arr = select_action(msg.data)
        for i, value in enumerate(action_arr):
            self.angle[i] = max(0, min(180, self.angle[i] + value)) #cat

        #call gpio control method
        self.servo_signal()

        #create message
        msg = Float32MultiArray()
        msg.data = self.angle
        #publish angle
        publish(msg)


    def servo_signal(self):
        #get self.angle, change into duty, servo control
        duty_list = calc_duty(self.angle)
        
        for i, servo in enumerate(servo1, servo2, servo3):
            servo.ChangeDutyCycle(duty_list[i])

 



def main(args=None):
    #main function call
    rclpy.init(args=args)
    node = MAKE_ACTION()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    """main function"""
    main()