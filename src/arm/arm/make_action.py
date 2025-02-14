"""
make_action node

Idea: DQN Forwarding -> select action from 27 possible action combination

Requirements
1) subcribe array state (length 18)
2) define pytorch layer & forwarding 
3) define service server, serve index of the action
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile

state_dim = 13
action_dim = 27



def select_action(vector):
    #select one element, for DQN
    return 1

class ACTOR(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACTOR, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x)) #relu
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


class MAKE_ACTION(Node):
    def __init__(self):
        super().__init__('make_action')
        qos_profile = QoSProfile(depth=10)

        #get pytorch layer
        self.actor = ACTOR(state_dim, action_dim)

        #subscriber
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/state', # length 18 array state
            self.forwarding,
            qos_profile)
        self.subscription

        #service server
        self.publisher = self.create_publisher(Int32, 'action', qos_profile)

    def forwarding(self, msg):
        #self.get_logger().info('state received, forwarding...')
        
        self.state = np.array(msg.data)
        self.state = torch.from_numpy(self.state)

        #forward through network
        result_probability = self.actor(self.state).detach().numpy()

        #choose the highest among probabilities (make it integer)
        index = result_probability.argmax().item()

        #publish
        act = Int32()
        act.data = index
        self.publisher.publish(act)


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