"""
controller launch
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable, 
                            IncludeLaunchDescription, SetLaunchConfiguration)
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import xacro

def generate_launch_description():

    #workspace package
    pkg_path = os.path.join(get_package_share_directory('arm'))

    #trying xacro instead of urdf!
    xacro_file = os.path.join(pkg_path, "xacro", "arm.xacro")
    robot_description = xacro.process_file(xacro_file)
    params = {"robot_description": robot_description.toxml()}

    #robot state publisher
    robot_state_publisher = Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                #namespace="vision60", #vision60 namespace
                output="screen",
                parameters=[params],)

    #get_frame node
    get_frame = Node(
    package='arm',
    executable='get_frame',
    name='get_frame',
    )

    #make_action node
    make_action = Node(
    package='arm',
    executable='make_action',
    name='make_action',
    )

    #servo_control node
    servo_control = Node(
    package='arm',
    executable='servo_control',
    name='servo_control',
    )

    return LaunchDescription(
        [
            get_frame,
            make_action,
            servo_control,
        ]
    )

