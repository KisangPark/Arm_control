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
    package_name = "arm"

    #robot state publisher, launch
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory(package_name), "launch", "xacro_load.launch.py")]
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    #workspace package
    pkg_path = os.path.join(get_package_share_directory('arm'))

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
            rsp,
        ]
    )

