import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    package_name = 'fura2a'
    this_dir = get_package_share_directory(package_name)
    
    # File paths
    gazebo_params_file = os.path.join(this_dir, 'gazebo', 'gazebo_params.yaml')
    world_file = os.path.join(this_dir, 'worlds', 'basic.world')
    rviz_config_file = os.path.join(this_dir, 'rviz', 'display.rviz')

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(
                get_package_share_directory(package_name), 'launch', 'rsp.launch.py'
            )]
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Include the Gazebo launch file, with the world file specified
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(
                get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py'
            )]
        ),
        launch_arguments={
            # 'world': world_file,
            'pause': 'false',
            'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file
        }.items()
    )


    # Run the spawner node from the gazebo_ros package
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'fura2a'],
        output='screen'
    )

    arm_cont_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_cont"],
    )

    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broad"],
    )


    # Add RViz2 node with the custom display configuration
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity,
        arm_cont_spawner,
        joint_broad_spawner,
        # rviz2
    ])
