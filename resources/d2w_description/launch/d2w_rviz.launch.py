"""ROS 2 display-only launch; this does not start a simulator or hardware."""
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("d2w_description"))
    source = share / "urdf" / "d2w_description.urdf"
    robot = ET.parse(source).getroot()
    # Keep the canonical URDF portable for direct file import. Resolve its local
    # mesh references only in the in-memory ROS robot_description parameter.
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename", "")
        if not filename:
            raise ValueError("A mesh filename is missing in the D2W URDF")
        if "://" in filename:
            continue
        target = (source.parent / filename).resolve()
        if not target.is_file() or not target.is_relative_to(share.resolve()):
            raise FileNotFoundError(f"Missing or out-of-package mesh: {filename}")
        mesh.set("filename", "package://d2w_description/" + target.relative_to(share.resolve()).as_posix())
    description = ET.tostring(robot, encoding="unicode")
    use_gui = LaunchConfiguration("gui")
    use_rviz = LaunchConfiguration("rviz")
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    params = {"robot_description": description, "use_sim_time": use_sim_time}
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="Enable joint sliders"),
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz2"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[params], output="screen"),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
             parameters=[params], condition=IfCondition(use_gui), output="screen"),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             parameters=[params], condition=UnlessCondition(use_gui), output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(share / "launch" / "check_joint.rviz")],
             parameters=[{"use_sim_time": use_sim_time}], condition=IfCondition(use_rviz), output="screen"),
    ])
