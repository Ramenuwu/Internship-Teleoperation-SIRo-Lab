import os
from ament_index_python.packages import get_package_share_directory
from controller_manager.launch_utils import generate_load_controller_launch_description
from launch.actions import GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription, LaunchContext
from launch_pal.arg_utils import LaunchArgumentsBase, read_launch_argument
from launch_pal.param_utils import parse_parametric_yaml
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration
from dataclasses import dataclass
from launch_pal.robot_arguments import CommonArgs
from launch_ros.actions import Node 


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):
    side: DeclareLaunchArgument = CommonArgs.side


def declare_actions(launch_description: LaunchDescription, launch_args: LaunchArguments):
    launch_description.add_action(OpaqueFunction(
        function=setup_controller_configuration))
    
    print("a1")

    joint_effort_controller = GroupAction([generate_load_controller_launch_description(
        controller_name=LaunchConfiguration("controller_name"),
        controller_params_file=LaunchConfiguration("controller_config"),
        extra_spawner_args=["--inactive"])])
    launch_description.add_action(joint_effort_controller)

    print("a2")

    return


def setup_controller_configuration(context: LaunchContext):
    side = read_launch_argument('side', context)
    arm_prefix = 'arm_right'
    if side:
        arm_prefix = f'arm_{side}'

    controller_name = f'{arm_prefix}_effort_controller'
    param_file = os.path.join(
        get_package_share_directory('teleop'),
        'config', 'joint_effort_controller.yaml')

    remappings = {"ARM_SIDE_PREFIX": arm_prefix}
    parsed_yaml = parse_parametric_yaml(
        source_files=[param_file], param_rewrites=remappings)

    gravcomp_node = Node(
        package='teleop',                        
        executable=f'joint_effort_{side}_gravcomp',
        name=f'joint_effort_{side}_gravcomp',
        output='screen',
    )

    return [
        SetLaunchConfiguration('controller_name', controller_name),
        SetLaunchConfiguration('controller_config', parsed_yaml),
        gravcomp_node,
    ]


def generate_launch_description():
    print("1")
    ld = LaunchDescription()
    print("2")
    launch_arguments = LaunchArguments()
    launch_arguments.add_to_launch_description(ld)
    print("3")
    declare_actions(ld, launch_arguments)
    print("4")
    return ld