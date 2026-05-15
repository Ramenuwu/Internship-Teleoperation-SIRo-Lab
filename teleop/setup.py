from setuptools import find_packages, setup

package_name = 'teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/joint_effort_controller.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/joint_effort_controller.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "teleop_node = teleop.teleop_node:main",
            "eepos_node = teleop.eepos_node:main",
            "impedance = teleop.impedance:main",
            "impedance_left = teleop.impedance_left:main",
            "effort_node = teleop.effort_node:main",
            "pinPos_node = teleop.pinPos_node:main",
            "fktest_node = teleop.fktest_node:main",
            "joint_effort_right_gravcomp = teleop.joint_effort_right_gravcomp:main",
            "joint_effort_left_gravcomp = teleop.joint_effort_left_gravcomp:main"
        ],
    },
)
