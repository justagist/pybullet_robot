cmake_minimum_required(VERSION 2.8.3)
project(pybullet_robot)

find_package(catkin REQUIRED COMPONENTS
    franka_panda_description
)

catkin_package()

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

catkin_python_setup()
