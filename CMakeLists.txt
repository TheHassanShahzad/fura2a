cmake_minimum_required(VERSION 3.8)
project(fura2a)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(xacro REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(rviz2 REQUIRED)
find_package(control_msgs REQUIRED)
find_package(ros2_control REQUIRED)
find_package(ros2_controllers REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install resource files (e.g., URDFs, meshes, configs, etc.)
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY gazebo
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  fura2a/simple_publisher.py  # Add all Python scripts explicitly
  fura2a/mpc.py
  DESTINATION lib/${PROJECT_NAME}
)

# Use ament_python to install the Python package
ament_python_install_package(${PROJECT_NAME})

# Enable testing (if applicable)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip linter checks if necessary (remove these lines when adding copyrights/licenses)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
