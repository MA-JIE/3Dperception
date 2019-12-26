cmake_minimum_required(VERSION 2.8.3)
project(3D_perception)
add_compile_options(-std=c++11)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(VISP REQUIRED)
find_package(OpenCV 3  REQUIRED)
find_Package(G2O REQUIRED)
find_Package(Eigen3 REQUIRED)
Find_Package(Sophus REQUIRED)
find_Package(CSparse REQUIRED)
include_directories(
  include
  ${VISP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
)
add_executable(3D_perception main.cpp include/3D_perception.h src/3D_perception.cpp)
target_link_libraries(3D_perception ${VISP_LIBRARIES}  ${OpenCV_LIBS}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
    ${CSPARSE_LIBRARY})
