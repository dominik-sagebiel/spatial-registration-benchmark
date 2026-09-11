cmake_minimum_required(VERSION 3.18)
project(sift_vs_superpoint)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
# Point this at the folder that contains "share/cmake/Torch"
set(CMAKE_PREFIX_PATH "C:/libtorch") 
find_package(Torch REQUIRED)

# --- OpenCV ---
set(OpenCV_DIR "C:/Users/domin/opencv-4.10.0/opencv/build")
set(OpenCV_ARCH "x64")
set(OpenCV_RUNTIME "vc16")
find_package(OpenCV REQUIRED)

add_executable(sift_vs_superpoint SIFTvsSuperPoint.cpp)

target_link_libraries(sift_vs_superpoint
    "${TORCH_LIBRARIES}"
    "${OpenCV_LIBS}"
)

target_include_directories(sift_vs_superpoint PRIVATE
    "${OpenCV_INCLUDE_DIRS}"
    "${CMAKE_SOURCE_DIR}/third_party"
)

set_property(TARGET sift_vs_superpoint PROPERTY CXX_STANDARD 17)

# Windows only: copy required DLLs next to the .exe after build
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET sift_vs_superpoint
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:sift_vs_superpoint>)
 
    file(GLOB OPENCV_DLLS "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/*.dll")
    add_custom_command(TARGET sift_vs_superpoint
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${OPENCV_DLLS}
                        $<TARGET_FILE_DIR:sift_vs_superpoint>)
endif (MSVC)
