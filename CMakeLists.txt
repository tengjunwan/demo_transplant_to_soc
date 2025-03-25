cmake_minimum_required(VERSION 3.5)
PROJECT(ss928_objectTrack)

include(${CMAKE_SOURCE_DIR}/cmake/ss928.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/cmakebase.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/project.cmake)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/output/")

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}/ss928sdk/common/
    ${CMAKE_SOURCE_DIR}/ss928sdk/common/audio/adp
    ${CMAKE_SOURCE_DIR}/ss928sdk/include/
    ${CMAKE_SOURCE_DIR}/ss928sdk/include/npu
    ${CMAKE_SOURCE_DIR}/ss928sdk/include/svp_npu
    ${CMAKE_SOURCE_DIR}/svp/
    ${CMAKE_SOURCE_DIR}/svp/common/
    ${CMAKE_SOURCE_DIR}/svp/npu/include/
    ${CMAKE_SOURCE_DIR}/svp/npu/sample_svp_npu/
    ${CMAKE_SOURCE_DIR}/svp/npu/
)
LINK_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}/ss928sdk/lib/
    ${CMAKE_SOURCE_DIR}/ss928sdk/lib/npu/
    ${CMAKE_SOURCE_DIR}/ss928sdk/lib/svp_npu
)

aux_source_directory(${CMAKE_SOURCE_DIR}/ss928sdk/common/ SOCSDKCOMMON_SRCLIST)
aux_source_directory(${CMAKE_SOURCE_DIR}/ss928sdk/common/audio SOCSDKCOMMON_SRCLIST)
aux_source_directory(${CMAKE_SOURCE_DIR}/svp/common/ SVP_SRCLIST)
aux_source_directory(${CMAKE_SOURCE_DIR}/svp/npu/sample_svp_npu SVP_SRCLIST)
aux_source_directory(${CMAKE_SOURCE_DIR}/svp/npu/ SVP_SRCLIST)
aux_source_directory(${CMAKE_SOURCE_DIR}/svp/ SVP_SRCLIST)
aux_source_directory(${CMAKE_SOURCE_DIR}/sample/ SVP_SRCLIST)

add_executable(${PROJECT_NAME} ${SVP_SRCLIST} ${SOCSDKCOMMON_SRCLIST} )

target_link_libraries(${PROJECT_NAME} ${SOC_LIBS} -lm -lpthread -ldl)