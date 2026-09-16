cmake_minimum_required(VERSION 3.15)

project(rknn_llm_ppl_eval C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT RKNN3_MODEL_ZOO_ROOT)
  message(FATAL_ERROR "Set RKNN3_MODEL_ZOO_ROOT to rknn3-model-zoo-1.1.0")
endif()

add_subdirectory(${RKNN3_MODEL_ZOO_ROOT}/3rdparty
                 ${CMAKE_CURRENT_BINARY_DIR}/model_zoo_3rdparty)
find_package(OpenMP REQUIRED)

add_executable(rknn_llm_ppl_eval main.cc)
target_include_directories(rknn_llm_ppl_eval PRIVATE
  ${LIBRKNNAPI_INCLUDES}
  ${LIBTOKENIZER_INCLUDES}
  ${RKNN3_MODEL_ZOO_ROOT}/3rdparty/json
)
target_link_libraries(rknn_llm_ppl_eval PRIVATE
  ${LIBRKNNAPI}
  ${LIBTOKENIZER}
  dl
  pthread
  OpenMP::OpenMP_CXX
)

set_target_properties(rknn_llm_ppl_eval PROPERTIES
  INSTALL_RPATH "$ORIGIN/lib"
)

install(TARGETS rknn_llm_ppl_eval DESTINATION .)
