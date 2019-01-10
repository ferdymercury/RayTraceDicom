cmake_minimum_required(VERSION 2.8)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(make_documentation)
WRITE_CMAKE_DOCUMENTATION( "${CMAKE_CURRENT_BINARY_DIR}/cmake.dox" SORTED )
