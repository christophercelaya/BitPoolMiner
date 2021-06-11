cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(bitpoolminer LANGUAGES CXX CUDA)

add_executable(bitpoolminer miner.cpp clouthash.cu)
target_link_libraries(bitpoolminer sioclient)
target_compile_features(bitpoolminer PUBLIC cxx_std_11)
set_target_properties(bitpoolminer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
