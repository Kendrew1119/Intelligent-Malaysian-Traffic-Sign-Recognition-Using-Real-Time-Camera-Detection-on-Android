# ============================================
# [Member 2] CMakeLists.txt
# ============================================
# Module: C++ Build Configuration for Android
# Owner: Member 2 (Systems Architect)
#
# Purpose:
#   - Configure CMake to build the "signdetector" native library
#   - Link: ncnn + OpenCV + Android log + JNI
#   - Set C++17 standard
#   - Configure ncnn and OpenCV paths (from .env.example)
#
# Build Flow:
#   Gradle → CMake → compile .cpp files → link libraries → produce libsigndetector.so
#   The .so file is packaged into the APK automatically
#
# References:
#   - Android CMake: https://developer.android.com/ndk/guides/cmake
#   - ncnn Android build: https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android
#   - See plan.md → ncnn Integration Guide → Step 3
#
# TODO: Implement full CMake configuration
# ============================================

cmake_minimum_required(VERSION 3.22)
project(signdetector)
