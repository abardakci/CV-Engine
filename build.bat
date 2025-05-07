@echo off
REM CUDA_NVCC_EXECUTABLE="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe"
REM Visual Studio 2022 paths
set VS_PATH="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64"

echo Cleaning previous build...
if exist build\ninja-release rmdir /s /q build\ninja-release

echo Configuring project with CMake presets...
cmake --preset=windows-ninja
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b %errorlevel%
)

echo Building project with Ninja...
cd build\ninja-release
ninja
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b %errorlevel%
)

echo Build completed successfully!

echo ----------------------------PROGRAM START----------------------------
cd ..\..

REM Optional: Run the compiled executable
REM echo Running the output...
REM build\ninja-release\YourExecutableName.exe

@echo off
"build\ninja-release\RealUpscale.exe"

echo ----------------------------PROGRAM END----------------------------

pause