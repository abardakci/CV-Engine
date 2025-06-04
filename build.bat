@echo off
REM CUDA_NVCC_EXECUTABLE="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe"
REM Visual Studio 2022 paths
@REM set VS_PATH="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64"

echo Cleaning previous build...
if exist build\msvc-debug rmdir /s /q build\msvc-debug

echo Configuring project with CMake presets...
cmake --preset=msvc-debug
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b %errorlevel%
)

echo Building project with Ninja...
cd build\msvc-debug
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
REM build\msvc-debug\YourExecutableName.exe

@echo off
"build\msvc-debug\VisionEngine.exe"

echo ----------------------------PROGRAM END----------------------------

pause