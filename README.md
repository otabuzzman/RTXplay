# RTXplay
A lab for playing around with NVIDIA's realtime raytracing concept. The lab intends to migrate the raytracer presented by RTOW to OptiX Ray Tracing Engine and thus utilizing the new RT cores which had been introduced by Turing microarchitecture.

### Build
Due to lack of appropriate hardware development and tests had been split on Windows and Linux respectively.

#### Windows (Development)
- Install Cygwin with development tools (GCC, make etc.)
- Install Visual Studio 2017 Community
- Unpack CUDA Toolkit 11 somewhere, e.g. `/usr/lab/cudacons/cuda_11.1.1_456.81_win10` (Cygwin)
- Unpack Optix 7 SDK somewhere, e.g. `/usr/lab/cudacons/NVIDIA-OptiX-SDK-7.2.0-win64` (Cygwin)
- Add `nvcc.exe` and `cl.exe` to PATH
- Install ImageMagick 7 (Windows installer)
- Run `make` in top-level directory of repo (compilation only, no linking)

#### Linux
- Get appropriate hardware, e.g. AWS EC2 instance type `g4dn.xlarge` with Amazon Linux AMI
  ```
  # on first start
  sudo yum update -y
  sudo reboot
  ```

- Install CUDA Toolkit 11
  ```
  # prerequisites
  sudo yum group install "Development Tools"

  wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
  sudo sh cuda_11.1.1_455.32.00_linux.run

  # environment (~/.bashrc)
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

  # check driver
  nvidia-smi

  # checko toolkit
  cd NVIDIA_CUDA-11.1_Samples/1_Utilities/deviceQuery
  make ; ../../bin/x86_64/linux/release/deviceQuery
  ```

- Install OptiX 7 SDK
  ```
  # prerequisites
  sudo yum install -y cmake3 freeglut-devel libXcursor-devel libXinerama-devel libXrandr-devel

  sudo mkdir /usr/local/optix-7.2
  sudo ln -s /usr/local/optix-7.2 /usr/local/optix
  sudo sh NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64.sh --prefix=/usr/local/optix-7.2

  # environment (~/.bashrc)
  export LD_LIBRARY_PATH=~/optix-samples/lib:$LD_LIBRARY_PATH

  # check
  mkdir optix-samples ; cd optix-samples
  ccmake3 /usr/local/optix/SDK ; make
  ```

- Install ImageMagick 7
  ```
  sudo yum install libpng-devel
  git clone https://github.com/ImageMagick/ImageMagick.git
  cd ImageMagick
  ./configure --disable-opencl ; make -j $(nproc) ; sudo make install
  ```

- Run `make` in `optx` directory of repo
