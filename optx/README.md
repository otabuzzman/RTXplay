# RTWO
This is *Ray Tracing with OptiX* (RTWO), another implementation of Pete Shirley's famous *Ray Tracing in One Weekend* (RTOW). It runs in batch mode just like Pete's but turns interactive in case of an X server exec'ing on the same host. A rather simple UI provides camera and denoiser controls. Compile and run `rtwo -h` for a complete list of functions:
```
Usage: rtwo [OPTION...]
  `rtwo´ renders the final image from Pete Shirley's book Ray Tracing in
  One Weekend using NVIDIA's OptiX Ray Tracing Engine and pipes the result
  (PPM) to stdout for easy batch post-processing (e.g. ImageMagick).

  If the host execs an X server (GLX enabled) as well, `rtwo´ continuously
  renders and displays results. A (rather) simple UI allows for basic
  interactions.

Examples:
  # render and convert image to PNG. Print statistical information on stderr.
  rtwo -S | magick ppm:- rtwo.png

  # output image and AOV (yields rtwo-0.png (image) and rtwo-1.png (RPP)).
  rtwo --print-aov RPP | magick - rtwo.png

  # improve AOV RPP contrast.
  magick rtwo-1.png -auto-level -level 0%,25% rtwo-rpp.png

  # apply denoiser and output guide layers (yields rtwo-0.png (normals),
  # rtwo-1.png (albedos),and rtwo-2.png (image)).
  rtwo --print-guides --apply-denoiser NAA | magick - rtwo.png

Options:
  -g, --geometry {<width>[x<height>]|RES}
    Set image resolution to particular dimensions or common values predefined
    by RES.

    Prefedined resolutions for RES:
      CGA    - 320x200      8:5
      HVGA   - 480x320      3:2
      VGA    - 640x480      4:3
      WVGA   - 800x480      5:3
      SVGA   - 800x600      4:3
      XGA    - 1024x768     4:3
      HD     - 1280x720    16:9 (default)
      SXGA   - 1280x1024    5:4
      UXGA   - 1600x1200    4:3
      FullHD - 1920x1080   16:9
      2K     - 2048x1080  ~17:9
      QXGA   - 2048x1536    4:3
      UWHD   - 2560x1080  ~21:9
      WQHD   - 2560x1440   16:9
      WQXGA  - 2560x1600    8:5
      UWQHD  - 3440x1440  ~21:9
      UHD-1  - 3840x2160   16:9
      4K     - 4096x2160  ~17:9
      5K-UW  - 5120x2160  ~21:9
      5K     - 5120x2880   16:9
      UHD-2  - 7680x4320   16:9

    Default resolution is HD.

  -s, --samples-per-pixel N
    Use N samples per pixel (SPP). Default is 50.

  -d, --trace-depth N
    Trace N rays per sample (RPS). A minimum value of 1 means just trace
    primary rays. Default is 16 or 50, depending on whether `rtwo´ was
    compiled for recursive or iterative ray tracing respectively.

  -a, --aspect-ratio <width>:<height>
    Set aspect ratio if not (implicitly) defined by -g, --geometry
    options. Default is 16:9 (HD).

  -v, --verbose
    Print processing details on stderr. Has impacts on performance.

  -t, --trace-sm
    Print FSM events and state changes to stderr (debugging).
    This option is not available in batch mode.

  -q, --quiet, --silent
    Omit result output on stdout. This option takes precedence over
    -A, --print-aov and -G, --print-guides options.

  -h, --help, --usage
    Print this help. Takes precedence over any other options

  -A, --print-aov <AOV>[,...]
    Print AOVs after image on stdout. Output format is PGM. Option not
    available in interactive mode.

    Available AOVs:
      RPP (Rays per pixel) - Sort of `data´ AOV (opposed to AOVs for lighting
                             or shading). Values add up total number of rays
                             traced for each pixel.

  -G, --print-guides
    Print denoiser guide layers before image on stdout. Available guide layers
    depend on denoiser type: SMP/ none, NRM/ normals, ALB/ albedos, NAA/ both,
    AOV/ both. Output format is PPM. Option not available in interactive mode.

  -S, --print-statistics
    Print statistical information on stderr.

  -D, --apply-denoiser <TYP>
    Enable denoiser in batch mode and apply type TYP after rendering with
    1 SPP. To enable for interactive mode see UI functions section below.
    Denoising in interactive mode applies to scene animation and while
    changing camera position and direction, as well as zooming.
    When finished there is a final still image rendered with SPP as given
    by -s, --samples-per-pixels or default.

    Available types for TYP:
      SMP - A simple type using OPTIX_DENOISER_MODEL_KIND_LDR. Feeds raw RGB
            rendering output into denoiser and retrieves result.
      NRM - Simple type plus hit point normals.
      ALB - Simple type plus albedo values for hit points.
      NAA - Simple type plus normals and albedos.
      AOV - The NAA type using OPTIX_DENOISER_MODEL_KIND_AOV. Might improve
            denoising result even if no AOVs provided

UI functions:
  window resize        - change viewport dimensions
  left button + move   - change camera position
  right button + move  - change camera direction
  right button + scoll - roll camera
  'a' key              - toggle scene animation
  'b' key              - enter blur mode (<ESC> to leave)
      scroll (wheel)   - change defocus blur
  'd' key              - enable/ disable denoising and select
                         type (loop)
  'f' key              - enter focus mode (<ESC> to leave)
      scroll (wheel)   - change aperture
  'z' key              - enter zoom mode (<ESC> to leave)
      scroll (wheel)   - zoom in and out
  <ESC> key            - leave RTWO
```

### Gallery
|    |    |
|:---|:---|
|1) The well-known image from RTOW chapter 12|2) AOV reflecting number of rays per pixel|
|![The well-known image from RTOW chapter 12](img/rtwo-0.png)|![AOV reflecting number of rays per pixel](img/rtwo-rpp.png)|
|3) SMP (simple) denoiser result|4) NRM denoiser with normals guide layer|
|![SMP (simple) denoiser result](img/rtwo_smp.png)|![NRM denoiser with normals guide layer](img/rtwo_nrm.png)|
|5) ALB denoiser with albedos guide layer|6) NAA denoiser with both<br>(all OPTIX_DENOISER_MODEL_KIND_LDR)|
|![ALB denoiser with albedos guide layer](img/rtwo_alb.png)|![NAA denoiser with both](img/rtwo_naa.png)|
|7) AOV denoiser with both (no AOV)<br>(OPTIX_DENOISER_MODEL_KIND_AOV)|8) Denoiser normals guide layer|
|![AOV denoiser with both](img/rtwo_aov.png)|![Denoiser normals guide layer](img/rtwo_A-0.png)|
|9) Denoiser albedos guide layer||
|![Denoiser albedos guide layer](img/rtwo_A-1.png)||

### Build
On your Linux box with NVIDIA Turing or Ampere graphics card install:
- Common development tools (gcc, g++, make ...
- CUDA 11.4
- OptiX 7.3
- ImageMagick 7

Installation details in Linux section of [../README.md](../README.md#linux)

- Download RTXplay repo (this) and cd into
- Run `make` in `optx` subfolder

If all went well there is `rtwo.png`.

- Start X server
- Run `DISPLAY=:0 ./rtwo` (interactive now)

### No Turing or Ampere
RTWO should work on older architectures as well, of course without RT Cores then. In `Makefile` set `-arch` flag in `NVCCFLAGS` variable according to archtecture in question ([list](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)) before compiling.

### No NVIDIA graphics card
Try a cloud service. RTWO developement actually happened on Windows (compilation only, no linking) and got its final cut on AWS. Any other provider should do as well. See [README.md](../README.md) in parent folder for setting up Windows and AWS. 

### Simple GPU workstation on AWS (VNC)
The faster RDP would have been a better approach but it's not compatible with Nvidia's GLX for Xorg (see [xrdp issues](https://github.com/neutrinolabs/xrdp/issues/721#issuecomment-293241800) *xorgxrdp driver not supporting Nvidia's GLX* [and](https://github.com/neutrinolabs/xrdp/issues/1550#issuecomment-614910727) *not seen that (Nvidia's GLX) work yet with xrdpdev*) which has been confirmed by tests with numerous configurations which all failed. Due to these obstacles and despite it is slow, falling back on VNC is considered ok because it works after all and it's only for testing anyway.

Steps below assume an [AWS EC2 G4 instance](https://aws.amazon.com/ec2/instance-types/g4/) (`g4dn.xlarge`) with [Amazon Linux 2 AMI](https://aws.amazon.com/amazon-linux-2/).

1. Install X
  ```
  # install server
  sudo yum install -y xorg-x11-server-Xorg

  # configure Xorg for NVIDIA
  sudo X -configure
  sudo mv /root/xorg.conf.new /etc/X11/xorg-nvidia.conf
  ```

2. Install LibVNC
  ```
  # clone libvncserver
  cd ~/lab ; git clone https://github.com/LibVNC/libvncserver.git ; cd libvncserver

  # build libvncserver
  mkdir build ; cd build ; cmake3 .. \
  	-DWITH_OPENSSL=ON \
  	-DWITH_GCRYPT=OFF \
  	-DWITH_GNUTLS=OFF ; make

  # install libvncserver
  sudo make install

  # prerequsites
  sudo yum install -y libXtst-devel openssl-devel

  # clone x11vnc
  cd ~/lab ; git clone https://github.com/LibVNC/x11vnc.git ; cd x11vnc

  # build x11vnc
  export LIBVNCSERVER_CFLAGS=-I/usr/local/include/rfb
  export LIBVNCCLIENT_CFLAGS=-I/usr/local/include/rfb
  export LIBVNCSERVER_LIBS="-L/usr/local/lib64 -lvncserver"
  export LIBVNCCLIENT_LIBS="-L/usr/local/lib64 -lvncclient"

  autoreconf -fiv
  ./configure ; make

  # install x11vnc
  sudo make install
  ```

3. Check configuration
  ```
  # start X server
  sudo X -ac -config /etc/X11/xorg-nvidia.conf &

  # start VNC server
  export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
  x11vnc -display :0 &

  # set password for `ec2-user´
  sudo passwd ec2-user
  ```

  [Download](https://www.realvnc.com/de/connect/download/viewer/windows/) and install VNC viewer

  Login as `ec2-user` with VNC viewer

  Run OptiX 7 course examples (within SSH session)
  ```
  cd ~/lab/optix7course/build

  DISPLAY=:0 ./ex03_testFrameInWindow
  ```

  Try iOS version of VNC viewer

### Performance cues

#### Low hanging fruits
- Use iterative instead of recursive raytracing

  Repo contains programs (`*.cu`) for both but `Makefile` defaults to *iterative*.
  ```
  # to try recursive
  make lclean
  CXXFLAGS=-DRECURSIVE make
  ```
- Use full optimization (`OPTIX_COMPILE_OPTIMIZATION_DEFAULT`) for module compilation
- Use `Frand48` (default) instead of CUDA `curand` RNG

  Code support both but `Makefile` defaults to `Frand48`.
  ```
  # try curand
  make lclean
  CXXFLAGS=-DCURAND make
  ```
- Try `OPTIX_GEOMETRY_FLAG_NONE` instead of `OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT` (`rtwo.cxx`, `*.cu`)

  OptiX documentation and develper forum recommend use of `OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT` even if there is no Anyhit Program. See [time series chart](https://docs.google.com/spreadsheets/d/1HAkMM2-QL5F1pwvjQTtyEAV1Zo2fghDp/edit?usp=sharing&ouid=100581696502355527803&rtpof=true&sd=true) for comparisons of various measures.

#### Kernel profiling with Nsight Compute
- Install Nsight Compute 2021.2 (Windows version)
- Nsight Compute CLI (`ncu`) is part od CUDA Toolkit
- Prepare module compile options in OptiX app (`rtwo.cxx`) and run NVCC with `-lineinfo` option (`Makefile`)
  ```
  OptixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO ; // or OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT
  ```

- Run `ncu` on OptiX app (mind [developer forum thread on kernel selection](https://forums.developer.nvidia.com/t/optix-and-performance-counter-reports-in-nsight-compute/180642))
  ```
  # ncu options:
  # -k regex:raygen    : RG kernel to profile (cannot HG, MS etc.)
  # --section regex:.* : record data for all sections (ncu -list-sections for all)
  # -f                 : force overwrite report file
  # -o rtwo            : report file (.ncu-rep auto-suffixed)

  ncu \
    -k regex:raygen    \
    --section regex:.* \
    -f                 \
    -o rtwo            \
    ./rtwo -q
  ```
- Analyse report in Nsight Compute GUI

### Compile OptiX 7 course
Steps below assume a working instance of the [RTXplay](https://github.com/otabuzzman/RTXplay) repo.

1. Install GLFW
  ```
  # prerequsites
  sudo yum install -y libXi-devel

  # clone GLFW
  cd ~/lab ; git clone https://github.com/glfw/glfw.git ; cd glfw

  # build GLFW
  mkdir build ; cd build ; cmake3 .. ; make

  # install GLFW
  sudo make install
  ```

2. Install OptiX 7 course tutorial
  ```
  # clone optix7course (fork)
  cd ~/lab ; git clone https://github.com/otabuzzman/optix7course.git ; cd optix7course

  # build optix7course
  mkdir build ; cd build ; OptiX_INSTALL_DIR=/usr/local/optix cmake3 .. ; make

  # check (no X required)
  ./ex01_helloOptix
  ./ex02_pipelineAndRayGen
  ```
