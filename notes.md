### Branches
- `main` - main development branch (default)
- `movie` - render multiple images to compose clip
- `frand48` - replace for NVIDIA's curand RNG
- `hitcorr` - calculate real intersection point on sphere
- `objout` - provide triangle mesh inspection support

### Compile OptiX 7 course
Steps below assume a working instance of the [RTXplay](https://github.com/otabuzzman/RTXplay) repository.

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
  sudo X -config /etc/X11/xorg-nvidia.conf &

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

### GPU workstation on AWS (CAS)
CAS (Cloud Access Software) and PCoIP from Teradici allow for fast access of remote workstations via public networks.

#### Manual setup on Amazon AMI
[AWS blog post](https://aws.amazon.com/blogs/compute/building-a-gpu-workstation-for-visual-effects-with-aws/) (2018) on how to perform a manual setup. The article gives a good overview and comprehensive hints on steps and proper sequence to set up a remote workstation that utilizes Teradici's trial version.

Steps below assume an [AWS EC2 G4 instance](https://aws.amazon.com/ec2/instance-types/g4/) (`g4dn.xlarge`) with [Amazon CentOS AMI](https://aws.amazon.com/mp/linux/#centos).

#### Setup on Teradici CAS AMI
Steps below assume an [AWS EC2 G4 instance](https://aws.amazon.com/ec2/instance-types/g4/) (`g4dn.xlarge`) with [Teradici Cloud Access Software for CentOS 7 AMI](https://aws.amazon.com/marketplace/pp/Teradici-Teradici-Cloud-Access-Software-for-CentOS/B07CT11PCQ).



### Findings
1. Extracting optixTriangle from OptiX SDK samples. Code copied from `cuda/CUDAOutputBuffer.h` contained `cudaFree ( <d_pointer> ) ; cudaMalloc ( <d_pointer> ) ;`. Original code worked fine, copied code produced random background, thus apparently didn't execute miss program. Removing `cudaFree( <d_pointer> )` fixed it. Using `cudaFree` with a device pointer not allocated before is not allowed according to CUDA API reference. Question is why this worked in original CUDAOutputBuffer class implementation.
2. Curly braces around `params` declaration in `shader_optixTriangle.cu`. Removing braces [shall yield same scope](https://vladonsoftware.wordpress.com/2019/02/25/c-when-the-curly-braces-really-matter/) (section *What happened*) but produces warning. Avoid with NVCC option `-rdc true` to [force relocatable device code](https://forums.developer.nvidia.com/t/warning-extern-declaration-of-the-entity-xxxx-is-treated-as-a-static-definition/69887) generation.
3. [Simple(r) implementation](https://zeux.io/2010/12/14/quantizing-floats/) to quantize `float` into `unsigned char`.
4. OPTIX_CHECK_LOG macro in `sutil/Exception.h` depends on variable names `log` and `sizeof_log` which thus must not change.
5. Changing shader sources requires cache file removal before starting application. Otherwise errors are likely.
6. Code in a PTX file must contain any referenced objects (or variables). Calling a class member function from inside a shader (kernel) expects the class in question to be defined in the same .cu file (e.g. by including a header file containing the definition) or by a further .cu file given to NVCC on the command line when compiling to PTX.
7. [Q&A on NVIDIA developer forum](https://forums.developer.nvidia.com/t/intersection-point/81612/7) on how to get a hit primitive's vertices in a closest hit shader. Using `optixGetGASTraversableHandle()` and related [might be bad for performance](https://raytracing-docs.nvidia.com/optix7/guide/index.html#device_side_functions#vertex-random-access). Passing device pointers pointing at primitive vertices and indices of `OptixBuildInput` objects (the *Things*) via SBT records thus recommended.
8. [Front face in OptiX](https://forums.developer.nvidia.com/t/optix-triangle-hit-face/83511) is counter-clockwise in right-handed coordinate system (missing in OptiX documentation).

### Git for short (copy&paste)

#### Branching on new features

1. Before start working on new feature
```
git checkout -b <branch>
```
2. While working on new feature
```
git add -A # every now and then
```
3. After working on new feature
```
git add -A
git commit -m <comment>
git push -u origin <branch>
```
4. Check new feature branch
```
git clone https://github.com/otabuzzman/RTXplay.git
git checkout <branch>
```
5. Merge new feature branch
```
git checkout main
git merge <branch>
git push -u origin main
```
6. Remove new feature branch
```
# delete remote branch on Git site
git branch -d <branch>  # local remove
git remote prune origin # clean tracking
```

### Links
RTOW
- The RTOW books on [GitHub](https://github.com/RayTracing/raytracing.github.io) and [Amazon](https://www.amazon.de/gp/product/B0785N5QTC/ref=series_rw_dp_sw)
- The 1st RTOW book on [GitHub](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

Math
- [Refreshment on quadratic equations](http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/)
- Sample C++ implementation of [sphere creation algorithm](http://paulbourke.net/geometry/platonic/sphere.cpp) based on [icosahedron](https://rechneronline.de/pi/icosahedron.php) subdivision.

Tools
- Handy summary of [class and sequence diagram notation](http://umich.edu/~eecs381/handouts/UMLNotationSummary.pdf)
- Web page on [running CMake](https://cmake.org/runningcmake/)
- [SO answer](https://stackoverflow.com/questions/6127328/how-can-i-delete-all-git-branches-which-have-been-merged?answertab=active#tab-top) on removing merged branches

AWS
- Information on [G4 instances](https://aws.amazon.com/de/ec2/instance-types/g4/) (g4dn.xlarge)
- Manually [installing NVIDIA drivers on Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#public-nvidia-driver)
- Ubuntu 18.04 VMI with [NVIDIA Quadro Virtual Workstation](https://aws.amazon.com/marketplace/pp/B07YV3B14W?qid=1607366456238&sr=0-3&ref_=srh_res_product_title)
- How to [connect to your Linux instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)

CUDA
- How to [share C++ classes between host and and device](https://stackoverflow.com/questions/39006348/accessing-class-data-members-from-within-cuda-kernel-how-to-design-proper-host) (kernel code

OptiX
- Developer forum QA on [subpixel jittering and progressive rendering antialiasing techniques](https://forums.developer.nvidia.com/t/anti-aliased-image/48255)

NVIDIA
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [CUDA Toolkit Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [OptiX Ray Tracing Engine](https://developer.nvidia.com/optix). Links to further resources at page bottom.
- [Optix Download](https://developer.nvidia.com/designworks/optix/download). Requires NVIDIA developer program membership (free).
- [OptiX learning path](https://forums.developer.nvidia.com/t/tutorials-webcasts/30022) recommendation

Issues
- Problems with SSH keys on Ubuntu Server 18.04 LTS AMI, probably due to [change in OpenSSH](https://sjsadowski.com/invalid-format-ssh-key/). [Instructions](https://aws.amazon.com/de/premiumsupport/knowledge-center/user-data-replace-key-pair-ec2/) on how to replace keys. Must attach root volume afterwards to a working instance and configure ec2-user in /etc/sudoers for sudo without password.
