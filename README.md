# RTXplay
A lab for playing around with NVIDIA's realtime raytracing concept. The lab intends to migrate the raytracer presented by RTOW to OptiX Ray Tracing Engine and thus utilizing the new RT cores which had been introduced by Turing microarchitecture.

### Findings
- Extracting optixTriangle from OptiX SDK samples. Code copied from `cuda/CUDAOutputBuffer.h` contained `cudaFree ( <d_pointer> ) ; cudaMalloc ( <d_pointer> ) ;`. Original code worked fine, copied code produced random background, thus apparently didn't execute miss program. Removing `cudaFree( <d_pointer> )` fixed it. Using `cudaFree` with a device pointer not allocated before is not allowed according to CUDA API reference. Question is why this worked in original CUDAOutputBuffer class implementation.

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

AWS
- Information on [G4 instances](https://aws.amazon.com/de/ec2/instance-types/g4/) (g4dn.xlarge)
- Manually [installing NVIDIA drivers on Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#public-nvidia-driver)
- Ubuntu 18.04 VMI with [NVIDIA Quadro Virtual Workstation](https://aws.amazon.com/marketplace/pp/B07YV3B14W?qid=1607366456238&sr=0-3&ref_=srh_res_product_title)
- How-to [connect to your Linux instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)

NVIDIA
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [CUDA Toolkit Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [OptiX Ray Tracing Engine](https://developer.nvidia.com/optix). Links to further resources at page bottom.
- [Optix Download](https://developer.nvidia.com/designworks/optix/download). Requires NVIDIA developer program membership (free).
- [OptiX learning path](https://forums.developer.nvidia.com/t/tutorials-webcasts/30022) recommendation

Issues
- Problems with SSH keys on Ubuntu Server 18.04 LTS AMI, probably due to [change in OpenSSH](https://sjsadowski.com/invalid-format-ssh-key/). [Instructions](https://aws.amazon.com/de/premiumsupport/knowledge-center/user-data-replace-key-pair-ec2/) on how to replace keys. Must attach root volume afterwards to a working instance and configure ec2-user in /etc/sudoers for sudo without password.
