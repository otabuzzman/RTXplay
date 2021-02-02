# RTXplay
A lab for playing around with NVIDIA's realtime raytracing concept. The lab intends to migrate the raytracer presented by RTOW to OptiX Ray Tracing Engine and thus utilizing the new RT cores which had been introduced by Turing microarchitecture.

### Findings
1. Extracting optixTriangle from OptiX SDK samples. Code copied from `cuda/CUDAOutputBuffer.h` contained `cudaFree ( <d_pointer> ) ; cudaMalloc ( <d_pointer> ) ;`. Original code worked fine, copied code produced random background, thus apparently didn't execute miss program. Removing `cudaFree( <d_pointer> )` fixed it. Using `cudaFree` with a device pointer not allocated before is not allowed according to CUDA API reference. Question is why this worked in original CUDAOutputBuffer class implementation.
2. Curly braces around `params` declaration in `shader_optixTriangle.cu`. Removing braces [shall yield same scope](https://vladonsoftware.wordpress.com/2019/02/25/c-when-the-curly-braces-really-matter/) (section *What happened*) but produces warning. Avoid with NVCC option `-rdc true` to [force relocatable device code](https://forums.developer.nvidia.com/t/warning-extern-declaration-of-the-entity-xxxx-is-treated-as-a-static-definition/69887) generation.
3. [Simple(r) implementation](https://zeux.io/2010/12/14/quantizing-floats/) to quantize `float` into `unsigned char`.
4. OPTIX_CHECK_LOG macro in `sutil/Exception.h` depends on variable names `log` and `sizeof_log` which thus must not change.
5. Changing shader sources requires cache file removal before starting application. Otherwise errors are likely.

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
6. Merge new feature branch
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

NVIDIA
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [CUDA Toolkit Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [OptiX Ray Tracing Engine](https://developer.nvidia.com/optix). Links to further resources at page bottom.
- [Optix Download](https://developer.nvidia.com/designworks/optix/download). Requires NVIDIA developer program membership (free).
- [OptiX learning path](https://forums.developer.nvidia.com/t/tutorials-webcasts/30022) recommendation

Issues
- Problems with SSH keys on Ubuntu Server 18.04 LTS AMI, probably due to [change in OpenSSH](https://sjsadowski.com/invalid-format-ssh-key/). [Instructions](https://aws.amazon.com/de/premiumsupport/knowledge-center/user-data-replace-key-pair-ec2/) on how to replace keys. Must attach root volume afterwards to a working instance and configure ec2-user in /etc/sudoers for sudo without password.
