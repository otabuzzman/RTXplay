# RTXplay
A lab for playing around with NVIDIA's realtime raytracing concept. The lab intends to migrate the raytracer presented by RTOW to OptiX Ray Tracing Engine and thus utilizing RT and Tensor cores. RT cores had been introduced by Turing, Tensor cores by the predeceasing Volta microarchitecture.

### .cu file processing
1. compile .cu to PTX file using nvcc
2. convert PTX to intermediary .c file using bin2c from CUDA
3. compile .c to intermediary .o file
4. link application with .o file

### Links
RTOW
- The RTOW books on [GitHub](https://github.com/RayTracing/raytracing.github.io) and [Amazon](https://www.amazon.de/gp/product/B0785N5QTC/ref=series_rw_dp_sw)
- The 1st RTOW book on [GitHub](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

Math
- [A refreshment on quadratic equations](http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/)

Tools
- Handy summary of [class and sequence diagram notation](http://umich.edu/~eecs381/handouts/UMLNotationSummary.pdf)

AWS
- Information on [G4 instances](https://aws.amazon.com/de/ec2/instance-types/g4/) (g4dn.xlarge)
- Manually [installing NVIDIA drivers on Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#preinstalled-nvidia-driver)
- Ubuntu 18.04 VMI with [NVIDIA Quadro Virtual Workstation](https://aws.amazon.com/marketplace/pp/B07YV3B14W?qid=1607366456238&sr=0-3&ref_=srh_res_product_title)
- How-to [connect to your Linux instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)

NVIDIA
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [OptiX Ray Tracing Engine]https://developer.nvidia.com/optix). Links to further resources at page bottom.
- [Optix Download](https://developer.nvidia.com/designworks/optix/download). Requires NVIDIA developer program membership (free).
- [OptiX learning path](https://forums.developer.nvidia.com/t/tutorials-webcasts/30022) recommendation
