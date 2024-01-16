# 学习用rust编写命令行程序
# vulkano 是rust的可视化库，类似于opengl

- 1、src/study用来学习rust基础
- 2、src/conf配置参数
- 3、src/DGIS地理信息算法学习
- 4、src/mqtt mqtt学习
- 5、src/vulkan_CV 计算机图形学学习

## rust模块组织
- 1、在src下创建新的文件夹，每个文件夹作为一个模块集合，每个文件夹下可以包含子文件夹（子文件夹中创建子模块）；然后在每个文件及中创建mod.rs文件，在该文件中声明当前文件夹下的模块。
- 2、模块的引用，在main.rs中，首先声明使用哪个模块
  ```rust
  mod xxx;
  use xxx;
  use xxx::yyy;
  ```


## DGIS开发
GIS系统开发基于WebGPU进行渲染，GDAL以及geos进行栅格和矢量操作。
- 1、（todo）基本几何对象的显示，点线面。点线面简单几何的定义。
- 2、（todo）几何算法开发，相交、缓冲等。


## mqtt开发


## WebGPU
webGPU是一套基于浏览器的图形API，浏览器封装了现代图形API（Dx12、Vulkan、Metal），提供给Web 3D程序员，为 Web释放了更多的GPU 硬件的功能。wgpu作为rust库可以通过rust开发WebGPU程序,wgpu在桌面端采用的是Vulkan、metal以及driectX 12或者OpenGL ES，在浏览器端采用的WebGPU或WebGL2.因此通过rust开发wgpu程序即可在桌面端运行也可通过打包为webassembly在浏览器端运行。
- 1、buffer，是GPU中的二进制数据，buffer是内存中存储连续的。buffer通常用来存储结构体以及数组。WebGPU中常用的buffer为顶点buffer以及index buffer。