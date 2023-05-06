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
- 1、（todo）基本几何对象的显示，点线面
- 2、（todo）几何算法开发，相交、缓冲等。


## mqtt开发