# my_reproject
手写了一遍点云重投影至像素坐标系操作  
这里首先感谢我亲爱的斌哥，没有斌哥我就写不出来这玩意  
opencv是4.0，pcl库用的1.10，如果版本不一样还请修改cmakelist文件  
操作方法就是在可执行文件的终端命令行输入：  
./可执行文件 /图像所在位置 /点云所在位置 /选择投影方式 选择微修系数  
比如说：  
./my_projection /home/liu/once/22.png /home/liu/once/22.pcd intensity 1 1 0 0  
由于亲手从零敲的，里面做了很多注释，想要修改的话应该也很轻松  
着色方案有些单调了，不过这个都可以改  
