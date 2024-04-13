#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/io/pcd_io.h>//pcl的输入输出，这个肯定得写的
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>//包含pcl中定义的不同类型的点的数据结构，xyz等，如果坐标转换，这个肯定也得写
#include <pcl/common/transforms.h>
#include <string>

using namespace std;

typedef pcl::PointXYZI PointT;//类型转换，为了更加简易

struct point_type
{
    cv::Point p_pixel;
    float range;
    float x,y,z;
    float intensity;
};
//OpenCV中用于表示二维点的类。它包含两个成员变量，分别代表点的x和y坐标。这里就是用来表他的像素


int main(int argc ,char *argv[])//argc代表命令行参数的数量，argv用于存储每个命令行参数的字符串
{
    if (argc!= 8)//当然，这个8代表参数的个数，这是根据自己来定的
    {
        cout << "请输入正确命令格式" << endl;
        return -1;//表示程序失败，返回一个错误码
    }

    string flag_way = argv[3];//这个位置代表的是上色方案
    float x_ = stof(argv[4]);//stof是将命令行上的转化为单精度浮点数
    float y_ = stof(argv[5]);
    float u_ = stof(argv[6]);
    float v_ = stof(argv[7]);

    cv::Mat image = cv::imread(argv[1],cv::IMREAD_COLOR);
    //mat可以用来保存照片的一些信息，用来输出照片的像素，确保照片质量正确
    //OPENCV4.0中的改版，使用cv::IMREAD_COLOR来对颜色图像进行提取
    cout << "image_rows" << image.rows << endl;
    cout << "image_cols" << image.cols << endl;
    //输出命令，mat中自带的内部类

    
    pcl::PointCloud<PointT>::Ptr cloud_in (new pcl::PointCloud<PointT>());
    //创建了一个名为 cloud_in 的智能指针，指向一个空的 pcl::PointCloud<PointT> 类型的点云数据
    //::Ptr：是指向点云数据的智能指针的一种约定命名方式。智能指针可以自动管理内存的生命周期，当不再需要时自动释放内存，避免内存泄漏
    pcl::io::loadPCDFile<PointT>(argv[2], *cloud_in);

    //这是在命令行的方式进行图像与点云的读取

    //那么在雷达坐标系下，进行点云滤波与点云显示处理
    pcl::PointCloud<PointT>::Ptr cloud_filter (new pcl::PointCloud<PointT>());
    //同上一样，这里设置的是滤波完成后的点云
    for (const auto& p:(*cloud_in))
    {
        if(p.x<=0)
        {
            continue;//跳过当前点，继续处理下一个点
        }
        cloud_filter->points.push_back(p);//用于将点云数据中满足条件的点 p 添加到 cloud_filter->points 这个容器中
    }
    //这里，设立的点云筛选条件是根据x轴来的，也就是，点在x轴正向的，保留，存入filter，否则剔除
    //当然，改改也能作为一种距离滤波用


    //经过滤波之后的点云，我们将其坐标系转换到相机坐标系
    Eigen::Matrix3d R;
    R << 0, -1,  0,
        -1,  0,  0,
         0,  -1,  0;//自己的外参旋转矩阵啊，为自己瞎写了一个

    Eigen::Vector3d t;
    t << 0, 0,  0;


    Eigen::Matrix4d transformed;
    transformed.block<3,3>(0,0) = R;
    transformed.topRightCorner(3,1) = t;
    Eigen::Matrix4d transformed_transpose = transformed.transpose();//这里是组成最终的外参矩阵然后转置
    //原因就是本来外参是相机到点云，但我们想将点云投到图像上，所以有这一部操作
    cout << transformed.matrix() << endl;//打印信息哈


    //建立转置后的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*cloud_filter, *cloud_transformed,transformed_transpose);
    //内置的函数，主要是用来点云转换与存储的，各参数：
    //*cloud_filter 是指向原始点云数据的指针，它包含了需要进行坐标变换的点云。
    
    cout << "相机坐标系下的点云数目为：" << cloud_transformed->points.size()<< endl;


    //接下来设计图片的各参数
    Eigen::Matrix3d K;
    K << 1.0*x_,  0.0   ,   1.0+u_,
         0.0,     1.0*y_,   1.0+v_,
         0.0,     0.0,      1.0; 
         //这里阿，主要就是我们加了一个系数，可以通过命令行来进行调整，毕竟由于点云噪声的原因，效果可能不会达到最好
         //主要就是一个优化

    double k1 = 1.0,    k2 = 1.0,   k3 = 1.0;
    double p1 = 1.0,    p2 = 1.0;
    //这里也是乱给的哈
    
    cv::Mat image_lidar(720,1280,CV_8UC3);//就是一个图像，后面那个是8位无符号整数、3通道的图像。这是在OpenCV中表示图像的常见方式
    float max_intensity = -200,    min_intensity = 1000;
    float max_range = 0,    min_range = 1000;
    float max_x = 0,min_x = 100;
    float max_y = 0,min_y = 100;
    float max_z = 0,min_z = 100;

    vector<point_type> pixel_sum;

    for(const auto &p:  (*cloud_transformed))//循环，通过p去访问指针指向的元素，后面可以直接通过p去访问，auto是自动推断迭代器类型
    {
        Eigen::Vector3d p_normalizated (p.x/p.z,    p.y/p.z ,1);//接下来是归一化过程，主要是用来去激光点畸变的

        double x = p_normalizated[0];
        double y = p_normalizated[1];
        double r = x*x +y*y;
        Eigen::Vector3d p_undisted(x * (1 + k1 * r + k2 * r * r + k3 * r * r * r) + 2 * p1 * x * y + p2 * (r + 2 * x * x),
                                   y * (1 + k1 * r + k2 * r * r + k3 * r * r * r) + p1 * (r + 2 * y * y) + 2 * p2 * x * y,
                                   1.0);//套公式的

        Eigen::Vector3d p_projected = K * p_undisted;//完成去畸变任务后，建立新的变量去赋值
        cv::Point p_pixel;
        p_pixel.x = p_projected[0];
        p_pixel.y = p_projected[1];

        if(p_pixel.x <= image.cols && p_pixel.y <= image.rows)
        {
            point_type point_tmp;
            point_tmp.x=p.x;//这里的p是for循环内转换后的点
            point_tmp.y = p.y;
            point_tmp.z= p.z;
            point_tmp.intensity = p.intensity;
            float range = sqrt(p.x * p.x  +p.y  *   p.y + p.z   *p  .z );//计算到原点的距离，确定点的远近
            point_tmp.range = range;
            point_tmp.p_pixel = p_pixel;
            pixel_sum.push_back(point_tmp);//将一个 point_type 结构体对象 point_tmp 添加到了名为 pixel_sum 的向量中

            if(p.x>=max_x)  max_x = p.x;
            if(p.x<= min_x) min_x = p.x;
            if(p.y>=max_y)  max_y = p.y;
            if(p.y<= min_y) min_y = p.y;
            if(p.z>=max_z) max_z = p.z;
            if(p.z<=min_z) min_z = p.z;
            if(p.intensity >= max_intensity) max_intensity = p.intensity;
            if(p.intensity <= min_intensity) min_intensity = p.intensity;
            if(range>=max_range) max_range = range;
            if(range<=min_range) min_range = range;
            //着一块是替换最值的，我感觉不加的话，着色方案会出现变化，不过我没尝试
        }

    }

        cout<<"投影在图像上的点云数目为:"<<pixel_sum.size()<<endl;
        cout<<"ｘ的范围"<<min_x<<"~"<<max_x<<endl;
        cout<<"ｙ的范围"<<min_y<<"~"<<max_y<<endl;
        cout<<"ｚ的范围"<<min_z<<"~"<<max_z<<endl;
        cout<<"intensity的范围"<<min_intensity<<"~"<<max_intensity<<endl;
        cout<<"range的范围"<<min_range<<"~"<<max_range<<endl;

        //接下来对提取的点进行处理，这里就是要上色了
        for(const auto &p : pixel_sum)
        {
            int cur_val;
            if(flag_way == "intensity")//这里确保你在命令行输入的是intensity这种着色方案
            {
                float val = p.intensity;
                float minval = min_intensity;
                float maxval = max_intensity;
                cur_val = (int)(255*(val-minval)/(maxval-minval));// 缩放数值，便于处理，你也可以不这么干，影响颜色罢了
            }

            //接下来的注释，是提供多种选择，你可以根据需求来选用什么，这个无所谓

        // else if (flag_way == "range")
        // {
        //     float val = p.range;
        //     float minVal = min_range;
        //     float maxVal = max_range;
        //     cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        // }
        // else if (flag_way == "x")
        // {
        //     float val = p.x;
        //     float minVal = min_x;
        //     float maxVal = max_x;
        //     cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        // }
        // else if (flag_way == "y")
        // {
        //     float val = p.y;
        //     float minVal = min_y;
        //     float maxVal = max_y;
        //     cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        // }
        // else if (flag_way == "z")
        // {
        //     float val = p.z;
        //     float minVal = min_z;
        //     float maxVal = max_z;
        //     cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        // }
        else
        {
            std::cout << "===========" << std::endl;
            std::cout << "字段输入错误" << std::endl;
            return -1;
        }

                //正式上色：这里的数值都随意改的，就看你想要什么颜色了

        int red, green, blue = 0;
        if(cur_val <= 51)
        {
            blue = 255;
            green = cur_val * 5;
            red = 0;
        }
        else if(cur_val <=102)
        {
            cur_val -= 51;
            blue = 255-cur_val *5;
            green = 255;
            red = 0;
        }
            else if (cur_val <= 153)
			{
				cur_val -= 102;
				blue = 0;
				green = 255;
				red = cur_val * 5;
			}
			else if (cur_val <= 204)
			{
				cur_val -= 153;
				blue = 0;
				green= 255 - static_cast<unsigned char>(cur_val * 128.0 / 51 + 0.5);
				red= 255;
			}
			else if (cur_val <= 255)
			{
				cur_val -= 204;
				blue = 0;
				green = 127 - static_cast<unsigned char>(cur_val * 127.0 / 51 + 0.5);
				red= 255;
			}
        //上色结束

        cv::circle(image_lidar,p.p_pixel,2,cv::Scalar(red,green,blue),-1);//就是限定了一个圆形的投影范围
        //想要全部投影或者改为其他形状，也都可以
        }

        //投影结束，显示图片
        cv::Mat image_result;
        cv::addWeighted(image_lidar , 0.6 , image , 0.4 , 0 , image_result);//设置那种方式更加突出，将其放在image_result
        cv::imshow("image_result",image_result);    
        cv::imwrite("result.png",image_result);//可加可不加，不加截图就行
        cv::waitKey(0);



    system("pause");
    return 0;
}