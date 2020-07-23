原项目地址:https://github.com/jfzhang95/pytorch-video-recognition



### 使用方式

1. 下载代码
2. 新建path文件夹(使用过程可能无法自动新建相应路径,可根据代码自己手动建立)
3. 下载chechpoint  https://github.com/liuzhaoo/C3D/releases
4. 运行inference.py,注意选择自己的视频,以及加载正确的权重路径



### 训练

1. 下载ucf101数据集
2. 按照path.py设置正确路径
3. 运行dataset.py,对数据集进行预处理,也要注意文件的路径
4. 运行train.py 进行训练



- 注意: 本项目在原作者项目上进行修改,添加了注释,供学习交流使用.