# OglesCnn
基于opengles的神经网络前向传播框架
测试apk中已包含cifar10 和 squeeze net 1000 分类模型
### benchmark (均运行Squeeze net)
| cpu        | ncnn(4线程)   |  OglesCnn  |
| --------   | -----:  | :----:  |
| 高通660     | 60ms |   70ms     |
| 高通710       |   - |  35ms  |  

TO DO LIST:

1.目前在ARM MAIL GPU上仍有bug，待修复

2.winograd conv 加载kernel的逻辑还未添加

3.通用的python 模型转换工具待添加

4.模型文件解析工具待添加
