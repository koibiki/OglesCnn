# OglesCnn
基于opengles的神经网络前向传播框架
测试apk中已包含cifar10 和 squeeze net 1000 分类模型
### benchmark (均运行Squeeze net)
| cpu        | ncnn(4线程)   |  OglesCnn  |
| --------   | -----:  | :----:  |
| 高通660     | 60ms |   89ms     |
| 麒麟970        |  57ms   |   77ms   |
| 高通835       |   - |  -  |  
