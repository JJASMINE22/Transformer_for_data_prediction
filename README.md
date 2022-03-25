## PyTorch实现Parallel data prediction based on Transformer
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Cautions](#注意事项)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤)
6. [预测效果 predict](#位置编码)
7. [参考资料 Reference](#参考资料)

## 所需环境
1. Python3.7
2. PyTorch>=1.7.0+cu110
3. numpy==1.19.5
4. pandas==1.2.4
5. pyod==0.9.8
6. CUDA 11.0+  

## 模型结构
Transformer  
![image]()

## 注意事项
1. 时序数据推理，删除了标准Transformer的位置掩码、位置编码、前馈层等机制
2. 使用一个正态分布变量替代起始符嵌入特征
3. 训练时，并行推理解码序列；预测时，贯续推理解码序列
4. 与标准Transformer不同，推理时无需设置起始符、组合推理结果
5. 修改MultiHeadAttention中的通道拆分、合并方式
6. 保留三角掩码，防止特征泄露
7. 加入权重正则化操作，防止过拟合

## 文件下载    
链接：https://pan.baidu.com/s/13T1Qs4NZL8NS4yoxCi-Qyw 
提取码：sets 
下载解压后放置于config.py中设置的路径即可。

## 训练步骤
运行train.py即可开始训练。  

## 预测效果
sequence_1  
![image]()  

sequence_2  
![image]() 

sequence_3  
![image]()  

## 参考资料
1. https://arxiv.org/pdf/1706.03762.pdf  
2. https://blog.csdn.net/qq_44766883/article/details/112008655

