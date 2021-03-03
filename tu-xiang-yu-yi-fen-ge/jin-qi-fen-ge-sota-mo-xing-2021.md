---
description: 近期顶会论文语义分割方向
---

# 近期分割SOTA模型（2021）

MMSegmentation中已经支持的几种较新近的语义分割模型

## 近期语义分割模型

### HRNet

**Deep High-Resolution Representation Learning for Visual Recognition**

HRNet的文章。创新点：基本结构采用并联降采样。

![](../.gitbook/assets/tu-pian-%20%2810%29.png)

3个compoents：parallel multi-res，multi-res fusion，representation head。

多尺度fusion时的操作：

![](../.gitbook/assets/tu-pian-%20%2822%29.png)

Representation head的三种形式：直接取出最大尺寸、进行多尺度的concate、形成一个feature pyramid。

![](../.gitbook/assets/tu-pian-%20%2811%29.png)

作者对hrnet中的parallel卷积和multi-res fusion进行了分析。首先，parallel 卷积可以看出是一个各个通道之间的分辨率不一样（尺寸不同）的一个feature map的分组卷积，即只跟自己的尺寸一样的部分进行卷积，按尺寸分组。那么，分组卷积有1x1xc的kernel来打通各个通道之间的信息，而hrnet中的fusion正好可以看做这样的操作。Fusion过程类似正常的卷积。或者说，正常的卷积可以表示成fusion的形式，如（c）。

所以，hrnet的并联和融合，可以看做是不同尺寸的多通道的分组卷积，或者说分组卷积在不同尺寸多通道feature map上的拓展。

### CGNet

**CGNet: A Light-weight Context Guided Network for Semantic Segmentation**

Tianyi Wu1;2, Sheng Tang1;, Rui Zhang1;2, Yongdong Zhang1

1Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.

2 University of Chinese Academy of Sciences, Beijing, China.

CGnet的基本思路：CG：context guided。将surrounding context和global context的信息作为约束，构建CG block单元，用于分割。由于先验参考信息更强，因此可以降低参数量。

![](../.gitbook/assets/tu-pian-%20%2813%29.png)

CG block的基本结构：首先，分别提取local feature和surrounding feature，然后两者concate起来，经过BN+pReLu后，进行GAP，得到的向量作为权重，对feature map各个通道进行加权，作为glocal feature，最后输出。GAP做权重的思路就是SENet的思路。

另外，还采用了residual learning来提高梯度bp的效果。两种residual的方式：

![](../.gitbook/assets/tu-pian-%20%2816%29.png)

整体结构：

![](../.gitbook/assets/tu-pian-%20%284%29.png)

其中，整个结构分为三个stage，每个stage降采样一次，第一个stage就是conv，第2，3个stage是CG block的堆叠。最终做一次upsample。并且，每个stage的输入都是上一个stage的输出和输出的组合。为了进一步减小参数量，还采用了channel-wise conv。但是和常规的channel-wise做完后在加一个1x1 conv来打通各个channel的操作不同，这里为了保持通道之间的独立性，没有做打通。（因为loc和sur所表示的feature含义本身就不一样。 ）

## **DANet**

**Dual Attention Network for Scene Segmentation**

Jun Fu 1;3 Jing Liu\* 1 Haijie Tian 1 Yong Li 2 Yongjun Bao 2 Zhiwei Fang 1;3 Hanqing Lu 1

1National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences 2Business Growth BU, JD.com 3 University of Chinese Academy of Sciences

DANet的文章。主要创新点：

TBC

