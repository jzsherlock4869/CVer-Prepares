# 目标检测相关思路梳理

最早的目标检测方法实际上就是图像分类方法，只是以ROI 中的区域代替整图进行，如R-CNN，过程为：选区域-CNN 提取特征-SVM 分类。然后，考虑到重复计算的冗余，SPP-net 提出，用空间金字塔池化，即自适应stride 的池化，保证任意大小都能得到定长特征向量。然后是Fast R-CNN，沿用了ROI pooling 的思路，并且加入了新的改进，即多任务学习，直接网络端到端回归bbox，并进行分类，去掉了SVM。到此为止，模型已经确定了，只是ROI 的方法还是利用Selective Search，导致预处理速度较慢，拖慢了整体的速度。因此，Faster R-CNN 引入了一个和检测网络共享部分权重的RPN 网络，用网络进行region proposal，从而提高了整体的效率，去掉了Selective Search。

上述的方法都是two-stage，即先给出候选框，然后在进行分类和微调位置。于是，另一种新的思路，即直接进行预测bbox 和类别，这就是one-stage 的方法。代表就是YOLO 模型，通过对图像划分成多个区块，直接输出一个和区块数相同的tensor，每个区块的预测向量里包含了其是否有目标，是什么类别，以及具体位置的修正。SSD 基于YOLO 的基本思路，考虑了多尺度的情况，将不同尺度的GT bbox 分配到不同尺度去预测，不同尺度的预测即通过不同层的feature map 进行。后续的YOLO v2 提出了对位置的直接预测方法，以及用k-means 聚类来手工得到先验框的方法，并利用darknet-19 作为backbone 网络。FPN 网络的基本思路是将不同尺度的feature map 通过网络连接的方式进行融合，并在每一层进行预测，从而可以利用不同尺度的信息。

RetinaNet 基本沿用了FPN 的思路，但是提出了focal loss 来处理样本比例不均衡的问题，对数量较少的正样本，以及难分样本给予较多的关注。Mask R-CNN 将实例分割与目标检测进行结合，融合了分类、回归、mask三个损失进行优化。YOLO v3 采用了具有residual 模块的darknet-53 作为backbone，并且用了binary logistic代替全局softmax，以及FPN 网络等trick，提升了性能。

