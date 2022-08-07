|第二届计图挑战赛

# Jittor 赛题一：风景图片生成赛题

## 简介

​		图像生成任务一直以来都是十分具有应用场景的计算机视觉任务，从语义分割图生成有意义、高质量的图片仍然存在诸多挑战，如保证生成图片的真实性、清晰程度、多样性、美观性等。

​		清华大学计算机系图形学实验室从Flickr官网收集了1万两千张高清（宽1024、高768）的风景图片，并制作了它们的语义分割图。其中，1万对图片被用来训练。训练数据集可以从[这里](https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1)下载。**其中 label 是值在 0~28 的灰度图，可以使用 matplotlib.pyplot.imshow 可视化。**下面展示了一组图片。

<img src="https://data.educoder.net/api/attachments/2796740?type=image/jpeg" alt="img" style="zoom: 33%;" /><img src="https://data.educoder.net/api/attachments/2796749?type=image/jpeg" alt="img" style="zoom: 33%;" />

标签包括29类物体，分别是  ：

`"mountain", "sky", "water", "sea", "rock", "tree", "earth", "hill", "river", "sand", "land", "building", "grass", "plant", "person", "boat", "waterfall", "wall", "pier", "path", "lake", "bridge", "field", "road", "railing", "fence", "ship", "house", "other"`



## 安装

### 运行环境

- python >= 3.7
- jittor >= 1.3.0
- ubuntu 20.04 LTS

### 数据集

可从简介中进行下载

### 训练

`python train`

### 评测指标

- mask accuary：根据用户生成的1000张图片，使用 SegFormer 模型[1]对图片进行分割，然后计算分割图和gt分割图的mask accuary=(gt_mask == pred_mask).sum() / (H * W)，确保生成的图片与输入的分割图相对应。mask accuary 越大越好，其数值范围是0~1。
- 美学评分：由深度学习美学评价模型为图片进行美学评分，大赛组委会参考论文 [2-4] 中的论文实现自动美学评分。该分数将归一化将到 0~1。
- FID（Frechet Inception Distance score）：计算生成的 1000 张图与训练图片的FID，该指标越小越好，将FID的100到0线性映射为 0 到 1。由于 baseline 代码的 FID 在 100 以内，所以 FID 大于 100 的将置为 100。

### 致谢

| 对参考的论文、开源库予以致谢，可选

此项目基于论文 *A Style-Based Generator Architecture for Generative Adversarial Networks* 实现，部分代码参考了 [jittor-gan](https://github.com/Jittor/gan-jittor)。



