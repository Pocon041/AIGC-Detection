## 一、项目背景精简版

当前 AIGC 图像检测方法通常直接将任务建模为二分类问题：

[
x \rightarrow \text{Real / AIGC}
]

但这种做法容易学到**语义捷径**，例如某些生成图像更常出现人像、插画、特定构图或特定颜色风格。模型表面上是在检测 AIGC，本质上可能是在利用数据集偏差，而不是捕捉生成图像真正的底层痕迹。

然而，AIGC 图像与自然图像的关键差异往往不只体现在语义内容，而更多隐藏在低层视觉统计中，例如：

* 高频纹理；
* 去噪残留；
* 局部噪声分布；
* 压缩与重建伪影；
* 生成模型造成的微结构异常；
* 自然摄影图像中才稳定存在的底层统计规律。

因此，本项目希望将 AIGC 检测从普通语义分类问题，转化为：

> **低层视觉隐变量分布建模问题。**

也就是说，我们不直接依赖图像语义判断真假，而是学习图像底层特征在隐空间中的分布差异：真实图像的低层隐变量应更接近自然图像分布，而 AIGC 图像的低层隐变量应表现出更高的分布偏离。

---

## 二、最终版本核心思想

最终方案不再显式划分：

[
z_c = \text{语义 latent}, \quad z_r = \text{噪声 latent}
]

因为语义、纹理、材质和低层统计之间并不是严格可分的。很多纹理本身就是语义的一部分，例如草地纹理、毛发纹理、花瓣纹理。

所以最终方案改为：

[
x \rightarrow \phi_{\text{low}}(x) \rightarrow q_\psi(z|u) \rightarrow z
]

其中：

* (x)：输入图像；
* (\phi_{\text{low}})：类似 *Beyond Generation* 的低层特征提取器；
* (u=\phi_{\text{low}}(x))：低层 forensic feature；
* (q_\psi(z|u))：隐变量编码器；
* (z)：低层判别隐变量；
* (z) 不被强行定义成“语义”或“噪声”，而是通过训练自动学习成对 AIGC 检测最有用的低层隐因子。

---

## 三、最终方案结构

整体结构可以概括为：

```text
图像 x
  ↓
低层特征提取器 φ_low
  ↓
低层特征 u
  ↓
隐变量编码器 qψ(z|u)
  ↓
低层隐变量 z
  ↓
AIGC / Natural 分类器
  ↓
检测结果
```

同时可以加入一个轻量 decoder：

```text
z → D(z) → 重建低层特征 û
```

注意：这里 decoder **不重建 RGB 图像**，而是重建低层特征 (u)。

原因是：

* 重建 RGB 图像会引入大量语义压力；
* 重建低层特征更符合 AIGC 检测目标；
* decoder 只是辅助约束，不是主任务。

---

## 四、DDA-COCO 的作用

DDA-COCO 不再用于显式学习一个语义 latent，而是用于提供**语义高度对齐的真实图像 / AIGC 图像 pair**。

对于一对样本：

[
(x_i^{real}, x_i^{aigc})
]

它们语义内容高度接近，因此可以认为语义因素已经在数据层面被控制住。

模型只需要在低层隐空间中学习：

[
E(z_i^{real}) < E(z_i^{aigc})
]

也就是：

> 在相同语义内容下，真实图像的低层隐变量更接近自然图像分布，AIGC 图像的低层隐变量更偏离自然图像分布。

---

## 五、核心 Loss 设计

最终目标不是让图像重建最好，而是让 AIGC / Natural 分类效果最好，尤其是**未见生成器泛化能力最好**。

### 1. 分类损失

[
\mathcal{L}_{cls}
=================

\operatorname{BCE}(C(z), y)
]

作用：

> 保证隐变量 (z) 对 AIGC / Natural 判别有用。

---

### 2. Pairwise naturalness energy loss

定义自然性能量：

[
E(z)=-\log p_{\text{nat}}(z)
]

简单情况下可以用：

[
E(z)=|z|_2^2
]

对于 DDA-COCO pair：

[
\mathcal{L}_{pair}
==================

\max(0, m + E(z_{real}) - E(z_{aigc}))
]

作用：

> 在语义对齐条件下，让真实图像 latent 更自然，AIGC 图像 latent 更异常。

---

### 3. 真实图像先验约束

对真实图像：

[
q(z|u_{real}) \approx p_{\text{nat}}(z)
]

例如：

[
p_{\text{nat}}(z)=\mathcal{N}(0,I)
]

对应：

[
\mathcal{L}_{KL}^{real}
=======================

D_{KL}(q(z|u_{real}) | \mathcal{N}(0,I))
]

作用：

> 建立真实图像的自然低层隐变量分布。

---

### 4. 低层特征重建损失

[
\hat{u}=D(z)
]

[
\mathcal{L}_{feat-rec}
======================

|D(z)-u|
]

作用：

> 防止 (z) 退化成纯分类黑盒，让它仍然能解释低层 forensic feature。

但这个 loss 是辅助项，权重不能太大。

---

## 六、最终总目标

[
\mathcal{L}
===========

\lambda_{cls}\mathcal{L}*{cls}
+
\lambda*{pair}\mathcal{L}*{pair}
+
\lambda*{KL}\mathcal{L}*{KL}^{real}
+
\lambda*{rec}\mathcal{L}_{feat-rec}
]

权重关系建议：

[
\lambda_{cls} \approx \lambda_{pair} > \lambda_{KL} > \lambda_{rec}
]

也就是：

```text
分类判别 + pairwise 能量约束
>
自然先验正则
>
低层特征重建
```

---

## 七、最终方案一句话总结

本项目将 AIGC 图像检测建模为**低层隐变量分布建模问题**：首先利用 diffusion-based low-level feature extractor 提取图像底层 forensic 特征，再通过随机隐变量编码器学习低层判别隐变量 (z)。借助 DDA-COCO 的语义对齐样本，模型在隐空间中施加 pairwise naturalness energy 约束，使真实图像的隐变量更接近自然图像分布，而 AIGC 图像的隐变量表现出更高的分布偏离。最终通过分类损失、自然先验约束和低层特征重建共同优化，使隐变量既具有检测判别性，又保留对底层视觉统计的解释能力。
