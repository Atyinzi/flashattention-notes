# flashattention-notes

《万字长文详解 FlashAttention v1/v2》的**结构化学习笔记**。

> 原文：[万字长文详解 FlashAttention v1/v2](https://zhuanlan.zhihu.com/p/642962397) — 作者 Civ，收录于知乎专栏《Large Language Model》。

---

## 内容

| 文件 | 说明 |
| --- | --- |
| [docs/FlashAttention-v1-v2-学习笔记.md](docs/FlashAttention-v1-v2-学习笔记.md) | 完整学习笔记：背景与动机 → FA v1 → FA v2 → EMA/xformers → 基准测试 + 全部 48 个公式附录（内嵌原文图表） |
| [docs/原文提取文本.md](docs/原文提取文本.md) | 从源 PDF 提取的原文文本（供逐句核对） |
| [assets/figures-index.md](assets/figures-index.md) | 原文图表清单（图 2–图 11，按页渲染） |

笔记结构：

```
0. 记号与术语对照
1. 背景与动机        Transformer 复杂度 / FLOPs 与 MAC / 核心思路（GPU 存储层级、分块）
2. FlashAttention v1 在线 softmax 推导 / 前向伪代码逐行对照 / MAC 分析 / 反向显存技巧 /
                     Block-Sparse 变体 / 实验结果
3. FlashAttention v2 算法（去掉每步除法 + Causal Mask 跳块）/ 并行策略 / Work Partitioning /
                     实验结果
4. EMA 与 xformers   空间复杂度 O(n²)→O(n) 的在线累加推导
5. 基准测试代码索引
6. 复习清单（自测题 15 问）
附录 A. 全部 48 个编号公式
附录 B. 参考资料
```

---

## 三句话抓住主线

1. **出发点**：Transformer 的瓶颈不只是 FLOPs。FlashAttention 把优化目标从「少算」改成「**少搬运**」——降低 HBM 与 SRAM 之间的存储访问开销（MAC）。
2. **手段**：**分块（tiling）+ 在线 softmax 动态重缩放 + 重算代替保存**。中间矩阵 $S,P$ 永远不落 HBM，MAC 从 $O(N^2)$ 降到 $O(N^2d^2/M)$，并且**数学结果与标准注意力一致**（精确算法，非近似）。
3. **v1 → v2**：v1 已比标准实现快 2–4×，但 A100 上吞吐量只有极限的 30–50%；v2 通过「**放宽在线 softmax 的缩放时机** + **把行方向分块纳入并行** + **以 $K,V$（v1）改为以 $Q$（v2）做 warp 切分**」把吞吐量推到 **230 TFLOPs/s（A100 极限的 73%）**。

---

## 关键结论速查

| 项目 | 标准 Self-Attention | FlashAttention v1 | FlashAttention v2 |
| --- | --- | --- | --- |
| 时间复杂度 | $O(N^2d)$ | $O(N^2d)$（略增，重算换 IO） | 同 v1 |
| 空间复杂度 | $O(N^2)$ | $O(N)$（不物化 $S,P$） | $O(N)$ |
| MAC（HBM 访问） | $O(N^2)$ | $O(N^2d^2/M)$ | 同 v1 |
| 并行维度 | batch × head | batch × head | batch × head **× 行分块（外循环）** |
| warp 切分对象 | — | $K,V$（结果需相加 + 同步） | $Q$（结果直接拼接，**无需通信**） |
| 反向保存量 | $S,P$、Dropout Mask | $O$、$m$、$\ell$ + RNG 状态 | 合并为 $L_i=\log\ell_i+m_i$ |
| 是否精确 | 基准 | ✅ 精确 | ✅ 精确 |
| 相对速度 | 1× | 2–4× | 3–10×（比 v1 再快 1.3–3×） |

**两个"唯一"要记住**：

- 全文唯一**近似**变体是 **Block-Sparse FlashAttention**（被 mask 的分块直接跳过）；
- 唯一**不是为了加速而是为了省显存**的算法是 **EMA**（空间 $O(n^2)\to O(n)$，代价是耗时 +10%~15%；序列长 16,384 时显存 1 GB → 17 MB）。

---

## 阅读建议

- **第一次读**：1.2 → 1.3 → 2.1 → 3.1，先把"为什么 IO 是瓶颈"和"在线 softmax 怎么重缩放"弄明白。
- **抠细节**：2.2 的伪代码逐行对照表 + 附录 A 的公式编号，两者配合看原论文 `Algorithm 1` 最快。
- **面试/自测**：直接做第 6 节的 15 道题，答不上来再回查对应小节。

---

## 关于本仓库

- 定位是**学习笔记**：保留原文的推导主线、章节顺序与全部编号公式，并补充术语记号对照、结论速查表、自测清单与公式附录。
- 原文 PDF 中公式为矢量图形、无法直接复制文本，附录 A 的公式由页面图像 OCR 结合上下文语义恢复；个别上下标排版可能与原文略有差异，**以原论文为准**。
- 原文版权归原作者 Civ 所有；本仓库仅用于个人学习与检索。
