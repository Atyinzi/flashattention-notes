# flashattention-notes

FlashAttention v1 / v2 的学习笔记，从**原始论文**整理而成：

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（arXiv:2205.14135）
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)（arXiv:2307.08691）
- [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682)（arXiv:2112.05682）

---

## 文件

| 文件 | 说明 |
| --- | --- |
| [docs/FlashAttention-v1-v2-学习笔记.md](docs/FlashAttention-v1-v2-学习笔记.md) | 主笔记：算术强度与 IO → 在线 softmax 推导 → FA v1 → FA v2 → EMA → 实践建议，含速查表与 15 道自测题 |
| [assets/figures/](assets/figures/) | 自制示意图（SVG，可缩放） |

---

## 三句话抓住主线

1. **瓶颈不在算得多，而在搬得多**。softmax 是 memory-bound 归约操作，标准实现必须把 $N\times N$ 的 $\mathbf{S},\mathbf{P}$ 物化到 HBM 并来回搬运三次 —— 耗时主要由访存而非 FLOPs 决定。
2. **用"分块 + 在线 softmax + 重算"把这三次搬运消掉**。沿途只保留 $O(N)$ 的统计量 $(m,\ell)$，中间矩阵从不落 HBM。HBM 访问从 $\Omega(Nd+N^2)$ 降到 $\Theta(N^2d^2M^{-1})$，显存从 $O(N^2)$ 降到 $O(N)$，而**输出与标准注意力完全一致（非近似）**。
3. **v2 的主线是"砍掉 matmul 之外的开销"**。A100 上非 matmul FLOPs 的单位成本是 matmul 的 **16 倍**，所以 v2 减少每步除法、把并行度扩展到序列方向、并把 warp 切分从 $K,V$ 改成 $Q$ 以消除同步与部分和归约 —— 前向吞吐从峰值 30–50% 提到 50–73%。

---

## 关键结论速查

| | 标准注意力 | FlashAttention v1 | FlashAttention-2 |
| --- | --- | --- | --- |
| 时间复杂度（FLOPs） | $O(N^2d)$ | $O(N^2d)$（重算略增） | 同 v1 |
| 空间复杂度 | $O(N^2)$ | $O(N)$ | $O(N)$（再省一个向量） |
| HBM 访问次数 | $\Omega(Nd+N^2)$ | $\Theta(N^2d^2M^{-1})$ | 同 v1 |
| 并行维度 | batch × head | batch × head | batch × head **× 行分块** |
| warp 切分对象 | — | $K,V$（需同步 + 求和） | $Q$（直接拼接，无需通信） |
| 反向保存 | $\mathbf{S},\mathbf{P}$、dropout mask | $(m,\ell)$ 两个向量 | $L=m+\log\ell$ 一个向量 |
| 前向吞吐（A100 峰值占比） | — | 30–50% | **50–73%** |
| 反向吞吐 | — | 25–35% | **63%** |
| 相对标准注意力 | 1× | 2–4× | **3–10×** |
| 是否精确 | 基准 | ✅ 精确 | ✅ 精确 |

**两个"唯一"要记住**

- 唯一的**近似**变体是 **Block-Sparse FlashAttention**（跳过被 mask 的块，比 FA 再快 2–4×，可扩展到 64K 序列）；
- 唯一以**省显存**为首要目标（而非省 IO）的是 **EMA**：空间 $O(n^2)\to O(n)$，代价是耗时增加。

---

## 阅读建议

- **先建立直觉**：1.1 → 1.2，搞清 arithmetic intensity 与"为什么 FLOPs 不等于时间"。
- **再抠核心推导**：第 2 节是全文最值得亲手推一遍的地方，2.3 给了一个可以手算验证的 4 元素例子。
- **然后看工程**：3.2 的逐行解释与 4.2–4.4 的 v2 三处改进。
- **最后自测**：附录 B 的 15 道题按概念 / 推导 / 工程分层，答不上来再回查对应小节。
- **速查**：附录 A 是一页纸总结。

---

### 说明

公式、伪代码、复杂度结论与实验数字均取自上述论文正文，引用见笔记附录 C。
示意图为自制（SVG），手算验证示例与总结性表格为整理者补充。
若发现与论文不一致之处，**以论文为准**。
