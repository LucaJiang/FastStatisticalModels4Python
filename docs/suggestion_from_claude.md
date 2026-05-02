Part 1 — 项目当前状态评估，还应该补什么
我把整个项目当作一个端到端审计了一遍。已经很扎实的部分（不要再动）：实验代码、子进程隔离的 benchmark harness、9 张可视化、5 份带实测数字的中文文档、可工作的 reveal.js slides、可工作的 poster。但站在「PyCon US 30 分钟 + 海报」这个交付物的角度，下面这几样目前缺失或薄弱，按重要性排序：

A. 演讲交付层面（高优先级）
Speaker script / talking points — slides 里只有 <aside class="notes"> 一两行，30 分钟讲稿没成形。建议：每张幻灯片配 60–120 词逐字稿（中英对照都行），重点把 Numba/JAX 实测数字对应到口头叙事的"挂钩点"。
演讲计时验收 — 大纲承诺 3+12+10+2+1 = 28 分钟。现在 slides 内容如果按节奏走可能超时。建议彩排一次并记录每节的实际用时。
Backup slides — 总会有人问"那 GPU 怎么样？"、"对比 Julia 呢？"、"对比 Cython 呢？"。准备 3–5 张备用 slides 放在最后的"hidden"区域，回答时再调出来。
现场 demo — 30 分钟的演讲，最好留 2 分钟做 live demo（不是预录视频）。建议在 experiments/kmeans/ 加一个 demo.py 可以一行起 Numba 编译→跑→出图，让 audience 看到 cold→warm 的真实切换。
Q&A 防御卡 — 把"为什么没用 free-threaded build 实测"、"为什么 Apple Silicon"、"为什么 K=5 而不是 K=50" 等可预见的质疑写成 FAQ 备查。
B. 技术内容层面（中等优先级）
k-means correctness 与 sklearn 的对照 — 你已经验证三个 impl 之间的 inertia 一致到 1 ULP，但没有和 sklearn.cluster.KMeans 对照。在演讲里加一句"和 sklearn 在同初始化下 inertia 相对差 < 1e-10"会显著提升可信度。
Free-threaded build 的实测 — 这是整个 talk 标题的核心卖点，你目前完全没在 free-threaded 上跑过。即使本机暂时装不了，也应该：
在云上租一台 Linux + 装 python3.14t（free-threaded 二进制）跑一次置换检验
把"GIL 1.4×"和"无 GIL X×"两条柱并排画
没有这条，"Breaking the Speed Limit" 标题对不上证据
Python 3.14 JIT 的实测 — 同上，目前只是论证"应该会有用"，没数据。在 GIL build with --enable-experimental-jit 上跑 kmeans_loops（小 N）做 JIT on/off 对照。即使加速比小，也比"实验未跑"强。
更大的 k-means K 和 d — 当前 K=5、d=10 是"k-means 教科书"配置，正好是 NumPy matmul 最不利的场景。补一组 K=50, d=100 的实验展示 matmul 优势开始显现，让"smart NumPy 是否有用"有完整答案。
GPU 一组数字 — 哪怕只是 Colab 上跑一次 JAX 置换检验 R=10000，把 75 s（CPU）→ 5 s（GPU）放在同一张图里，是该 talk 最有冲击力的 5 秒。
C. 工程基础设施（低优先级但加分）
CI / 自动化 — 把 sweep_kmeans.py + sweep_permtest.py 包进 make bench 或 GitHub Actions（可手动触发）。让 reviewer 一键复现。
环境锁定 — 现在 requirements-base.txt 用 >=。给一份 requirements-lock.txt 用准确版本号，避免某个微小升级把你的数字打乱。
README.md 的 results 摘要 — 顶层 README 没有引用 v2 的 results。加 4–6 行 "key findings" 摘要 + 一张 hero 图（建议用 tradeoff_radar.png）。
License & citation — 现在仓库有 LICENSE 但没有 CITATION.cff。PyCon 听众有引用习惯，提供一个 BibTeX/CFF 让他们一键引用。
Slide 中的「devex」环节有内容缺口 — 当前只列了"start simple / profile / AI as copilot"三条。建议换成基于本仓库实践的具体经验：例如"Numba 报错信息 → Numba 经验"、"JAX shape error → JAX 经验"，把空话变成现身说法。
D. 学术 / 演讲的传播
同步制作 1 页 talk handout — 海报已经做好（poster.pdf），但听众散场带走的是 1 页。建议从海报里抽一张"决策卡"PDF（runtime/memory/effort 的 4 列表）。
录一个 5 分钟 lightning preview — PyCon 一般有 lightning。把核心结论压缩到 5 分钟视频，做 talk 的预热／替补。
预先推一篇 blog 帖 — 演讲前 1–2 周在 lucajiang.github.io 上发一篇带 v2 数字的预览帖。这能在演讲前给海报上的 GitHub 链接带流量。
Part 2 — 可以再做的实验（让 project 更扎实）
按"性价比"（实施成本 vs 演讲价值）排序：

性价比最高（强烈建议做）
E1. Free-threaded Python 3.14 实测（前面 #7）

工作量：1 天（云机器 + setup_envs.sh 加 free-threaded 分支）
价值：直接坐实 talk 标题，把"线程 1.4×"打到"4–6×"
实施：用 py-free-threading.github.io 的预编译包；在 Linux EC2/GCP 上跑 permtest_freethreaded.py 即可，代码不用改
E2. k-means scaling 在更"现实"的 K, d 下重测（前面 #9）

工作量：30 分钟（重跑 sweep）
价值：把"matmul-NumPy 反而更慢"的 caveat 拓宽为完整故事："何时该用、何时不该用"
实施：--n-features 100 --k 50 改一下 sweep；新加一个 kmeans_scaling_K50.png
E3. GPU 一组 JAX 数字（前面 #10）

工作量：2 小时（Colab 免费 T4 / 学校 GPU）
价值：把 JAX 在 talk 里的"反例"翻成"正例"，结尾留 hook
实施：现有代码不需要改，JAX 自动用 GPU。只跑 R=10000 一档即可
E4. Numba 子进程 RSS 也加上探针

工作量：复用已有 _measure_children_rss_mb 函数 30 分钟
价值：现在 multiprocessing 显示 833 MiB；如果 Numba prange 也有"隐藏"内存（比如线程局部缓冲区），实测出来才公平
性价比中等（看时间允许）
E5. 真实 EM 算法 / GMM 一组实验

工作量：1–2 天
价值：你 README 里说自己是 biostat PhD，平时做 EM。把 k-means 升级到 1 个真实 GMM EM step 跑一遍，演讲里"我们组真实代码加速了 X 倍"的故事更具说服力
文件位置：建议 experiments/em_gmm/
E6. Cython / mypyc baseline

工作量：1 天（写 Cython 版 kmeans 内核）
价值：有人会问"为什么不写 Cython"。直接给一组数字（大概率：和 Numba 平手 ± 30%，但开发体验差）
实施：单独的 kmeans_cython.pyx + setup.py
E7. JAX lax.while_loop 版本（早停）

工作量：半天
价值：当前 JAX 用 scan 做固定 30 iter，naive NumPy 早停在 4 iter。**这是 JAX vs NumPy 在 N=500k 看上去赢得不明显的真实原因。**用 while_loop 实现早停后重测，JAX 数字应该会好看一些
E8. NumPy vs Cython 之外，加一个 R + Rcpp 数字

工作量：1 天（你 README 里说统计学家用 R + Rcpp 是"现状"）
价值：这是 talk motivation 的核心对照（"我们要替代 Rcpp"）。给一个 R + Rcpp k-means 数字，让"省掉 Rcpp"有量化支撑
性价比偏低但有价值
E9. 不同 BLAS 后端的对比（OpenBLAS / Apple Accelerate / MKL）

价值：解释 Apple Silicon 上 NumPy 矩阵乘法慢得反常的现象。可能涉及 conda 环境切换
E10. 内存压力下的稳定性测试

让 N 从 1M 升到 5M，naive NumPy 应该会 OOM。把"哪一档先 OOM"做成额外卖点
E11. 多次重启的 wall-time variance

当前 warm std 比较小（0.01–0.4 s），但跨重启的 variance 没量化。在 5 个独立 Python 解释器里各跑一次，给"跨日测量"的一致性
我个人最优先建议（如果只能做一件事）
做 E1（free-threaded 实测）。这是 talk 标题里写的卖点，目前完全没数据——任何细心的 PyCon 评委或观众都会发现。在 PyCon US 这种舞台上，标题字面对得上证据是底线问题。云上 1 小时就够了。

如果还能多做一件，做 E3（GPU JAX）——它会把 JAX 从 talk 里的"反派角色"拉回"留有想象空间"，让结尾更有张力。
