# DAS 车辆信号分类算法文档

## 1. 问题背景

DAS (Distributed Acoustic Sensing) 光纤传感系统记录地表振动信号。车辆经过传感器时产生特征性的地震波形，不同类型的车辆（大车/小车）因物理属性差异产生不同的信号特征。本算法的目标是在无标注数据的条件下，通过无监督聚类将信号自动分为大车（class 0）、小车（class 1）和其他（class 2）三类。

### 物理基础

车辆对 DAS 传感器的激励可近似为移动点荷载。信号的主要差异来源于：

- **车辆质量 M**：荷载力 F = Mg，大车质量大 → 振幅大
- **车辆长度 L**：车辆通过传感器的持续时间 T = L/v，大车更长 → 信号持续时间更长
- **轴数 N_axle**：每个车轴产生一个独立的力脉冲，大车（2-5 轴）比小车（2 轴）产生更多的波峰
- **轴距分布**：大车轴距更大且不规则，小车前后轴距紧凑

## 2. 特征提取 (`features.py`)

从每条信号 x[n], n = 0, 1, ..., N-1 中提取 26 个特征，分为三组。

### 2.1 基础统计特征（8 个）

| # | 特征 | 公式 | 物理含义 |
|---|------|------|----------|
| 1 | mean | $\bar{x} = \frac{1}{N}\sum x[n]$ | 信号直流分量，理想情况下接近零 |
| 2 | std | $\sigma = \sqrt{\frac{1}{N}\sum(x[n]-\bar{x})^2}$ | 信号波动强度 |
| 3 | rms | $x_{rms} = \sqrt{\frac{1}{N}\sum x[n]^2}$ | 均方根值，与振动能量成正比 |
| 4 | peak_to_peak | $x_{max} - x_{min}$ | 峰峰值，反映最大振幅动态范围 |
| 5 | skew | $\gamma_1 = \frac{E[(x-\mu)^3]}{\sigma^3}$ | 偏度，波形对称性 |
| 6 | kurtosis | $\gamma_2 = \frac{E[(x-\mu)^4]}{\sigma^4} - 3$ | 峰度，波形尖锐程度（超额峰度） |
| 7 | zero_crossing_rate | $ZCR = \frac{1}{N-1}\sum_{n=1}^{N-1} \mathbb{1}[\text{sgn}(x[n]) \neq \text{sgn}(x[n-1])]$ | 过零率，反映信号的主要振荡频率 |
| 8 | signal_energy | $E = \sum x[n]^2$ | 总能量，与车辆质量和速度相关 |

### 2.2 频谱特征（6 个）

对信号做实数 FFT：$X[k] = \text{FFT}(x[n])$，功率谱 $P[k] = |X[k]|^2$，频率轴 $f[k]$。

| # | 特征 | 公式 | 物理含义 |
|---|------|------|----------|
| 9 | spectral_centroid | $f_c = \frac{\sum f[k] \cdot P[k]}{\sum P[k]}$ | 频谱质心，功率谱的"重心频率"。大车低频能量占比高，质心偏低 |
| 10 | spectral_bandwidth | $BW = \sqrt{\frac{\sum (f[k]-f_c)^2 \cdot P[k]}{\sum P[k]}}$ | 频谱带宽，功率谱的展宽程度。纯音信号带宽极窄 |
| 11 | spectral_rolloff | 累积功率达到 85% 时的频率 $f_r$：$\sum_{k=0}^{k_r} P[k] \geq 0.85 \sum P[k]$ | 频谱滚降点，信号能量的频率上界 |
| 12 | spectral_flatness | $SF = \frac{\exp(\frac{1}{K}\sum \ln P[k])}{\frac{1}{K}\sum P[k]} = \frac{\text{几何均值}}{\text{算术均值}}$ | 频谱平坦度。白噪声 SF → 1，纯音 SF → 0。用于检测异常的纯音干扰信号 |
| 13 | low_high_energy_ratio | $R = \frac{\sum_{k < K/4} P[k]}{\sum_{k \geq K/4} P[k]}$ | 低频/高频能量比。将频谱按 1/4 分位切分，大车低频占比更高 |
| 14 | dominant_freq | $f_d = f[\arg\max_k P[k]]$ | 主频，功率谱峰值对应的频率 |

### 2.3 包络形状特征（12 个）— 核心创新

这组特征是区分大车/小车的关键。通过 Hilbert 变换提取信号的瞬时包络，从而捕捉波形的时域形状。

#### 2.3.1 Hilbert 变换与解析信号

对实信号 x[n]，通过 Hilbert 变换构造解析信号：

$$z[n] = x[n] + j \cdot \mathcal{H}\{x\}[n]$$

其中 $\mathcal{H}$ 是 Hilbert 变换算子，在频域中定义为：

$$\mathcal{H}\{x\}[k] = \begin{cases} -jX[k], & f[k] > 0 \\ +jX[k], & f[k] < 0 \\ 0, & f[k] = 0 \end{cases}$$

瞬时包络为解析信号的模：

$$A[n] = |z[n]| = \sqrt{x[n]^2 + \mathcal{H}\{x\}[n]^2}$$

包络 A[n] 是一条平滑的非负曲线，描绘信号振幅随时间的变化轮廓。物理上，它反映了车辆从远处驶近、经过传感器正上方、再驶离的整个过程。

#### 2.3.2 包络宽度特征

| # | 特征 | 定义 | 物理含义 |
|---|------|------|----------|
| 15 | envelope_width_50 | 包络 $A[n] \geq 0.5 \cdot A_{peak}$ 的采样点数 | 半峰全宽 (FWHM)。直接度量车辆通过传感器的有效持续时间。大车约 600-1000 点，小车约 200-400 点（250Hz 采样率下） |
| 16 | envelope_width_25 | 包络 $A[n] \geq 0.25 \cdot A_{peak}$ 的采样点数 | 更宽的包络宽度，捕捉信号的"尾部"展宽。大车因车身长度，25% 处宽度远大于 50% 处宽度 |
| 17 | envelope_area | $\frac{\sum A[n]}{A_{peak} \cdot N}$ | 归一化包络面积。衡量信号的"填充率"。大车信号更持续，填充率高；小车信号尖锐集中，填充率低 |

#### 2.3.3 上升/下降时间特征

定义峰值位置 $n_p = \arg\max A[n]$，两个阈值 $A_{10} = 0.1 A_{peak}$，$A_{90} = 0.9 A_{peak}$：

| # | 特征 | 定义 | 物理含义 |
|---|------|------|----------|
| 18 | rise_time | 包络从 $A_{10}$ 上升到 $A_{90}$ 的采样点数（在 $n_p$ 之前） | 车辆接近传感器的过程持续时间。大车更长更缓慢 |
| 19 | fall_time | 包络从 $A_{90}$ 下降到 $A_{10}$ 的采样点数（在 $n_p$ 之后） | 车辆驶离传感器的过程持续时间 |
| 20 | rise_fall_asymmetry | $\alpha = \frac{T_{rise} - T_{fall}}{T_{rise} + T_{fall}}$ | 上升/下降不对称度。$\alpha > 0$ 表示接近比离开更缓慢，$\alpha < 0$ 反之。范围 [-1, 1] |

#### 2.3.4 峰结构特征

对包络进行滑动平均平滑（窗口 50 点 = 0.2 秒 @250Hz），然后用 `find_peaks` 检测显著峰（prominence > 0.3 * peak，最小间距 50 点）：

| # | 特征 | 定义 | 物理含义 |
|---|------|------|----------|
| 21 | num_envelope_peaks | 检测到的显著峰数量 | 车轴数量的代理量。小车通常 1-2 峰（前后轴），大车 2-4 峰（多轴组） |
| 22 | peak_prominence_mean | 各峰突出度的均值 | 峰的"显著性"。多轴车辆的各轴峰更独立、突出度更高 |
| 23 | peak_spacing_std | 相邻峰间距的标准差（<2 峰时为 0） | 轴距规律性。大车各轴距分布不均匀（前轴-后轴组-挂车轴）→ 标准差大 |

**峰检测的参数选择依据**：
- prominence 阈值 0.3 * peak：排除包络上的微小波动，只保留独立的轴响应
- 最小间距 50 点 = 0.2 秒 @250Hz：以 60km/h 计算，0.2 秒对应约 3.3 米，小于最短轴距

#### 2.3.5 能量分布与幅值特征

| # | 特征 | 公式 | 物理含义 |
|---|------|------|----------|
| 24 | crest_factor | $CF = \frac{A_{peak}}{x_{rms}}$ | 峰值因子。小车信号尖锐（峰值远大于 RMS）→ CF 高；大车信号持续平稳 → CF 低。该特征振幅无关（比值消去绝对量级） |
| 25 | energy_gini | Gini 系数 $G = 1 - \frac{2\sum_{i=1}^{K} C_i}{S \cdot K}$ | 能量时间分布的不均匀度。将信号等分为 K=20 段，计算各段能量，排序后求 Gini 系数。G → 1 表示能量集中在少数段（小车），G → 0 表示均匀分布（大车） |
| 26 | log_rms | $\log_{10}(x_{rms} + \epsilon)$ | 对数 RMS。取对数压缩动态范围，使不同距离的车辆在同一尺度上更可分 |

其中 Gini 系数的具体计算：
1. 将信号等分为 K=20 段，计算每段能量 $e_i = \sum_{n \in \text{seg}_i} x[n]^2$
2. 将 $e_i$ 升序排列得 $e_{(1)} \leq e_{(2)} \leq \cdots \leq e_{(K)}$
3. 计算累积和 $C_i = \sum_{j=1}^{i} e_{(j)}$，总和 $S = \sum e_{(i)}$
4. $G = 1 - \frac{2 \sum_{i=1}^{K} C_i}{S \cdot K}$

## 3. 聚类流程 (`cluster_vehicle_signals.py`)

### 3.1 总体流程

```
原始信号 (.pkl)
    │
    ▼
特征提取 (26 维)
    │
    ▼
噪声预过滤 (可选, --preassign_other)
    │
    ▼
StandardScaler 标准化
    │
    ▼
形状特征加权
    │
    ▼
PCA 降维 (可选, --use_pca)
    │
    ▼
HDBSCAN / KMeans 聚类
    │
    ▼
复合评分类别映射 → class 0(大车), 1(小车), 2(other)
    │
    ▼
输出 CSV + JSON + 波形可视化
```

### 3.2 噪声预过滤

在聚类之前，将明显不是车辆信号的样本预先标记为 class 2 (other)，不参与后续聚类。判据：

$$\text{is\_other}(x) = (\text{RMS}(x) \leq \text{rms\_floor}) \;\lor\; (\text{SF}(x) \leq \text{flatness\_max} \;\land\; \text{BW}(x) \leq \text{bandwidth\_max})$$

- **RMS 过低**：信号能量极弱，可能是无车通过时的背景噪声
- **频谱平坦度和带宽同时过低**：表明信号以单一频率为主（纯音），可能是设备干扰

默认阈值：`rms_floor=1e-8`, `flatness_max=0.02`, `bandwidth_max=0.02`。

### 3.3 特征标准化与加权

#### 标准化

对每个特征列做 z-score 标准化：

$$\hat{f}_i = \frac{f_i - \mu_i}{\sigma_i}$$

使所有特征在相同的数值尺度上，避免高数值特征（如 signal_energy）主导距离计算。

#### 形状特征加权

标准化之后，对形状特征乘以权重因子 $w_i > 1$，放大它们在欧氏距离中的贡献：

$$\tilde{f}_i = w_i \cdot \hat{f}_i$$

权重设定：

| 特征 | 权重 | 理由 |
|------|------|------|
| envelope_width_50 | 2.0 | 最强的大/小车区分特征，直接度量车辆长度 |
| num_envelope_peaks | 2.0 | 轴数差异是大/小车的本质区别 |
| envelope_width_25 | 1.5 | 补充包络宽度信息 |
| envelope_area | 1.5 | 补充填充率信息 |
| crest_factor | 1.5 | 振幅无关的尖锐度度量 |
| energy_gini | 1.5 | 振幅无关的时间集中度度量 |
| rise_time | 1.2 | 次要形状特征 |
| fall_time | 1.2 | 次要形状特征 |
| 其他特征 | 1.0 | 不加权 |

**数学效果**：在加权后的特征空间中，欧氏距离变为加权欧氏距离：

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{26} w_i^2 (\hat{x}_i - \hat{y}_i)^2}$$

权重 $w_i > 1$ 的特征对距离的贡献按 $w_i^2$ 放大，使聚类算法更关注形状差异。

### 3.4 PCA 降维

可选步骤。对加权特征矩阵做主成分分析：

1. 先用 `PCA(n_components=0.95)` 确定保留 95% 方差所需的成分数 $k_{95}$
2. 实际取 $k = \min(30, \max(10, k_{95}))$，限制在 [10, 30] 范围内
3. 如果 $k \geq$ 原始特征数，则不降维

### 3.5 聚类算法

#### HDBSCAN（默认）

Hierarchical Density-Based Spatial Clustering of Applications with Noise。相比 KMeans 的优势：
- 不需要预设簇数
- 能识别任意形状的簇
- 自动标记噪声点 (label = -1)

网格搜索策略：遍历 `min_cluster_size` 和 `min_samples` 的组合，选择 silhouette score 最高的（同时惩罚偏离 3-4 个簇的情况）：

$$\text{score}_{total} = S_{silhouette} - 0.1 \cdot |n_{clusters} - 3.5|$$

如果 HDBSCAN 产生 <2 或 >6 个簇，回退到 KMeans。

#### KMeans（备选）

经典的 k-means 聚类。遍历 k_min 到 k_max，选择 silhouette score 最高的 k 值。当 `--num_classes=3` 时直接使用 k=3。

#### Silhouette Score

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

其中 $a(i)$ 是样本 $i$ 到同簇其他点的平均距离，$b(i)$ 是到最近异簇所有点的平均距离。$s \in [-1, 1]$，越高表示聚类越紧凑且分离度越好。

### 3.6 复合评分类别映射

聚类完成后，需要将抽象的簇编号映射到语义类别（大车/小车/other）。

#### 旧方法的问题

旧方法仅比较两个最大簇的平均 RMS：
$$\text{big} = \arg\max_{c \in \{C_1, C_2\}} \overline{\text{RMS}}_c$$

当两个簇内部都混有大车和小车时，平均 RMS 的差异被平均效应抹平，映射错误。

#### 新方法：复合尺寸评分

对每个样本 $i$ 计算一个"车辆尺寸评分" $S(i)$，综合 9 个特征的 z-score：

$$S(i) = \sum_{j \in \text{正相关}} \frac{f_j(i) - \mu_j}{\sigma_j} - \sum_{j \in \text{负相关}} \frac{f_j(i) - \mu_j}{\sigma_j}$$

正相关特征（值越大 → 车越大）：
- envelope_width_50, envelope_width_25, envelope_area
- rise_time, fall_time
- num_envelope_peaks
- log_rms

负相关特征（值越大 → 车越小）：
- crest_factor（信号越尖锐 → 小车）
- energy_gini（能量越集中 → 小车）

然后取两个最大簇的样本平均评分，评分高的为大车簇，低的为小车簇，其余为 other。

$$\bar{S}_c = \frac{1}{|C_c|} \sum_{i \in C_c} S(i)$$

$$\text{big} = \arg\max_{c \in \{C_1, C_2\}} \bar{S}_c$$

这种方式即使簇内有少量混分，9 个特征的综合评分仍能正确判断整体倾向。

## 4. 运行方式

```bash
python cluster/cluster_vehicle_signals.py \
  --pkl_dir <pkl文件或目录> \
  --out_dir <输出目录> \
  --algo hdbscan \
  --preassign_other \
  --num_classes 3
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--pkl_dir` | 必填 | 输入数据路径 |
| `--out_dir` | 必填 | 输出目录 |
| `--algo` | hdbscan | 聚类算法: hdbscan / kmeans |
| `--num_classes` | 3 | 最终类别数 |
| `--preassign_other` | 关闭 | 开启噪声预过滤 |
| `--use_pca` | 关闭 | 开启 PCA 降维 |
| `--per_file` | 关闭 | 每个 pkl 文件单独聚类 |
| `--min_cluster_size` | 30 | HDBSCAN 最小簇大小 |
| `--min_samples` | 10 | HDBSCAN 核心点最小邻居数 |
| `--k_range` | 3,6 | KMeans 搜索范围 |
| `--length_mode` | crop | 信号长度处理: crop / pad |
| `--target_len` | 自动 | 目标信号长度 |
| `--rms_floor` | 1e-8 | RMS 过滤阈值 |
| `--flatness_max` | 0.02 | 纯音检测阈值 |
| `--bandwidth_max` | 0.02 | 纯音检测阈值 |

### 输出文件

```
out_dir/
  clusters_raw.csv           # 原始信号的分类结果
  clusters_norm.csv          # 归一化信号的分类结果
  cluster_summary_raw.json   # 聚类统计信息
  cluster_summary_norm.json
  data_info.json             # 数据加载信息
  cluster_samples/
    raw/
      cluster_0.png          # class 0 (大车) 波形示例
      cluster_1.png          # class 1 (小车) 波形示例
      cluster_2.png          # class 2 (other) 波形示例
    norm/
      ...
```
