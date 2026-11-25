# 第四章完整研究方案与仿真架构技术文档（`TRA-HGDRL`）

> **论文定位**：硕士毕业论文第四章  
> **中文题目**：面向低轨卫星直连手机网络的业务与风险感知多智能体协同切换方法  
> **英文题目**：Traffic- and Risk-Aware Multi-Agent Cooperative Handover for Direct-to-Cell LEO Satellite Networks  
> **核心算法**：`TRA-HGDRL`（多维异质图注意力编码 `HGAT` + 排列等变分位数分布强化学习 `QR-DQN` + 业务自适应近似 `CVaR` 风险敏感选星 + 负均方负载 $O(1)$ 解析近似反事实协作奖励 + 双阶段自适应触发 / SMDP 跨步回报）  
> **代码基线**：以仓库 **`paper-simulation` 分支（commit `9876f01`）** 作为第 3 章物理层、业务队列与冻结调度器底座，在独立目录 `handover/` 中向连续动态轨道切换扩展。

---

## 目录

1. [章节承接定位与文献对标分析](#1-章节承接定位与文献对标分析)
2. [第三章 `paper-simulation` 基线审计与复用修复清单](#2-第三章-paper-simulation-基线审计与复用修复清单)
3. [卫星—小区—用户三层动态仿真场景搭建规范](#3-卫星小区用户三层动态仿真场景搭建规范)
4. [双阶段自适应切换触发与卫星侧统一准入控制](#4-双阶段自适应切换触发与卫星侧统一准入控制)
5. [POMDP/SMDP 建模、异质图 HGAT 与 TRA-HGDRL 核心算法](#5-pomdpsmdp-建模异质图-hgat-与-tra-hgdrl-核心算法)
6. [代码目录架构与核心扩展接口规范](#6-代码目录架构与核心扩展接口规范)
7. [基准算法、消融实验与评价指标体系](#7-基准算法消融实验与评价指标体系)
8. [分阶段开发与验收路线图（Phase 0 ~ Phase 6）](#8-分阶段开发与验收路线图phase-0--phase-6)

---

## 1. 章节承接定位与文献对标分析

### 1.1 第 3 章与第 4 章的分工与衔接边界

第 3 章聚焦**准静态多星覆盖快照下的短时标跳波束与功率调度**（给定可见矩阵与周期性 ISA 负载均衡关联），而第 4 章将其拓展至**连续轨道动态演进下的长时标小区级星间切换**，并将第 3 章的 `ISA` 演进为 `ISA-HO` 跨章对照基准、将 `paper-simulation` 中已训练且在统一公平环境下全面领先的冻结 `MAPPO-BH` 作为最终端到端联合验证的底层调度器。

| 对比维度 | 第 3 章（`paper-simulation` 分支 `9876f01`） | 第 4 章（`chapter4-handover` 分支） |
| :--- | :--- | :--- |
| **核心研究问题** | 给定覆盖快照内的波束与功率资源利用效率 | 连续动态轨道演进下的通信连续性、负载均衡与 RT 可靠性 |
| **决策智能体** | 卫星侧（$N=16$ 颗静态卫星） | 固定地面小区侧（$M=144$ 个地球固定小区） |
| **时间尺度** | 微观跳波束时隙 $T_{\text{slot}}=2\text{ ms}$ | 宏观切换步长 $T_H=1\text{ s}$，单 Episode $300\sim 600\text{ s}$ |
| **动作空间** | 离散波束激活（$\le K_{\max}=4$）+ 连续功率分配 | 离散选星 `Stay` 或 Top-$C_{\max}$ 候选星（配合动态掩码） |
| **图神经网络** | 排列等变同质图 `GAT`（8 维 `F_rich` + `I_mat`/`Omega`/`C` 先验，可选跨星 `IterativeCoopPowerGNN`） | 异质图 `HGAT`（卫星节点 + 小区节点；覆盖/服务/竞争三类有向边） |
| **强化学习范式** | On-policy `MAPPO + GRU`（CTDE + 反事实差值奖励 / 解耦 APG） | Off-policy `QR-DQN`（可选 `Twin QR-DQN`）+ 业务自适应近似 `CVaR` |
| **协作与归因** | 局部+全局混合奖励 / 差值奖励 | 局部业务奖励 + 负均方负载 $O(1)$ 解析近似反事实协作奖励 |
| **底层调度执行** | 独立 2 ms 跳波束仿真 | 训练期使用经 `paper-simulation` 校准的等效波束服务模型；最终闭环测试挂载冻结 `MAPPO-BH` |

### 1.2 核心参考文献与差异化创新定位

1. **第 3 章自有底座论文**：Hao Qin, Fan Wang, Tengfei Hui, Jingling Li, Bin Song, *"Hierarchical Load Balancing and Cooperative Beam Hopping for Multi-Satellite Direct-to-Cell LEO Networks"*
   - 继承 D2C 物理层参数（515 km、40° 最小仰角、2 GHz、40 MHz、80 RB、34 dBW、34 dBi 圆孔径贝塞尔方向图、0 dBi 手持终端、300 K 噪声温度）、144 固定小区（16 个局部 $3\times 3$ 紧邻波束簇）、$J=14$、$K_{\max}=4$、RT/BULK 双队列（TTL=80 ms、MMPP 突发）、PF 调度、`ISA` 初始关联与冻结 `MAPPO-BH`（`mappo-apg-local` / `mappo-coop-gnn`）。
2. **核心多智能体图 RL 强基线**：付一阳, 胡博, 刘人鹏, 陈山枝, 《星地融合网络中基于异质图表征的多智能体协作切换方法》, 《计算机学报》, 2026, 49(2): 347-364（`MA-HGDQN`）
   - **原方案局限**：用户级切换（仅 30 用户、赤道 $2^\circ$ 小区域、30° 仰角），无业务队列与跳波束资源耦合，仅以用户数比例 $l_n$ 定义负载，仅用 $\text{RVT}<10\text{ s}$ 或断链触发，采用蒙特卡洛 Shapley 值计算开销较大，且为风险中性标准 DQN。
   - **本文超越**：升级至 144 固定小区 D2C 跳波束场景，引入双维度波束资源负载 $(\rho_n, \zeta_n)$、业务/链路/覆盖双阶段触发、$O(1)$ 解析反事实协作奖励，并引入分布 RL 与业务自适应风险敏感决策。
3. **分布鲁棒 RL 强参考**：Renpeng Liu, Xiaozhe Wang, Bo Hu, Shanzhi Chen, *"Twin Adaptive Distributional Reinforcement Learning for Robust Handover Control in LEO Satellite Networks"*, IEEE WCL, 2025, 14(12): 4012-4016（`TAD-RL`）
   - **原方案特点与差异**：`TAD-RL` 为最大熵 Actor-Critic 架构（DSAC 风格），使用**对称高斯分布** $\mathcal{N}(Q_\theta, \sigma_\theta^2)$ 建模软回报，主要用于抑制 TD 噪声与异构奖励尺度，**无法刻画 LEO 切换中“常态平稳、偶发断链或 RT 集中超时暴跌”的非对称左偏厚尾分布**。本文采用无参数分位数分布（`QR-DQN`）直接刻画左尾风险，并依据小区实时业务属性动态调整 `CVaR` 风险水平。
4. **前期切换建模参考**：
   - 刘人鹏, 胡博, 李鹤群, 《星地融合网络中基于多目标优化的星间切换决策方法》, 《天地一体化信息网络》, 2023, 4(3): 59-66（3GPP CHO 条件切换、时空快照与负载方差形式）。
   - 刘人鹏, 涂文康, 胡博 等, 《星地融合网络中基于无监督图学习的卫星切换方法研究》, 《移动通信》, 2024, 48(1): 33-39（8 步 RRC 切换信令与时空覆盖演进图）。
5. **外部核心对比与理论基础文献**：
   - **小区级 RL 基准（`Cell-RL`）**：Cheng Wang et al., *"Cell-Level Handover for LEO Satellite Networks: A Reinforcement Learning-Based Approach"*, IEEE Commun. Lett., 2026, 30: 2421-2425。
   - **跳波束主动交换基准（`Swap-HO`）**：Yaohua Sun, Mugen Peng et al., *"Beam Management in Low Earth Orbit Satellite Communication With Handover Frequency Control and Satellite-Terrestrial Spectrum Sharing"*, IEEE TCOM, 2025。
   - **D2C 切换背景**：Yuhe Feng et al., *"MADRL Based Energy-Efficient Handover for Direct-to-Cell Ultra-Dense LEO Satellite Network"*, IEEE VTC-Spring, 2025。
   - **分布 RL 与 CVaR 理论边界**：Bellemare et al. (ICML 2017)、Dabney et al. (AAAI 2018, QR-DQN)、**Lim & Malik (NeurIPS 2022)**。严格遵循 Lim & Malik 的理论结论：**不宣称标准分布 Bellman 算子直接求得严格全局静态/动态 CVaR 最优解，而是定位为“基于回报分布分位数估计的业务自适应近似风险敏感决策方法”**。

---

## 2. 第三章 `paper-simulation` 基线审计与复用修复清单

### 2.1 `paper-simulation` 分支真实系统参数核对（来自 `config.py` @ `9876f01`）

| 参数类别 | 参数名称 | `paper-simulation` 代码设定 | 第四章处理方式 |
| :--- | :--- | :--- | :--- |
| **网络规模** | 卫星数 $N$ / 地面小区数 $M$ | $16$（静态 $4\times 4$）/ $144$（16 个局部 $3\times 3$ 簇） | 改为动态 Walker 星座活跃星集 $\mathcal{N}(t)$ / 继承 $144$ 固定小区 |
| **波束能力** | 单星最大关联数 $J$ / 最大同时波束 $K_{\max}$ | $14$ / $4$ | 100% 继承 |
| **轨道与几何** | 轨道高度 $H$ / 最小仰角 $\vartheta_{\min}$ / 3 dB 波束宽 | $515\text{ km}$ / $40^\circ$ / $3^\circ$ | 100% 继承，补充 $53^\circ$ 倾角动态轨道 |
| | 簇内小区间距 / 簇心间距 / 小区用户半径 | $25\text{ km}$ / $165\text{ km}$ / $7.5\text{ km}$（总跨度 $545\text{ km}$） | 100% 继承，映射至地球球面 ECEF |
| | 距离先验参数 $d_{\text{th}}$ / $\eta$ | $30\text{ km}$ / $5\text{ km}$ | 100% 继承（供冻结 `MAPPOActor` 使用） |
| **射频与信道** | 载波频率 $f_c$ / 系统带宽 $B_{\text{tot}}$ / 每波束 RB 数 | $2\text{ GHz}$ / $40\text{ MHz}$ / $80$ | 100% 继承 |
| | 发射总功率 $P_{\text{tot}}$ / 发射增益 $G_{\max}^T$ / 接收增益 $G_R$ | $34\text{ dBW}$ / $34\text{ dBi}$（贝塞尔）/ $0\text{ dBi}$ | 100% 继承，随实时三维矢量更新离轴角 |
| | 噪声温度 $T_{\text{noise}}$ / 阴影衰落标准差 | $300\text{ K}$ / $3\text{ dB}$（及可选小区级 AR(1)） | 继承 $300\text{ K}$，升级为 **按 `(sat_id, cell_id)` 链路级独立维护** 的 AR(1) 阴影衰落 |
| **业务与队列** | 用户数 $U$ / 默认评估用户数 / 每用户速率 | $100\sim 500$ / `EVAL_USERS=200` / $10\sim 50\text{ Mbps}$ | 默认 $U=200$（扫描 $200\sim 500$），增加空间热点聚集、RT 占比分化与热点漂移 |
| | RT 均值占比 / BULK 均值占比 / 包长 $L_p$ | $20\%$ / $80\%$ / $1000\text{ bits}$ | 100% 继承全网均值与包长 |
| | MMPP 突发参数（ON 概率 / 保持概率 / ON 倍率） | $0.16$ / $0.94$（约 33 ms）/ $5.5\times$ | 100% 继承 |
| | 微观时隙 $T_{\text{slot}}$ / RT TTL $t_{\text{th},R}$ / PF 窗口 | $2\text{ ms}$ / $40\text{ slots}$ ($80\text{ ms}$) / $50\text{ slots}$ | 100% 继承，新增小区级切换中断倒计时 `ho_remaining_slots` |

### 2.2 `paper-simulation` 已解决的优势与仍需在 `handover/` 修复的遗留问题

**已由 `paper-simulation`（`9876f01`）直接解决的能力**：
- 所有第 3 章对比策略共享 100% 统一的物理层环境，50 Mbps 下 `MAPPO-APG-Local`（38.59 Mbps / RT 丢包 1.77%）和 `MAPPO-CoopGNN`（40.41 Mbps / RT 丢包 0.78%）全面领先 `USWGP-BH`（36.90 Mbps / RT 丢包 4.33%）；
- `env.py` 已输出 `served_rt`、`served_rt_age`、`expired_rt`、`arrived_rt`，可直接统计真实交付时延与 RT 超时丢包率；
- 已修复 `Omega_prior` / `action_mask` 原地覆写引用问题；
- `MAPPOActor` 已升级为**严格排列等变结构**（Mean Pooling + 共享逐节点头，无 `Flatten(J*d_g)`），天然兼容候选小区重排与动态卫星数量 `N_act`。

**仍需在 `handover/` 中独立修复的 5 项遗留问题**：
1. **[P0] 覆盖矩阵人为补全与截断耦合**：`paper-simulation` 的 `env.py` 仍将每星可见集合硬截断至最近 14 个小区，再对无星覆盖的小区强行补最近卫星。第四章必须彻底拆分为几何可见矩阵 $V^{\text{geo}}$、小区候选星集 $\mathcal{C}_m(t)$ 与实际关联矩阵 $Z(t)$，**允许无可行卫星时产生真实通信中断**。
2. **[P0] `ISA` 初始化容量越界与空可见崩溃**：`association.py` 的 `greedy_init()` 在可见星均满 $J$ 时仍退化为无容量限制分配，且全不可见时 `argmin` 报错。第四章的 `ISA` 初始关联与 `ISA-HO` 必须严格遵守硬约束 $\sum_m Z_{n,m} \le J$ 并正确处理中断小区（`cell_sat[m] = -1`）。
3. **[P0] `np.argmax(Z, axis=0)` 无法表达未连接小区**：当某小区 $m$ 中断（$Z[:, m]$ 全为 0）时，`argmax` 会误返回 0 号卫星。必须改为显式判断 `Z.sum(axis=0) > 0`，未连接小区记为 `-1`。
4. **[P0] 链路级独立 AR(1) 阴影衰落**：`paper-simulation` 的 `TIME_VARYING_FADING` 为小区级共享（`cell_fade_db[m]` 对所有卫星乘同一衰落因子），第四章切换必须改为 **按 `(sat_id, cell_id)` 链路级独立维护 AR(1) 阴影衰落**，否则换星无法避开遮挡衰落。
5. **[P1] 评估公平性铁律与数组深拷贝**：所有切换算法与消融变体共享 100% 相同的物理层参数、轨道轨迹、业务随机种子、AR(1) 阴影序列与卫星侧统一准入控制模块；任何写入 Replay Buffer 的状态/掩码数组强制 `.copy()`。

### 2.3 第 3 章冻结 `MAPPOActor`（`paper-simulation`）的动态挂载规范

相比 `main`，`paper-simulation` 的 `MAPPOActor` 移除了非等变 Flatten 层，挂载到第 4 章动态拓扑更干净，只需在 `mappo_adapter.py` 中完成三件事：

1. **按卫星唯一 ID 维护 GRU 隐状态**：以字典 `hidden_states[sat_id] = torch.zeros(1, 1, 128)` 独立管理每颗活跃卫星的时序隐状态；新星进入初始化为零向量，离开后释放。
2. **动态刷新 8 维 `F_rich` 与精确星内耦合矩阵 `I_mat`**：
   - `F_rich (N_act, J, 8)` 对应 `[rt_q, bulk_q, rt_delay, hol, snr_db_norm, hist_tput, v_local, num_users]`；
   - 每 1 s 拓扑更新后，随 `H_sq` 重新计算活跃卫星的 `I_mat (N_act, J, J)`、`Omega` 与 `C`。
3. **支持两档冻结权重无缝切换（Phase 6 均建议测试）**：
   - **默认主挂载（纯星上分布式、无星间实时信令）**：`checkpoints/mappo-apg-local/mappo_bh.pt`（仅需 8 维局部 `F_rich` + `I_mat`，不依赖 `NeighborInterference` 与跨星矩阵 `E_mat`）；
   - **高阶协同挂载（跨星协同功率控制上限）**：`checkpoints/mappo-coop-gnn/mappo_bh.pt`（每 1 s 拓扑/关联变更时调用 `NeighborInterference` 刷新 3 维邻居干扰特征与跨星耦合矩阵 `E_mat (N_act*J, N_act*J)`；`IterativeCoopPowerGNN` 仅在激活波束子集上运行，天然兼容动态 `N_act`）。

---

## 3. 卫星—小区—用户三层动态仿真场景搭建规范

### 3.1 轨道几何实测校验（匹配 `paper-simulation` 的 545 km 区域跨度）

在 `paper-simulation` 设定的轨道高度 $H=515\text{ km}$ 与最小仰角 $\vartheta_{\min}=40^\circ$ 下，由球面几何精确可得：
- 轨道周期：$T_{\text{orb}} = 2\pi\sqrt{(R_E+H)^3 / \mu} = 5686.7\text{ s}$（$94.78\text{ min}$）；
- 最大地心覆盖半角：$\psi_{\max} = \arccos\!\left(\frac{R_E}{R_E+H}\cos 40^\circ\right) - 40^\circ = 4.87^\circ$；
- 单星地面覆盖半径：$R_{\text{cov}} = R_E \psi_{\max} = \mathbf{541.1\text{ km}}$（直径约 $1082\text{ km}$）；
- 单星正过顶最长可见时长：$T_{\text{pass}}^{\max} = 2\psi_{\max}/\omega_{\text{orb}} = \mathbf{153.7\text{ s}}$（约 2.56 分钟，600 s Episode 内每个小区自然经历 **4～6 次**过境切换）。

由于 `paper-simulation` 的 144 个地面小区由 16 个 $3\times 3$ 紧邻波束簇（簇内间距 $25\text{ km}$、簇心间距 $165\text{ km}$）组成，总地面跨度为 **$545\text{ km}\times 545\text{ km}$**（较 `main` 的 648 km 更紧凑，簇内同频邻波束干扰更贴近真实 3° 波束脚印）。经轨道实测：
- 若在 $42^\circ\text{N}$ 使用 $72\times 22$ 星座，最小活跃星会降至 10 颗（$10\times 14 = 140 < 144$，容量比 $0.97\times$ 轻微不足），因此**不采用 $42^\circ\text{N}$**；
- 在 **$45^\circ\text{N}$** 和 **$47^\circ\text{N}$** 部署，容量冗余与多星覆盖均 100% 可行且零盲区。

### 3.2 第 1 层：动态 Walker 星座与轨道外推（`orbit.py`, `geometry.py`）

1. **解析轨道外推（ECI $\rightarrow$ ECEF）**：
   - 采用 Walker Delta $T/P/F$ 圆轨道解析传播模型，计入地球自转角速度 $\omega_e = 7.2921159\times 10^{-5}\text{ rad/s}$：
     $$\Omega_p = \frac{2\pi p}{P}, \quad u_{p,s}(t) = \frac{2\pi s}{S} + \frac{2\pi F p}{PS} + \omega_{\text{orb}}(t + t_0)$$
   - 每个 Episode 随机采样轨道起始偏移 $t_0 \in [0, T_{\text{orb}})$，避免 RL 策略过拟合特定轨道相位。
   - **精确 RVT 与仰角变化率**：基于 $T_{\text{epi}} + 180\text{ s}$ 预计算可见窗，直接查表获得每对 $(n, m)$ 精确到 1 s 的剩余可见时间 $\text{RVT}_{n,m}(t)$，并计算 $\dot{\theta}_{n,m}(t) = \theta_{n,m}(t) - \theta_{n,m}(t-1)$。

2. **实测校验后的三套星座场景配置（$H=515\text{ km}, i=53^\circ, \vartheta_{\min}=40^\circ$，`paper-simulation` 144 小区布局）**：

| 场景名称 | Walker 构型（总星数） | 区域中心纬度 | 活跃星数 $\|\mathcal{N}(t)\|$ | 系统容量冗余 ($14 N_{\text{act}}/144$) | 单小区可见星数 | 实验定位 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **紧容量主实验场景（推荐基准）** | **$72\times 22, F=17$（1584 星）** | **$45^\circ\text{N}$** | **12～18（均值 14.6）** | **$1.17\times \sim 1.75\times$** | **最小 2，均值 5.23** | **主训练与高负载竞争主对比**（零盲区且容量张力充分） |
| **常规冗余基准场景** | $72\times 22, F=17$（1584 星） | $47^\circ\text{N}$ | 16～21（均值 18.2） | $1.56\times \sim 2.04\times$ | 最小 3，均值 6.51 | 常规冗余下的切换频率与负载均衡对照 |
| **密集星座泛化场景** | $72\times 44, F=17$（3168 星） | $35^\circ\text{N}$ | 12～20（均值 17.2） | $1.17\times \sim 1.94\times$ | 最小 4，均值 6.60 | 对标付一阳（2026）与 `TAD-RL` 跨星座密度泛化 |

### 3.3 第 2 层：固定地面小区层与三层拓扑解耦

1. **球面坐标映射**：以区域中心 $(\text{lat}_0, \text{lon}_0)$ 构建局部东-北-天（ENU）切平面，将 `paper-simulation` 的 16 个 $3\times 3$ 局部波束簇（簇内间距 $25\text{ km}$、簇心间距 $165\text{ km}$、用户半径 $7.5\text{ km}$）投射到地球球面（$R_E=6371\text{ km}$），得到固定 ECEF 位置 $\mathbf{p}_m^{\text{cell}}$ 与局部天顶法向量 $\mathbf{u}_m = \mathbf{p}_m^{\text{cell}}/R_E$。
2. **精确仰角**：
   $$\theta_{n,m}(t) = \arcsin\left(\frac{(\mathbf{p}_n^{\text{sat}}(t) - \mathbf{p}_m^{\text{cell}}) \cdot \mathbf{u}_m}{\|\mathbf{p}_n^{\text{sat}}(t) - \mathbf{p}_m^{\text{cell}}\|}\right)$$
3. **拓扑矩阵严格约束**：
   - $V^{\text{geo}}_{n,m}(t) = \mathbb{I}(\theta_{n,m}(t) \ge 40^\circ)$；
   - 小区候选集合 $\mathcal{C}_m(t)$：取满足 $V^{\text{geo}}_{n,m}(t)=1$ 的 Top-$C_{\max}$（默认 $C_{\max}=6$，始终包含当前可见服务星）；
   - 关联矩阵 $Z_{n,m}(t) \in \{0,1\}$ 严格满足：
     $$Z_{n,m}(t) \le V^{\text{geo}}_{n,m}(t), \quad \sum_{n\in\mathcal{N}(t)} Z_{n,m}(t) \le 1, \quad \sum_{m=1}^{M} Z_{n,m}(t) \le J=14$$
   - 严谨性声明：单星关联仅避免同一小区被多星重复照射，**不消除相邻小区间的星内与星间同频干扰**（尤其是簇内 $25\text{ km}$ 紧邻小区间的强烈同频干扰）。

### 3.4 第 3 层：用户分布与时空动态业务模型

1. **用户位置与规模**：
   - 相对 LEO 卫星星下点速度（$\approx 7.0\text{ km/s}$），手持终端地面移动速度可忽略，故沿用文献与第 3 章标准做法：**用户在所属小区 $7.5\text{ km}$ 半径内均匀分布且位置固定**；
   - 用户总数默认对齐 `paper-simulation` 的 $U = 200$（敏感性测试覆盖 $200/300/400/500$）。
2. **空间非均匀热点 + 差异化 RT 占比**：
   - 保持全网每用户平均业务速率 $10\sim 50\text{ Mbps}$、全网加权平均 RT 占比 $20\%$ / BULK $80\%$ 与 `paper-simulation` 完全一致；
   - 在空间上构造 3～5 个高斯热点簇（约 25% 的热点小区聚集 55%～60% 用户）；同时让不同小区的 RT 占比在 $[5\%, 45\%]$ 之间分化（实时交互热点区 35%～45%，普通数据区 5%～15%，全网加权均值严格归一至 20%），确保不同小区在业务自适应 CVaR 风险水平 $\alpha_m(t)$ 上呈现显著差异。
3. **微观突发与宏观漂移双动态**：
   - **微观突发**：完整继承 `paper-simulation` 的 MMPP 突发过程（ON 概率 $0.16$、保持概率 $0.94$、ON 倍率 $5.5\times$、BULK 反相关系数 $0.6/1.08$）及 40 桶 RT 年龄队列 + BULK 队列；
   - **宏观热点漂移/拥塞冲击（主动切换核心验证工况）**：在 600 s Episode 中间（如 $t=200\text{ s}, 400\text{ s}$），允许热点中心迁移或指定区域到达率阶段性抬升 $1.5\times\sim 2.0\times$（持续 60～80 s），检验算法的主动负载迁移与防拥塞能力。

### 3.5 第 4 层：动态物理层、链路级时间相关阴影衰落与双维度波束负载

1. **路径损耗与 3GPP 贝塞尔方向图**：
   - 每 1 s 基于实时 ECEF 矢量更新斜距 $d_{n,u}(t)$、自由空间路径损耗 $\text{FSPL}=(4\pi d_{n,u}/\lambda)^2$，波束指向小区中心 $\mathbf{p}_m^{\text{cell}}$ 并计算离轴角 $\vartheta_{n,m,u}(t)$，直接复用 `paper-simulation` 的圆孔径贝塞尔方向图 `bessel_beam_gain()`。
2. **链路级 dB 域一阶高斯-马尔可夫 AR(1) 阴影衰落**（按 `(sat_id, cell_id)` 独立演进，拒绝每秒独立重采样或全星共享小区衰落）：
   $$L_{\text{sh,dB}}^{n,m}(t) = \rho_{\text{sh}} L_{\text{sh,dB}}^{n,m}(t-1) + \sqrt{1 - \rho_{\text{sh}}^2}\,\sigma_{\text{sh}}(\theta_{n,m}(t))\,\epsilon_{n,m}(t), \quad \epsilon_{n,m}(t) \sim \mathcal{N}(0, 1)$$
   - 相关系数 $\rho_{\text{sh}} = \exp(-T_H / \tau_{\text{corr}})$，默认 $\tau_{\text{corr}} = 10\text{ s}$（$\rho_{\text{sh}} \approx 0.9048$）；
   - $\sigma_{\text{sh}}(\theta)$ 参照 3GPP TR 38.811 随仰角在 $40^\circ \to 90^\circ$ 平滑递减（例如 $4.0\text{ dB} \to 2.0\text{ dB}$，均值约 $3.0\text{ dB}$ 对齐 `paper-simulation`；鲁棒性扫描 $2\sim 6\text{ dB}$）。
3. **双维度跳波束资源负载模型（避免循环依赖）**：
   - **名义预计速率**（用于切换层状态特征、负载计算与准入校验，不替代真实跳波束速率）：
     $$\hat{R}_{n,m}(t) = B_{\text{tot}} \log_2 \left( 1 + \frac{(P_{\text{tot}} / K_{\max})\, g_{n,m}(t)}{\kappa T_{\text{noise}} B_{\text{tot}} + \hat{I}_{n,m}(t)} \right)$$
     其中 $\hat{I}_{n,m}(t)$ 采用历史实测干扰滑窗统计或离线校准名义同频干扰模型。
   - **单小区波束时间需求比**（不截断至 1，完整保留热点过载压力）：
     $$d_{n,m}(t) = \frac{\bar{A}_m(t) + Q_m(t)/T_H}{\hat{R}_{n,m}(t) + \varepsilon}$$
   - **卫星整体波束资源负载率 $\rho_n(t)$ 与单小区最大过载比 $\zeta_n(t)$**：
     $$\rho_n(t) = \frac{1}{K_{\max}} \sum_{m=1}^{M} Z_{n,m}(t)\, d_{n,m}(t), \qquad \zeta_n(t) = \begin{cases} \max\limits_{m:\, Z_{n,m}(t)=1} d_{n,m}(t), & \sum_m Z_{n,m}(t) > 0 \\ 0, & \text{否则} \end{cases}$$

### 3.6 第 5 层：双时间尺度仿真闭环与两级底层服务模式

- **队列连续性与切换中断机制**：
  - 业务队列由网关/统一逻辑缓存维护，**切换瞬间绝不重置队列、不重置包等待年龄与 HOL**；
  - 当小区 $m$ 发生切换（$H_m(t)=1$）时，在中断窗口 $\tau_{\text{HO}}$（默认 $30\text{ ms}$，即 15 个 2 ms 时隙，敏感性扫描 $20\sim 50\text{ ms}$）内暂停该小区服务；期间新包继续到达、RT 队列继续老化，超出 $80\text{ ms}$ TTL 的包自然丢弃并计入真实 RT 丢包率 $\tilde{P}_m^{\text{RT,drop}}(t)$，**奖励中不再额外叠加重复丢包惩罚**。
  - **等效群体切换信令开销**：$C_m^{\text{HO}} = C^{\text{com}} + U_m C^{\text{ter}}$（归一化后用于切换成本惩罚）。

- **两级底层服务模式（兼顾训练效率与物理真实性）**：
  - **模式 A（训练期快速等效波束服务模型 `service_model.py`）**：因 600 s = 300,000 个 2 ms 时隙，纯 Python 逐时隙跑 PF 调度单 Episode 需数分钟，难以支撑数百轮 RL 训练。训练期采用**经 `paper-simulation` 真实 500-slot `USWGP-BH`/`MAPPO-APG-Local` 离线校准的等效波束服务与 RT 老化丢包代理模型**，将单 Episode 控制在 1 s 以内，并在 Phase 2 严格验证其吞吐量与 RT 丢包趋势误差。
  - **模式 B（最终闭环联合评估）**：直接挂载第 3 章 `paper-simulation` 冻结的 2 ms `MAPPO-APG-Local`（默认分布式）及 `MAPPO-CoopGNN`（高阶协同），每秒刷新活跃星信道矩阵 `H_sq` 与 `I_mat`（及可选 `E_mat`），完整运行 500 个 2 ms 微观时隙（实时星内/星间 SINR、PF RB 调度、40 桶 RT 老化与 `ho_remaining_slots` 中断暂停）。

- **每 1 s 宏观步统一执行顺序**：
  ```text
  t=0: 随机相位 t_0 -> 生成热点/用户 -> 计算 V_geo(0), 初始信道 -> 统一运行 ISA 得到 Z(0) -> 队列预热
  for t = 1 ... T_epi:
    1. update_topology(t)      : 推进 Walker 轨道 1 s -> 更新活跃星 N(t), V_geo(t), elev, d_elev, RVT, 链路级 AR(1) 阴影
    2. trigger_check(t)        : 计算强制触发 I_force 与主动触发 I_pro -> 得到 I_trig
    3. parallel_action(t)      : 触发小区基于局部异质图(含邻居 I_trig)并行选星；未触发小区强制 Stay
    4. admission_control(t)    : 卫星侧汇总请求 -> 按优先级(强制>主动, RT紧迫度降序)校验 J 与 rho_max -> 拒绝回退 -> 输出新 Z(t), F_m(t)
    5. execute_handover(t)     : 对变更关联的小区置位 ho_remaining_slots[m] = round(tau_HO / 2ms)，队列与 HOL 保持连续
    6. run_service_step(t)     : 模式 A (校准等效模型) 或 模式 B (500 x 2ms 冻结 MAPPO-BH) -> 统计真实吞吐、交付时延、RT 丢包率
    7. compute_reward_and_log(): 结算局部奖励 + O(1) 反事实协作奖励，记录 SMDP / 分层回放样本与全量指标
  ```

### 3.7 第四章完整仿真参数总表

| 参数类别 | 参数名称 | 推荐取值 | 来源/对齐说明 |
| :--- | :--- | :--- | :--- |
| **动态星座** | Walker Delta 构型 | 基准 $72\times 22, F=17$（1584 星）；泛化 $72\times 44$（3168 星） | 对标 Starlink 一期与付一阳（2026） |
| | 轨道高度 $H$ / 倾角 $i$ / 最小仰角 $\vartheta_{\min}$ | $515\text{ km}$ / $53^\circ$ / $40^\circ$（单星覆盖半径 $541.1\text{ km}$） | 高度与仰角 100% 对齐 `paper-simulation` |
| | 研究区域定位与小区布局 | 中心 $45^\circ\text{N}$（对照 $47^\circ\text{N}$），16 个 $3\times 3$ 簇（簇内 $25\text{ km}$，簇心 $165\text{ km}$，跨度 $545\text{ km}$） | 几何 100% 对齐 `paper-simulation`，活跃星均值 $14.6$（$47^\circ\text{N}$ 为 $18.2$） |
| **物理层** | 频段 / 带宽 / RB 数 / 噪声温度 | $2\text{ GHz}$ / $40\text{ MHz}$ / $80\text{ RB}$ / $300\text{ K}$ | 100% 对齐 `paper-simulation` |
| | 功率 / 发射增益 / 波束宽 / 接收增益 | $34\text{ dBW}$ / $34\text{ dBi}$（圆孔径贝塞尔）/ $3^\circ$ / $0\text{ dBi}$ | 100% 对齐 `paper-simulation` |
| | 链路级 AR(1) 阴影衰落 | 按 `(sat_id, cell_id)` 独立维护，$\tau_{\text{corr}}=10\text{ s}$，$\sigma_{\text{sh}}(\theta)\in[2.0, 4.0]\text{ dB}$（扫描 $2\sim 6\text{ dB}$） | 3GPP TR 38.811 动态扩展 |
| | 星上容量与波束约束 | 单星最大关联 $J=14$，最大激活波束 $K_{\max}=4$ | 100% 对齐 `paper-simulation` |
| **用户业务** | 用户规模 $U$ / 小区用户分布半径 | 默认 $200$（扫描 $200\sim 500$）/ $7.5\text{ km}$ | 对齐 `paper-simulation` + 热点聚集与漂移 |
| | 业务到达率与 RT/BULK 结构 | $10\sim 50\text{ Mbps/user}$，全网均值 RT $20\%$（小区间 $5\%\sim 45\%$） | 均值对齐 `paper-simulation` + 空间分化 |
| | MMPP / 包长 / RT TTL / PF 窗口 | ON $0.16$, stay $0.94$, $5.5\times$ / $1000\text{ bits}$ / $40\text{ slots}$ ($80\text{ ms}$) / $50\text{ slots}$ | 100% 对齐 `paper-simulation` |
| **切换控制** | 宏观步长 $T_H$ / Episode 时长 | $1\text{ s}$ / 训练 $300\text{ s}$，评估 $600\text{ s}$ | 对标付一阳（2026）并覆盖 4～6 次过境 |
| | 切换中断窗口 $\tau_{\text{HO}}$ | 默认 $30\text{ ms}$（15 微观时隙；敏感性扫 $20\sim 50\text{ ms}$） | 自然耦合至 80 ms RT TTL |
| | 触发阈值与滞回 | $T_{\text{cov}}=10\text{ s}$，$T_{\text{dwell}}^{\min}=6\sim 10\text{ s}$，$L_{\text{TTT}}=2\sim 3\text{ s}$ | 3GPP CHO + 预切换防乒乓 |
| | 准入软上限 / 候选星上限 | $\rho_{\max}=1.2$ / $C_{\max}=6$ | 允许短时排队，匹配均值 5.23 可见星 |

---

## 4. 双阶段自适应切换触发与卫星侧统一准入控制

### 4.1 受条件切换（CHO）启发的双阶段触发机制

设小区 $m$ 当前服务星为 $n_m(t-1)$，持续驻留时长为 $T_m^{\text{dwell}}(t)$：

1. **强制切换预警 $\mathbb{I}_m^{\text{force}}(t) = 1$**（覆盖/连通性兜底）：
   - 满足以下任一条件即触发：原星仰角 $\theta_{n_m,m}(t) < 40^\circ$、剩余可见时间 $\text{RVT}_{n_m,m}(t) \le T_{\text{cov}}$（默认 $10\text{ s}$）、或当前处于通信中断状态；
   - 若原星已不可服务（仰角 $< 40^\circ$ 或断链），动作掩码强制屏蔽 `Stay`。
2. **主动业务/链路预警 $\mathbb{I}_m^{\text{pro}}(t) = 1$**（拥塞/衰落前瞻避险）：
   - 前提：满足最小驻留时长 $T_m^{\text{dwell}}(t) \ge T_{\text{dwell}}^{\min}$（默认 $6\sim 10\text{ s}$，抑制乒乓切换）；
   - 条件：连续 $L_{\text{TTT}}$（默认 $2\sim 3\text{ s}$）满足以下任一滞回预警（避免对几十毫秒短时 MMPP 突发过度反应）：
     - **链路风险**：名义 SINR 低于 $\gamma_{\text{th}} - \Delta_{\text{hys}}$；
     - **业务风险**：$\rho_{n_m}(t) > \rho_{\text{th}}$，或 $\zeta_{n_m}(t) > \zeta_{\text{th}}$，或 RT 队头紧迫度 $u_m^{\text{RT}}(t) = \min\!\left(1, D_m^{\text{HOL}}(t) / D_{\max}^{\text{RT}}\right) > u_{\text{th}}$（或近期窗口内 RT 丢包率超阈值）。
3. **非触发强制保持**：当 $\mathbb{I}_m^{\text{trig}}(t) = \max(\mathbb{I}_m^{\text{force}}(t), \mathbb{I}_m^{\text{pro}}(t)) = 0$ 时，动作掩码仅允许 `Stay`，直接跳过 HGAT 与分位数网络前向推理，大幅降低在线算力与信令开销。

### 4.2 卫星侧统一准入控制（所有基准算法共用）

所有对比算法必须经过完全相同的并发冲突裁决流程：
`并行选星 -> 请求汇总 -> 资源校验与优先级排序 -> 拒绝回退 -> 关联更新`

- **硬约束**：可见性 $V^{\text{geo}}_{n,m}(t) = 1$ 且单星关联小区数 $\sum_m Z_{n,m}(t) \le J = 14$；
- **软约束**：波束资源负载率软上限 $\rho_{\max} > 1$（默认 $1.2$，允许短时队列缓冲）；
- **准入优先级**：`强制切换请求 > 主动切换请求`；同优先级内按 **RT 紧迫度 $u_m^{\text{RT}}(t)$ 降序**依次准入；对强制切换请求，只要未突破硬容量上限 $J$，允许短时超过 $\rho_{\max}$ 以避免人为制造断链；
- **拒绝回退逻辑**：
  - 主动切换被拒且原星仍可用 $\rightarrow$ 退回 `Stay`（不产生中断与切换成本）；
  - 强制切换被拒 $\rightarrow$ 按动作价值（或算法优先级）顺位尝试其余可行候选星；若全部失败则记通信中断 $F_m(t) = 1$。

---

## 5. POMDP/SMDP 建模、异质图 HGAT 与 TRA-HGDRL 核心算法

### 5.1 局部多维异质图观测 $o_m^t = \mathcal{G}_m^t$ 与 HGAT 编码

为每个触发决策的小区 $m$ 构建局部有向异质图（候选星 Top-$C_{\max}$，共享候选星的竞争邻小区 Top-$M_{\text{nbr}}$）：

- **卫星节点特征** $\mathbf{x}_n^{\text{sat}}$：相对位置/速度（或仰角/角速度投影）、预计波束负载率 $\rho_n(t)$、单小区最大过载比 $\zeta_n(t)$、关联小区数占比 $|\mathcal{M}_n|/J$、RT 业务承载负载占比；
- **小区节点特征** $\mathbf{x}_j^{\text{cell}}$：用户数 $U_j$、RT/BULK 到达率、RT 占比 $p_j^{\text{RT}}(t)$、RT 队头紧迫度 $u_j^{\text{RT}}(t)$、当前服务星 $\text{RVT}_j(t)$、**预切换触发标志 $\mathbb{I}_j^{\text{trig}}(t)$**（关键防多小区并发扎堆特征）；
- **三类有向边特征**：
  1. **覆盖关系边**（星 $\leftrightarrow$ 小区）：$\text{RVT}_{n,j}(t)$、仰角 $\theta_{n,j}(t)$ 与变化率 $\dot{\theta}_{n,j}(t)$、名义 SINR 余量、预计速率 $\hat{R}_{n,j}(t)$、需求比 $d_{n,j}(t)$；
  2. **服务关系边**（当前服务星 $\leftrightarrow$ 小区）：持续驻留时长 $T_j^{\text{dwell}}(t)$、历史吞吐量满足率、近期窗口 RT 丢包率；
  3. **资源竞争边**（小区 $\leftrightarrow$ 小区）：候选卫星 Jaccard 重合度 $e_{m,j}^{\text{comp}} = \frac{|\mathcal{C}_m \cap \mathcal{C}_j|}{|\mathcal{C}_m \cup \mathcal{C}_j|}$ 及其业务需求加权值。
- **2 层多关系 HGAT**：按三类关系拆分子图，在注意力系数计算中显式融合边特征（对标付一阳 2026 式(28)-(30)），将各关系子图聚合特征求和；堆叠 2 层即可捕获“本小区 $\rightarrow$ 候选星 $\rightarrow$ 共享该星的竞争邻小区”的二阶竞争拓扑，输出更新后的小区嵌入 $\mathbf{h}_m$ 与各候选卫星嵌入 $\mathbf{h}_n$。

### 5.2 排列等变逐候选星分位数打分头（核心工程改进）

为避免候选星顺序随轨道运动跳变导致固定输出槽位 `Linear(hidden, 1 + C_max)` 出现语义漂移，采用**共享参数的排列等变分位数头**：
- 对 `Stay` 动作：输入 $(\mathbf{h}_m \,\|\, \mathbf{h}_{n_{\text{cur}}} \,\|\, \mathbf{e}_{\text{serv}} \,\|\, \mathbf{1}_{\text{stay}})$，经共享网络输出 $N_q$ 个分位点 $\boldsymbol{\theta}(o_m^t, \text{Stay}) \in \mathbb{R}^{N_q}$；
- 对每个合法候选星 $n \in \mathcal{C}_m(t)$：输入 $(\mathbf{h}_m \,\|\, \mathbf{h}_n \,\|\, \mathbf{e}_{n,m}^{\text{cov}} \,\|\, \mathbf{0}_{\text{stay}})$，经同一共享网络输出 $\boldsymbol{\theta}(o_m^t, n) \in \mathbb{R}^{N_q}$；
- 非法或填充槽位由动态 `action_mask` 屏蔽。该设计对候选星排列严格等变，且天然支持候选星数量动态变化。

### 5.3 事件触发 SMDP 跨步回报与分层回放

考虑到小区切换后会进入 $T_{\text{dwell}}^{\min}=6\sim 10\text{ s}$ 的强制驻留期，单步 1 s TD 难以把未来几十秒的拥塞/断链长尾风险迅速归因到本次选星决策，因此采用**事件触发型 Semi-MDP（SMDP）跨步转换**（支持最大截断步长 $K_{\text{step}}$ 如 20 s）：
- 当小区 $m$ 在时刻 $t$ 触发决策并执行 $a_m(t)$ 后，若其下一次触发时刻（或截断步/Episode 终止）为 $t' = t + \Delta t$，则构造跨步经验元组：
  $$\left(o_m^t,\; a_m^t,\; R_m^{t\to t'},\; o_m^{t'},\; \text{mask}_m^{t'},\; \Delta t,\; \text{done}\right), \qquad R_m^{t\to t'} = \sum_{k=0}^{\Delta t - 1} \gamma^k r_m(t+k)$$
- 分布 Bellman 目标中采用跨步折扣 $\gamma^{\Delta t}$（$\gamma \in \{0.99, 0.995\}$），并严格在下一状态使用存储的 $\text{mask}_m^{t'}$ 计算目标动作，防止非法动作过估计；同时对“强制触发 / 主动触发 / Stay”样本采用分层回放采样，避免常规驻留样本淹没关键切换经验。

### 5.4 局部奖励 + 负均方负载 $O(1)$ 解析近似反事实协作奖励

每 1 s 步进的即时奖励定义为：
$$r_m(t) = r_m^{\text{local}}(t) + \eta\, r_m^{\text{coop}}(t)$$

1. **局部业务奖励**（各项归一化，丢包仅按真实超时计一次，不重复惩罚）：
   $$r_m^{\text{local}}(t) = w_1 \tilde{R}_m(t) - w_2 \tilde{P}_m^{\text{RT,drop}}(t) - w_3 H_m(t)\tilde{C}_m^{\text{HO}} - w_4 F_m(t)$$
2. **负均方负载 $O(1)$ 解析近似反事实协作奖励**：
   - 定义活跃卫星集 $\mathcal{N}(t)$ 上的全局负载均衡与效率势函数为**负均方负载**：
     $$G(\boldsymbol{\rho}) = -\frac{1}{N_t}\sum_{n \in \mathcal{N}(t)} \rho_n^2 = -\left(\bar{\rho}^2 + \operatorname{Var}(\boldsymbol{\rho})\right)$$
     > **相较纯负方差 $-\operatorname{Var}(\boldsymbol{\rho})$ 的优势**：由于同一小区在原星与目标星的波束需求比满足 $d_{n_{\text{old}},m} \neq d_{n_{\text{new}},m}$，纯负方差存在“将小区切到链路很差的空星、通过抬高平均负载来缩小方差”的博弈漏洞；而负均方负载同时惩罚**均值过高（链路低效）**和**方差过大（负载失衡）**，且与第 3 章 `ISA` 的优化目标 $\sum L_n^2$ 在数学形式上完全呼应。
   - 反事实差值奖励（Difference Reward）：
     $$r_m^{\text{coop}}(t) = G(\boldsymbol{\rho}(\mathbf{a})) - G(\boldsymbol{\rho}(a_m^{\text{ref}}, \mathbf{a}_{-m}))$$
     其中参考动作 $a_m^{\text{ref}}$ 在原星可用时取 `Stay`，原星不可用（强制切换）时取移除小区 $m$ 负载的空参考态；通过增量更新原星与目标星的 $\rho_{n_{\text{old}}} - d_{n_{\text{old}},m}/K_{\max}$ 与 $\rho_{n_{\text{new}}} + d_{n_{\text{new}},m}/K_{\max}$，仅需 $O(1)$ 即可精确求得。

### 5.5 QR-DQN 分位数学习与业务自适应近似 CVaR 选星

1. **分位数回归损失（Quantile Huber Loss）**：
   - 每个动作输出 $N_q = 32$（或 $64$）个分位点 $\{\theta_1, \dots, \theta_{N_q}\}$，中点分位概率 $\hat{\tau}_i = \frac{2i-1}{2N_q}$；
   - 给定跨步 TD 目标分位点 $y_j = R_m^{t\to t'} + (1-\text{done})\,\gamma^{\Delta t}\,\theta_j^{\text{target}}(o_m^{t'}, a^*)$，最小化所有 $(i, j)$ 对的 Quantile Huber Loss $\mathcal{L}_\kappa(\theta_i, y_j)$。
2. **业务自适应风险水平 $\alpha_m(t)$**：
   $$\alpha_m(t) = \operatorname{clip}\Big(1 - \beta_1\, p_m^{\text{RT}}(t) - \beta_2\, u_m^{\text{RT}}(t),\; \alpha_{\min},\; 1\Big)$$
   高 RT 占比、高队头紧迫度的小区自动降低 $\alpha_m(t)$（更关注左尾低分位风险，规避拥塞与丢包）；低 RT 占比的 BULK 主导小区 $\alpha_m(t) \to 1$（趋近风险中性，最大化吞吐）。
3. **非整数 $\alpha N_q$ 连续线性插值近似 CVaR**：
   将分位点升序排列为 $\theta_{(1)} \le \dots \le \theta_{(N_q)}$，令 $k = \lfloor \alpha N_q \rfloor$，$\delta = \alpha N_q - k$：
   $$\widehat{\text{CVaR}}_{\alpha}(o_m^t, a) = \frac{1}{\alpha N_q} \left( \sum_{i=1}^{k} \theta_{(i)}(o_m^t, a) + \delta \cdot \theta_{(k+1)}(o_m^t, a) \right)$$
4. **两阶段渐进训练与可选 Twin QR-DQN 规范**：
   - **Stage 1（风险中性预热）**：令 $\alpha_m \equiv 1$ 预热训练 QR-DQN，先稳定学准全分布分位点；
   - **Stage 2（风险敏感微调）**：线性退火开启业务自适应 $\alpha_m(t)$ 选星与微调；
   - 若实测存在过估计并开启可选的 `Twin QR-DQN`，必须按保守准则**整套选用单个目标网络的完整分位数分布**（比较两套目标网络的均值/CVaR 后取较小者的整组分位点），**严禁逐分位点取 $\min$**（避免人为左偏扭曲分布）。

---

## 6. 代码目录架构与核心扩展接口规范

### 6.1 独立目录规划（保留 `paper-simulation` 第 3 章文件完整不变）

```text
leo-bh-mappo/
├── config.py                    # [第3章 paper-simulation] 原始参数（保持不动）
├── association.py               # [第3章 paper-simulation] ISA / NLB / MEA / MMTG
├── env.py                       # [第3章 paper-simulation] 静态快照跳波束环境
├── coordination.py              # [第3章 paper-simulation] 跨星邻居干扰与 E_mat
├── models.py                    # [第3章 paper-simulation] 排列等变 MAPPOActor + IterativeCoopPowerGNN
├── heuristics.py                # [第3章 paper-simulation] P-BH / QLF-BH / USWGP-BH
├── train.py / evaluate.py       # [第3章 paper-simulation] 训练与评估脚本
├── checkpoints/                 # [第3章 paper-simulation] 冻结权重（mappo-apg-local, mappo-coop-gnn 等）
│
├── docs/
│   └── Chapter4_TRA_HGDRL_Design.md   # 本技术方案文档
│
└── handover/                    # [第4章新增独立包]
    ├── __init__.py
    ├── config.py                # 第4章星座、触发、准入、HGAT、QR-DQN 超参数
    ├── orbit.py                 # Walker 星座 ECI->ECEF 1s 解析传播与轨道缓存
    ├── geometry.py              # 144 小区球面映射（16x3x3 簇）、仰角、V_geo、精确 RVT、候选星筛选
    ├── channel.py               # 实时斜距 FSPL、贝塞尔方向图、链路级 AR(1) 阴影衰落与 H_sq/I_mat 更新
    ├── traffic.py               # 非均匀多热点、差异化 RT 占比、MMPP 突发与热点漂移
    ├── service_model.py         # 模式 A 校准等效波束服务模型 + 模式 B 真实 2ms 调度适配器
    ├── trigger.py               # 双阶段 CHO 触发器（强制 + 滞回主动）
    ├── admission.py             # 卫星侧统一并发准入控制与拒绝回退
    ├── dynamic_env.py           # 1s 宏观切换环境主类（含三核心接口与队列连续演化）
    ├── graph.py                 # 局部多维异质图构造（卫星/小区节点、三类边、邻居触发标志）
    ├── reward.py                # 局部业务奖励 + 负均方负载 O(1) 反事实协作奖励
    ├── hgat_qrdqn.py            # 2层 HGAT + 排列等变 QR-DQN + 业务自适应近似 CVaR
    ├── replay_buffer.py         # SMDP 跨步转换与分层经验回放池（强制深拷贝 + mask 对齐）
    ├── baselines.py             # Max-RVT / ISA-HO / Swap-HO / Cell-RL / MA-HGDQN / TAD-RL
    ├── mappo_adapter.py         # 第3章冻结 MAPPOActor 动态适配（sat_id GRU + F_rich/I_mat/可选 E_mat）
    ├── train.py                 # 两阶段训练脚本（风险中性预热 -> CVaR 微调）
    ├── evaluate.py              # 全量对比、消融、敏感性与冻结 MAPPO 联合闭环评估
    └── tests/                   # 各 Phase 单元测试与校准验证脚本
```

### 6.2 三大核心底层环境接口规范

Phase 1～2 最优先落地并单测通过以下三个核心接口，保证队列连续性、信道动态更新与关联约束在任意时刻严格成立：

1. `update_topology(t: int) -> TopologySnapshot`
   - 推进星座至宏观步 `t`，更新活跃卫星列表 `active_sat_ids`、三维位置/速度、仰角 `elev_rad`、仰角变化率 `elev_rate`、几何可见矩阵 `V_geo`、精确剩余可见时间 `rvt_s`、链路级 AR(1) 阴影衰落、名义链路质量以及星内耦合矩阵 `I_mat`。
2. `update_association(Z_new: np.ndarray, active_sat_ids: list[int]) -> AssocDiff`
   - 断言校验硬约束（$Z \le V^{\text{geo}}$、单星关联 $\le J$、单小区关联 $\le 1$）；更新 `cell_sat`（未连接为 `-1`）、各活跃卫星的候选小区列表 `cand_cells[sat_id]` 与 `action_mask`，并初始化/清理进出区域卫星的 GRU 隐状态（若启用 `mappo-coop-gnn` 则同步刷新 `E_mat`）。
3. `execute_handover(cell_id: int, target_sat_id: int, tau_ho_ms: float) -> None`
   - 触发切换时仅置位 `ho_remaining_slots[cell_id] = round(tau_ho_ms / 2.0)` 并记录切换次数、乒乓状态与信令成本；**严禁清零 `rt_buckets`、`bulk_queue` 与 `hist_tput`**。

---

## 7. 基准算法、消融实验与评价指标体系

### 7.1 对比基准矩阵（全部共用同一物理层、队列、中断与卫星准入机制）

1. **`Max-RVT`**：选择可行候选星中剩余可见时间最长者（经典几何基线）；
2. **`ISA-HO`**：周期性动态重跑第 3 章改进模拟退火关联（跨章对照基线，修复容量与中断容错）；
3. **`Swap-HO`**：适配 Sun 等（IEEE TCOM 2025）的跳波束主动交换切换算法；
4. **`Cell-RL`**：适配 Wang 等（IEEE LCOMM 2026）的小区级强化学习切换基准（候选筛选 + Action Mask）；
5. **`MA-HGDQN`**：适配付一阳等（《计算机学报》2026）的异质图多智能体 DQN（小区级适配版，标准 DQN 期望决策）；
6. **`TAD-RL`（可选）**：适配 Liu 等（IEEE WCL 2025）的高斯分布双 Critic 切换基准；
7. **`TRA-HGDRL`（本文提出）**：完整模型。

### 7.2 消融实验矩阵（逐一单模块剥离）

- `TRA-HGDRL`（完整模型）
- `w/o Traffic`：去除节点/边中的业务与队列特征、触发器去除业务条件、风险系数改为固定值；
- `w/o HGAT`：去除异质图编码，退化为平铺局部特征 MLP；
- `w/o DistRL`：去除分位数分布建模，退化为标准 Double DQN（同步失去 CVaR 能力）；
- `w/o Risk`：保留 `QR-DQN`，强制 $\alpha_m(t) \equiv 1$（风险中性期望决策，单独验证近似 CVaR 增益）；
- `w/o Trigger`：去除双阶段自适应触发，退化为仅 $\text{RVT} \le 10\text{ s}$ 或断链强制触发。

### 7.3 评价指标与敏感性实验

- **核心指标（8 项）**：累计切换次数、乒乓切换率（$T_{\text{pp}} = 10\text{ s}$ 内切回原星占比）、通信中断率、**RT 超时丢包率**、**RT 实际交付平均时延（ms）**、用户有效吞吐量（Mbps）、跨星波束负载方差与 Jain 公平指数、单次在线决策耗时（ms）。
- **敏感性与泛化测试**：
  1. 业务负载与规模扫描：用户数 $U \in \{200, 300, 400, 500\}$，每用户业务速率 $10\sim 50\text{ Mbps}$；
  2. 宏观热点漂移 / 突发拥塞动态响应曲线；
  3. 阴影衰落强度 $\bar{\sigma}_{\text{sh}} \in \{2, 3, 4, 5, 6\}\text{ dB}$ 鲁棒性扫描；
  4. 切换中断窗口 $\tau_{\text{HO}} \in \{20, 30, 40, 50\}\text{ ms}$ 敏感性扫描；
  5. 风险参数 $(\beta_1, \beta_2, \alpha_{\min})$ 帕累托权衡分析；
  6. 跨星座密度泛化（紧容量 $72\times 22$ @ $45^\circ\text{N}$、常规 $47^\circ\text{N}$、密集 $72\times 44$ @ $35^\circ\text{N}$）；
  7. **最终系统级闭环验证**：所有切换算法直接挂载第 3 章 `paper-simulation` 冻结 2 ms `MAPPO-APG-Local`（及可选 `MAPPO-CoopGNN`）跑端到端联合评估。

---

## 8. 分阶段开发与验收路线图（Phase 0 ~ Phase 6）

严格按序推进，**每阶段必须通过单元测试/冒烟实验后方可进入下一阶段**：

- **Phase 0（`paper-simulation` 基线审计与冻结 `MAPPOActor` 动态压测）**：
  - 实现 `mappo_adapter.py`（支持 `mappo-apg-local` 与 `mappo-coop-gnn`，按 `sat_id` 管理 GRU 隐状态，动态构建 `F_rich`/`I_mat`/可选 `E_mat`）；
  - 编写压测脚本验证冻结权重在动态 `N_act`、关联变更、单星关联 $<4$、候选槽位打乱与突发负载偏移下的稳定性与推理耗时，输出基线压测报告。
- **Phase 1（动态 Walker 星座与 1 s 覆盖信道环境）**：
  - 实现 `orbit.py`、`geometry.py`、`channel.py`（16 个 $3\times 3$ 簇球面映射 + 链路级 AR(1) 阴影）；
  - 验收：运行 600 s 覆盖自检，验证三套星座配置下 $|\mathcal{N}(t)|$、单小区可见星数、精确 RVT 分布与链路级 AR(1) 自相关系数符合理论值。
- **Phase 2（切换执行、双维度负载、统一准入与等效服务模型校准）**：
  - 实现 `traffic.py`、`trigger.py`、`admission.py`、`service_model.py`、`dynamic_env.py`；
  - 验收：单测验证 `Z <= V_geo`、单星 $\le J$、队列与 HOL 切换不重置、$\tau_{\text{HO}}$ 中断自然触发 RT 老化丢包；运行模式 A 与 `paper-simulation` 真实 500-slot `USWGP-BH`/`MAPPO-APG-Local` 校准对比，确保吞吐与 RT 丢包趋势误差在可接受范围内且单 Episode $< 1\text{ s}$。
- **Phase 3（传统基准算法与物理直觉检验）**：
  - 实现 `Max-RVT`、`ISA-HO`、`Swap-HO`，跑通 600 s 基准与热点漂移仿真，验证 8 项指标符合通信直觉。
- **Phase 4（基础 RL 与异质图编码验证）**：
  - 实现 `graph.py`、`reward.py`、`replay_buffer.py`、共享参数 `Double DQN`（`Cell-RL`）与 `MA-HGDQN`；验证 SMDP 回放收敛性与 HGAT 拓扑表征增益。
- **Phase 5（核心算法 `TRA-HGDRL` 实现与两阶段训练）**：
  - 实现 `hgat_qrdqn.py`（排列等变分位数头 + Quantile Huber Loss + 业务自适应近似 CVaR）与 `train.py`；
  - 执行“风险中性预热 $\rightarrow$ 业务自适应 CVaR 退火微调”，绘制分位数分布左尾可视化，验证是否需开启可选 `Twin QR-DQN`。
- **Phase 6（全量对比、消融、敏感性与冻结 `MAPPO-BH` 端到端联合闭环验证）**：
  - 跑齐全部 7 基准、5 消融、敏感性/跨星座泛化实验，以及挂载 `paper-simulation` 冻结 2 ms `MAPPO-APG-Local`（及 `MAPPO-CoopGNN`）的最终系统级联合测试，导出论文全部图表与统计置信区间数据。
