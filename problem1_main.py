# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:38:57 2026

@author: Lenovo
"""

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# ===================== 参数设置 =====================
NUM_USERS = 16
NUM_RESOURCE_BLOCK = 50  # 总资源块，约束：所有用户分配之和=50
# 用户分类：2 URLLC,4 eMBB,10 mMTC
user_type = [0]*2 + [1]*4 + [2]*10  # 0:URLLC  1:eMBB  2:mMTC

# SLA QoS阈值
SLA = {
    0: {"rate_min": 12, "delay_max": 8},    # URLLC
    1: {"rate_min":6, "delay_max":20},      # eMBB
    2: {"rate_min":2, "delay_max":100}      # mMTC
}

# 生成信道增益
np.random.seed(42)
large_scale = np.random.exponential(scale=2, size=NUM_USERS)
small_scale = np.random.rayleigh(scale=1, size=NUM_USERS)
channel_gain = large_scale * small_scale

B = 180e3  # 每个资源块带宽 Hz
SNR0 = 10

def calc_rate(rb_alloc, ch_gain):
    """根据分配资源块，计算每个用户传输速率，香农公式"""
    rate_list = []
    for i in range(NUM_USERS):
        rb = rb_alloc[i]
        if rb <= 0:
            rate_list.append(0.0)
            continue
        snr = SNR0 * ch_gain[i]
        r = rb * B * np.log2(1 + snr) / 1e6
        rate_list.append(r)
    return np.array(rate_list)


def calc_QoS(rb_alloc):
    """计算全部用户QoS总和，不满足SLA施加惩罚"""
    rate_arr = calc_rate(rb_alloc, channel_gain)
    total_qos = 0.0
    for i in range(NUM_USERS):
        ut = user_type[i]
        r_i = rate_arr[i]
        sla_min_rate = SLA[ut]["rate_min"]

        q = 0.0
        if r_i >= sla_min_rate:
            q = np.clip(r_i / sla_min_rate, 0, 1.0)
        else:
            q = -1.0  # 不满足业务最低速率，惩罚
        total_qos += q
    return total_qos


def objective(rb_alloc):
    """
    scipy差分进化是求最小值，所以返回 -QoS，等价最大化QoS
    额外加入惩罚：资源块总和偏离50的时候强惩罚
    """
    sum_rb = np.sum(rb_alloc)
    qos_val = calc_QoS(rb_alloc)
    penalty = 100 * np.abs(sum_rb - NUM_RESOURCE_BLOCK)
    return - qos_val + penalty


# ===================== 变量约束：每个用户分配0~50，整数 =====================
bounds = [(0, NUM_RESOURCE_BLOCK) for _ in range(NUM_USERS)]
integer_mask = [True]*NUM_USERS

print("开始求解混合整数规划资源分配优化...")
result = differential_evolution(
    objective,
    bounds=bounds,
    seed=42,
    maxiter=200,
    popsize=10,
    integrality=integer_mask
)

best_rb = np.rint(result.x).astype(int)
best_qos = calc_QoS(best_rb)
sum_best_rb = np.sum(best_rb)
rate_result = calc_rate(best_rb, channel_gain)

print("=======最优分配结果=======")
print(f"分配资源块向量：{best_rb}")
print(f"资源块总和：{sum_best_rb} (目标{NUM_RESOURCE_BLOCK})")
print(f"总体QoS：{best_qos:.4f}")
print("各用户速率(Mbps):")
for idx,(t,rb,r) in enumerate(zip(user_type, best_rb, rate_result)):
    t_name = {0:"URLLC",1:"eMBB",2:"mMTC"}[t]
    print(f"user{idx:2d} [{t_name:5s}] RB={rb:2d} Rate={r:.3f} Mbps")


# ==========绘图==========
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.bar(range(NUM_USERS), best_rb)
plt.xlabel("User index")
plt.ylabel("Allocated Resource Block")
plt.title("Resource‑Block Allocation per User")
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.bar(range(NUM_USERS), rate_result)
plt.xlabel("User index")
plt.ylabel("Rate Mbps")
plt.title("User Transmission Rate")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./simulation_overview.png", dpi=150)
plt.show()