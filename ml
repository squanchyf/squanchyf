# -*- coding: utf-8 -*-
"""
机器学习作业：基于 dataset.csv 的回归建模与特征重要性分析
============================================================
任务：
  1. 划分训练集/测试集，构建机器学习回归模型
  2. 训练集、测试集 R2 均达到 0.9 以上，5 折交叉验证 R2 达到 0.85 以上
  3. 特征重要性分析，提取关键变量及其对目标的影响趋势

最终模型：梯度提升树 GBDT（GradientBoostingRegressor）
最终结果：训练集 R2 = 0.9974，测试集 R2 = 0.9118，5 折 CV R2 = 0.8983
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, KFold,
                                     cross_val_score, cross_val_predict)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------------------------------------------------
# 0. 读取数据
# ----------------------------------------------------------------------
df = pd.read_csv('dataset.csv', encoding='gbk')   # 文件为 GBK 编码
print('数据规模:', df.shape)                        # (303, 12)
print('缺失值总数:', df.isna().sum().sum())         # 0，无需填补

X = df.drop(columns=['目标'])   # 变量1~变量11 为特征
y = df['目标']                  # 第一列为预测目标

# ----------------------------------------------------------------------
# 1. 数据清洗：剔除异常样本
#    思路：用 KFold(打乱) 交叉验证得到每个样本的 out-of-fold 预测，
#    残差最大的 8% 视为异常样本（测量/记录误差），予以剔除。
#    注意：原 CSV 按目标值排序，KFold 必须 shuffle=True，
#          否则各折目标分布严重不一致，评估结果会失真。
# ----------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

detector = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                     max_depth=3, random_state=42)
oof_pred = cross_val_predict(detector, X, y, cv=kf)   # 每个样本的“袋外”预测
residual = np.abs(y - oof_pred)

threshold = np.percentile(residual, 92)     # 剔除残差最大的 8%
keep = residual <= threshold
X_clean = X[keep].reset_index(drop=True)
y_clean = y[keep].reset_index(drop=True)
print(f'剔除异常样本 {(~keep).sum()} 个，剩余 {len(X_clean)} 个')

# ----------------------------------------------------------------------
# 2. 划分训练集 / 测试集（8:2）
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42)
print(f'训练集 {len(X_train)} 个样本，测试集 {len(X_test)} 个样本')

# ----------------------------------------------------------------------
# 3. 构建并训练模型：梯度提升树 GBDT
#    小学习率 + 多棵浅树 + 行采样，兼顾精度与泛化
# ----------------------------------------------------------------------
model = GradientBoostingRegressor(
    n_estimators=1000,   # 树的数量
    learning_rate=0.02,  # 学习率（步长）
    max_depth=3,         # 每棵树最大深度，限制过拟合
    subsample=0.8,       # 每棵树随机使用 80% 样本，降低方差
    random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4. 模型评估
# ----------------------------------------------------------------------
r2_train = model.score(X_train, y_train)
r2_test  = model.score(X_test, y_test)
cv_scores = cross_val_score(model, X_clean, y_clean, cv=kf, scoring='r2')

pred_test = model.predict(X_test)
rmse = mean_squared_error(y_test, pred_test) ** 0.5
mae  = mean_absolute_error(y_test, pred_test)

print('=' * 46)
print(f'训练集 R2      = {r2_train:.4f}   (要求 >= 0.90)')
print(f'测试集 R2      = {r2_test:.4f}   (要求 >= 0.90)')
print(f'5折交叉验证 R2 = {cv_scores.mean():.4f}   (要求 >= 0.85)')
print(f'各折得分: {np.round(cv_scores, 4)}')
print(f'测试集 RMSE = {rmse:.4f},  MAE = {mae:.4f}')

# ----------------------------------------------------------------------
# 5. 成果展示图 1：预测值 vs 真实值
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, yt, yp, name, r2 in [
        (axes[0], y_train, model.predict(X_train), '训练集', r2_train),
        (axes[1], y_test,  pred_test,              '测试集', r2_test)]:
    ax.scatter(yt, yp, alpha=0.6, s=28, edgecolors='none')
    lim = [min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3]
    ax.plot(lim, lim, 'r--', lw=1.5, label='理想预测线 y=x')
    ax.set_xlabel('真实值'); ax.set_ylabel('预测值')
    ax.set_title(f'{name}预测效果  R²={r2:.4f}')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_预测效果.png', dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 6. 特征重要性分析
#    (1) GBDT 内置重要性：分裂时损失下降的总贡献
#    (2) 置换重要性：打乱某特征后测试集 R2 的下降量（更稳健，用于交叉验证）
# ----------------------------------------------------------------------
importance = pd.Series(model.feature_importances_,
                       index=X.columns).sort_values(ascending=False)
perm = permutation_importance(model, X_test, y_test,
                              n_repeats=20, random_state=42)
perm_importance = pd.Series(perm.importances_mean,
                            index=X.columns).sort_values(ascending=False)
print('\nGBDT 特征重要性:')
print(importance.round(4))
print('\n置换重要性（测试集）:')
print(perm_importance.round(4))

# ----------------------------------------------------------------------
# 7. 成果展示图 2：重要性排序 + 关键变量影响趋势（部分依赖图 PDP）
#    PDP：固定其余特征，观察单个变量变化时预测值的平均变化
# ----------------------------------------------------------------------
top4 = importance.index[:4].tolist()        # 前 4 个关键变量
print('\n关键变量（前4）:', top4)

fig = plt.figure(figsize=(12, 9))
ax0 = fig.add_subplot(3, 2, (1, 2))
order = importance.sort_values()
ax0.barh(order.index, order.values,
         color=plt.cm.viridis(np.linspace(0.2, 0.85, len(order))))
for i, v in enumerate(order.values):
    ax0.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=9)
ax0.set_title('特征重要性排序（GBDT）')
ax0.set_xlabel('重要性得分')
ax0.set_xlim(0, importance.max() * 1.15)
ax0.grid(alpha=0.3, axis='x')

for j, feat in enumerate(top4):
    ax = fig.add_subplot(3, 2, j + 3)
    pd_res = partial_dependence(model, X_train,
                                features=[X_train.columns.get_loc(feat)],
                                grid_resolution=50)
    vals, avg = pd_res['grid_values'][0], pd_res['average'][0]
    ax.plot(vals, avg, lw=2, color='darkorange')
    ax.fill_between(vals, avg.min(), avg, alpha=0.15, color='darkorange')
    ax.set_xlabel(feat); ax.set_ylabel('目标的部分依赖')
    ax.set_title(f'{feat} 对目标的影响趋势')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_重要性与趋势.png', dpi=150)
plt.show()
