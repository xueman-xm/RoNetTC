"""
data: 2024/11/21-13:59
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt



# 设置随机种子并生成示例数据
np.random.seed(42)
X = np.random.rand(100, 64)  # 100 个样本，64 个特征
y = np.random.randint(0, 2, 100)  # 二分类标签

# 训练随机森林模型
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 检查模型输出的概率矩阵
sample_idx = 0  # 选择第 0 个样本
sample = X[sample_idx].reshape(1, -1)
predicted_probabilities = model.predict_proba(sample)
print(f"Predicted probabilities for sample {sample_idx}: {predicted_probabilities}")

# 初始化 LIME 解释器
explainer = LimeTabularExplainer(
    training_data=X,
    mode="classification",
    feature_names=[f"Feature {i+1}" for i in range(X.shape[1])],
    class_names=["Class 0", "Class 1"],
    discretize_continuous=True,
)

# 检查预测类别
predicted_label = np.argmax(predicted_probabilities, axis=1)[0]
print(f"Predicted label (highest probability): {predicted_label}")

# 强制生成所有类别的解释
exp = explainer.explain_instance(
    data_row=sample.flatten(),
    predict_fn=model.predict_proba,
    labels=[0, 1],  # 指定生成所有类别的解释
    num_features=10
)

# 检查可用标签
available_labels = exp.available_labels()
print(f"Available labels: {available_labels}")
print(f"exp.top_labels: {exp.top_labels}")

# 强制使用存在的类别
for label in available_labels:
    print(f"Explanation for label {label}:")
    print(exp.as_list(label=label))

# 绘制类别 0 的解释图
if 0 in available_labels:
    exp.as_pyplot_figure(label=0)
    plt.title("Explanation for label 0")
    plt.show()

# 绘制类别 1 的解释图
if 1 in available_labels:
    exp.as_pyplot_figure(label=1)
    plt.title("Explanation for label 1")
    plt.show()