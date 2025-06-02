import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("🌸 鸢尾花分类器 - Iris Flower Classifier")

st.write("请输入以下花的特征，我们将告诉你是哪一类鸢尾花：")

# 用户输入
sepal_length = st.number_input("花萼长度 (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("花萼宽度 (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("花瓣长度 (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("花瓣宽度 (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("预测花的类别"):
    # 加载数据与训练模型
    iris = load_iris()
    clf = RandomForestClassifier()
    clf.fit(iris.data, iris.target)

    # 构建预测数据
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = clf.predict(input_data)
    pred_species = iris.target_names[prediction][0]

    st.success(f"✅ 预测结果：这是 **{pred_species}**")


