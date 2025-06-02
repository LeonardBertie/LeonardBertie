
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


st.title("🌸 机器学习入门体验：点击按钮感受模型训练与预测")


@st.cache_data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data

df, data = load_data()


if st.checkbox("查看原始数据"):
    st.write(df.head())


st.subheader("🎯 输入特征值进行预测")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("花萼长度 (cm)", value=5.1)
    sepal_width = st.number_input("花萼宽度 (cm)", value=3.5)
with col2:
    petal_length = st.number_input("花瓣长度 (cm)", value=1.4)
    petal_width = st.number_input("花瓣宽度 (cm)", value=0.2)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]


if st.button("🚀 训练模型"):
    X = df[data.feature_names]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ 模型训练完成！测试准确率: {acc:.2f}")
    st.session_state.model = model  # 保存模型


if st.button("🔍 使用模型进行预测"):
    if "model" not in st.session_state:
        st.warning("⚠️ 请先点击上面的 '训练模型' 按钮")
    else:
        prediction = st.session_state.model.predict(input_data)[0]
        pred_class = data.target_names[prediction]
        st.success(f"🌟 预测结果：模型判断这是 → {pred_class}")
