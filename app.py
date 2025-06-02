import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="机器学习体验", layout="centered")
st.title("🌸 机器学习入门体验：导入数据 → 选择模型 → 训练 → 预测")

# 加载数据按钮
if st.button("📥 导入数据"):
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    st.session_state.df = df
    st.session_state.feature_names = data.feature_names
    st.session_state.target_names = data.target_names
    st.success("✅ 数据已成功导入！")

# 显示数据
if "df" in st.session_state:
    df = st.session_state.df

    st.subheader("🧾 原始数据")
    st.dataframe(df.head())

    st.markdown(f"""
    - 🧬 特征数量: `{len(st.session_state.feature_names)}`
    - 🧪 样本数量: `{df.shape[0]}`
    - 🎯 类别名称: `{list(st.session_state.target_names)}`
    """)

    st.subheader("🔧 设置训练集比例和模型")

    # 用户设置训练集比例
    test_size = st.slider("测试集占比", min_value=0.1, max_value=0.9, value=0.2, step=0.05)

    # 用户选择模型
    model_choice = st.selectbox("选择模型", ["逻辑回归 LogisticRegression", "K近邻 KNN", "支持向量机 SVM"])

    # 模型训练按钮
    if st.button("🚀 训练模型"):
        X = df[st.session_state.feature_names]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 根据选择构建模型
        if model_choice.startswith("逻辑回归"):
            model = LogisticRegression(max_iter=200)
        elif model_choice.startswith("K近邻"):
            model = KNeighborsClassifier()
        elif model_choice.startswith("支持向量机"):
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ 模型训练完成！测试准确率: {acc:.2f}")
        st.session_state.model = model
        st.session_state.X_columns = X.columns

    # 预测区域
    if "model" in st.session_state:
        st.subheader("🔍 使用模型进行预测")
        cols = st.columns(4)
        user_input = []
        for i, col in enumerate(cols):
            with col:
                val = st.number_input(f"{st.session_state.X_columns[i]}", value=5.0, format="%.2f")
                user_input.append(val)

        if st.button("📊 进行预测"):
            pred = st.session_state.model.predict([user_input])[0]
            pred_label = st.session_state.target_names[pred]
            st.success(f"🌟 模型预测结果是： `{pred_label}`")
