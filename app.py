import streamlit as st

st.title("机器学习体验应用")
st.write("欢迎！请在下方输入数据并点击按钮来感受机器学习的过程。")

number = st.number_input("请输入一个数字")
if st.button("执行模型"):
    st.success(f"模型预测结果是：{number * 2}")
