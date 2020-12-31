import plotly.graph_objects as go
import streamlit as st

def plot_bar(selected_model, probability, answer):
    fig = go.Figure(data=[go.Bar(
        x=selected_model, y=probability, text=answer, textposition='inside')])
    fig.update_layout(title="Classifiers Comparison",
                      xaxis_title="Model",
                      yaxis_title="Probability")
    st.write(fig)
