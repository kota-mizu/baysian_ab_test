import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pymc as pm
import arviz as az
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger('pymc')
logger.setLevel(logging.DEBUG)

# スタイル設定
plt.style.use('seaborn')
st.set_page_config(layout="wide")

# Google Fontsの読み込み
st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
''', unsafe_allow_html=True)

# セッションステートの初期化
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# パスワード認証
if not st.session_state.authenticated:
    with st.sidebar.container():
        password = st.text_input("パスワード", type="password")
        if password == os.environ.get("password"):
            st.session_state.authenticated = True
            st.rerun()
        elif password:
            st.error("正しいパスワードを入力してください。")

if st.session_state.authenticated:
    st.title('ベイジアンA/Bテスト分析ツール')
    
    # サイドバー設定
    st.sidebar.header("テスト設定")
    
    # 期間設定
    st.sidebar.subheader("期間設定")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("開始日", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("終了日", value=datetime.today())
    
    # URL入力欄の追加
    st.sidebar.subheader("関連URL（必要であれば）")
    url_link = st.sidebar.text_area("関連URLを記載してください", placeholder="URLを記載してください")
    st.sidebar.markdown("-----------------")
    
    # 訪問者数とCV数の入力
    st.sidebar.subheader('データ入力')
    
    # Aのデータ入力
    col3, col4 = st.sidebar.columns(2)
    with col3:
        visitors_a = st.number_input('Aの訪問者数', value=1000)
    with col4:
        conversion_a = st.number_input('AのCV数', value=50)
    cvr_a = conversion_a / visitors_a
    st.sidebar.markdown(f'AのCVR :  **{"{:.1%}".format(cvr_a)}**')
    
    # Bのデータ入力
    col5, col6 = st.sidebar.columns(2)
    with col5:
        visitors_b = st.number_input('Bの訪問者数', value=1000)
    with col6:
        conversion_b = st.number_input('BのCV数', value=50)
    cvr_b = conversion_b / visitors_b
    st.sidebar.markdown(f'BのCVR :  **{"{:.1%}".format(cvr_b)}**')
    
    # 事前分布の設定
    st.sidebar.subheader('モデル設定')
    prior_dist = st.sidebar.selectbox(
        '事前分布の選択',
        ['一様分布(Uniform)', 'ベータ分布(Beta)', '正規分布(Normal)']
    )
    
    # 事前分布のパラメータ設定
    if prior_dist == 'ベータ分布(Beta)':
        col7, col8 = st.sidebar.columns(2)
        with col7:
            alpha_prior = st.number_input('α (形状パラメータ1)', value=1.0, min_value=0.1)
        with col8:
            beta_prior = st.number_input('β (形状パラメータ2)', value=1.0, min_value=0.1)
    elif prior_dist == '正規分布(Normal)':
        col7, col8 = st.sidebar.columns(2)
        with col7:
            mu_prior = st.number_input('μ (平均)', value=0.0)
        with col8:
            sigma_prior = st.number_input('σ (標準偏差)', value=1.0, min_value=0.1)
    
    # MCMCの設定
    st.sidebar.subheader('MCMCの設定')
    n_draws = st.sidebar.slider('サンプル数', 1000, 10000, 2000, step=1000)
    n_chains = st.sidebar.slider('チェーン数', 2, 4, 2)
    n_tune = st.sidebar.slider('チューニングステップ数', 500, 2000, 1000, step=500)
    
    # メインコンテンツ
    st.header('1. テスト概要')
    
    # 基本統計量の表示
    col9, col10, col11 = st.columns(3)
    with col9:
        st.metric("A: CVR", f"{cvr_a:.2%}")
    with col10:
        st.metric("B: CVR", f"{cvr_b:.2%}")
    with col11:
        relative_diff = (cvr_b - cvr_a) / cvr_a
        st.metric("相対的な差", f"{relative_diff:.2%}")
    
    # ベイジアンモデルの構築と推論
    def run_bayesian_model():
        with pm.Model() as model:
            # 事前分布の設定
            if prior_dist == '一様分布(Uniform)':
                p_a = pm.Uniform('p_a', 0, 1)
                p_b = pm.Uniform('p_b', 0, 1)
            elif prior_dist == 'ベータ分布(Beta)':
                p_a = pm.Beta('p_a', alpha=alpha_prior, beta=beta_prior)
                p_b = pm.Beta('p_b', alpha=alpha_prior, beta=beta_prior)
            else:  # 正規分布
                p_a = pm.TruncatedNormal('p_a', mu=mu_prior, sigma=sigma_prior, lower=0, upper=1)
                p_b = pm.TruncatedNormal('p_b', mu=mu_prior, sigma=sigma_prior, lower=0, upper=1)
            
            # 尤度
            obs_a = pm.Binomial('obs_a', n=visitors_a, p=p_a, observed=conversion_a)
            obs_b = pm.Binomial('obs_b', n=visitors_b, p=p_b, observed=conversion_b)
            
            # 差とリフト
            diff = pm.Deterministic('diff', p_b - p_a)
            lift = pm.Deterministic('lift', (p_b - p_a) / p_a)
            
            # サンプリング
            trace = pm.sample(n_draws, tune=n_tune, chains=n_chains, return_inferencedata=True)
            
        return trace, model
    
    st.header('2. ベイジアン分析結果')
    
    with st.spinner('モデルを計算中...'):
        trace, model = run_bayesian_model()
    
    # 結果の可視化
    col12, col13 = st.columns(2)
    with col12:
        st.subheader('変換率の事後分布')
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        az.plot_posterior(trace, var_names=['p_a', 'p_b'], ax=ax1)
        plt.title('ConversionRate Posterior Distributions')
        st.pyplot(fig1)
    with col13:
        st.subheader('差分の事後分布')
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        az.plot_posterior(trace, var_names=['diff'], ax=ax2)
        plt.title('Difference (B - A) Posterior Distribution')
        st.pyplot(fig2)
    
    # トレースプロット
    st.subheader('トレースプロット')
    fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
    pm.plot_trace(trace, var_names=['p_a', 'p_b'], axes=axes)
    st.pyplot(fig3)
    
    # 統計的まとめ
    st.header('3. 統計的まとめ')
    prob_b_better = (trace.posterior['p_b'] > trace.posterior['p_a']).mean().item()
    expected_lift = trace.posterior['lift'].mean().item()
    col14, col15 = st.columns(2)
    with col14:
        st.metric("Bが優れている確率", f"{prob_b_better:.1%}", help="BのCVRがAのCVRを上回る確率")
    with col15:
        st.metric("期待されるリフト", f"{expected_lift:.1%}", help="BがAに対して期待される相対的な改善率")
    
    st.subheader('パラメータの要約統計量')
    summary = az.summary(trace, var_names=['p_a', 'p_b', 'diff', 'lift'])
    st.dataframe(summary)



else:
    st.sidebar.error("パスワードを入力してください。")
