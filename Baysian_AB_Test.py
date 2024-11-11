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
    
    # 期間設定
    st.sidebar.subheader("取得データの入力")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("開始日", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("終了日", value=datetime.today())
    
    # データ入力方式の選択
    input_type = st.sidebar.radio(
        "データ入力方式",
        ['訪問者数とCV数', '観測期間と発生率']
    )
    
    if input_type == '訪問者数とCV数':
        # 従来の入力方式
        col3, col4 = st.sidebar.columns(2)
        with col3:
            visitors_a = st.number_input('Aの訪問者数', value=1000)
        with col4:
            conversion_a = st.number_input('AのCV数', value=50)
        cvr_a = conversion_a / visitors_a
        st.sidebar.markdown(f'AのCVR :  **{"{:.1%}".format(cvr_a)}**')
        
        col5, col6 = st.sidebar.columns(2)
        with col5:
            visitors_b = st.number_input('Bの訪問者数', value=1000)
        with col6:
            conversion_b = st.number_input('BのCV数', value=50)
        cvr_b = conversion_b / visitors_b
        st.sidebar.markdown(f'BのCVR :  **{"{:.1%}".format(cvr_b)}**')
    else:
        # ポアソン分布用の入力方式
        col3, col4 = st.sidebar.columns(2)
        with col3:
            time_period_a = st.number_input('A観測期間(時間)', value=168.0)  # 1週間=168時間
        with col4:
            events_a = st.number_input('Aのイベント数', value=50)
        rate_a = events_a / time_period_a
        st.sidebar.markdown(f'Aの発生率(events/hour) :  **{"{:.3f}".format(rate_a)}**')
        
        col5, col6 = st.sidebar.columns(2)
        with col5:
            time_period_b = st.number_input('B観測期間(時間)', value=168.0)
        with col6:
            events_b = st.number_input('Bのイベント数', value=50)
        rate_b = events_b / time_period_b
        st.sidebar.markdown(f'Bの発生率(events/hour) :  **{"{:.3f}".format(rate_b)}**')

    st.sidebar.markdown("-----------------")

    # 事前分布の設定
    st.sidebar.subheader('モデル設定')
    prior_dist = st.sidebar.selectbox(
        '事前分布の選択',
        ['一様分布(Uniform)', 'ベータ分布(Beta)', '正規分布(Normal)', 'ガンマ分布(Gamma)']
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
    elif prior_dist == 'ガンマ分布(Gamma)':
        col7, col8 = st.sidebar.columns(2)
        with col7:
            alpha_gamma = st.number_input('α (形状パラメータ)', value=1.0, min_value=0.1)
        with col8:
            beta_gamma = st.number_input('β (レートパラメータ)', value=1.0, min_value=0.1)
    elif prior_dist == '一様分布(Uniform)':
        col7, col8 = st.sidebar.columns(2)
        with col7:
            lower_bound = st.number_input('下限値', value=0.0)
        with col8:
            upper_bound = st.number_input('上限値', value=1.0, min_value=lower_bound)
    
    st.sidebar.markdown("-----------------")

    # MCMCの設定
    st.sidebar.subheader('MCMCの設定')
    n_draws = st.sidebar.slider('取得するサンプル数', 1000, 10000, 2000, step=1000)
    n_chains = st.sidebar.slider('乱数系列の数', 2, 4, 2)
    n_tune = st.sidebar.slider('捨てるサンプル数', 500, 2000, 1000, step=500)
    target_accept = st.sidebar.selectbox(
        'target_accept (デフォルト: 0.8)',
        [0.8, 0.90, 0.95, 0.98, 0.99, 0.995]
    )

    # メインコンテンツ
    st.subheader('1. テスト概要')
    
    # データ表示を入力方式に応じて変更
    if input_type == '訪問者数とCV数':
        # 期間差分を計算（テスト日数を求める）
        days_difference = (end_date - start_date).days
        
        st.markdown(rf'''
            <style>
            table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
            }}
            th, td {{
                padding: 10px;
                text-align: center;
                border: 1px solid black;
                font-size: 18px;
            }}
            </style>
        
            <table>
              <tr>
                <th>対象</th>
                <th>訪問者数</th>
                <th>CV数</th>
                <th>CVR</th>
                <th>CVR改善率（B/A）</th>
                <th>増加差分</th>
                <th>月間換算</th>
              </tr>
              <tr>
                <td>A</td>
                <td>{visitors_a}</td>
                <td>{conversion_a}</td>
                <td>{"{:.1%}".format(cvr_a)}</td>
                <td rowspan="2">{"{:.1%}".format(cvr_b / cvr_a)}</td>
                <td rowspan="2">{"{:.1f}".format((cvr_b - cvr_a) * (visitors_a + visitors_b))}</td>
                <td rowspan="2">{"{:.1f}".format((cvr_b - cvr_a) * (visitors_a + visitors_b) / days_difference * 30) if days_difference > 0 else "N/A"}</td>
              </tr>
              <tr>
                <td>B</td>
                <td>{visitors_b}</td>
                <td>{conversion_b}</td>
                <td>{"{:.1%}".format(cvr_b)}</td>
              </tr>
            </table>
            ''', unsafe_allow_html=True)
    else:
        st.markdown(rf'''
            <style>
            table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
            }}
            th, td {{
                padding: 10px;
                text-align: center;
                border: 1px solid black;
                font-size: 18px;
            }}
            </style>
        
            <table>
              <tr>
                <th>対象</th>
                <th>観測期間(時間)</th>
                <th>イベント数</th>
                <th>発生率(events/hour)</th>
                <th>改善率（B/A）</th>
              </tr>
              <tr>
                <td>A</td>
                <td>{time_period_a}</td>
                <td>{events_a}</td>
                <td>{"{:.3f}".format(rate_a)}</td>
                <td rowspan="2">{"{:.1%}".format(rate_b / rate_a)}</td>
              </tr>
              <tr>
                <td>B</td>
                <td>{time_period_b}</td>
                <td>{events_b}</td>
                <td>{"{:.3f}".format(rate_b)}</td>
              </tr>
            </table>
            ''', unsafe_allow_html=True)

    # ベイジアンモデルの構築と推論
    def run_bayesian_model():
        with pm.Model() as model:
            if input_type == '訪問者数とCV数':
                # 従来の二項分布モデル
                if prior_dist == '一様分布(Uniform)':
                    p_a = pm.Uniform('p_a', lower=lower_bound, upper=upper_bound)
                    p_b = pm.Uniform('p_b', lower=lower_bound, upper=upper_bound)
                elif prior_dist == 'ベータ分布(Beta)':
                    p_a = pm.Beta('p_a', alpha=alpha_prior, beta=beta_prior)
                    p_b = pm.Beta('p_b', alpha=alpha_prior, beta=beta_prior)
                elif prior_dist == '正規分布(Normal)':
                    p_a = pm.TruncatedNormal('p_a', mu=mu_prior, sigma=sigma_prior, lower=0, upper=1)
                    p_b = pm.TruncatedNormal('p_b', mu=mu_prior, sigma=sigma_prior, lower=0, upper=1)
                
                # 尤度
                obs_a = pm.Binomial('obs_a', n=visitors_a, p=p_a, observed=conversion_a)
                obs_b = pm.Binomial('obs_b', n=visitors_b, p=p_b, observed=conversion_b)
                
                # 差とリフト
                diff = pm.Deterministic('diff', p_b - p_a)
                lift = pm.Deterministic('lift', (p_b - p_a) / p_a)
            
            else:
                # ポアソン-ガンマモデル
                if prior_dist == 'ガンマ分布(Gamma)':
                    lambda_a = pm.Gamma('lambda_a', alpha=alpha_gamma, beta=beta_gamma)
                    lambda_b = pm.Gamma('lambda_b', alpha=alpha_gamma, beta=beta_gamma)
                else:
                    # ガンマ分布以外の場合はデフォルトのガンマ事前分布を使用
                    lambda_a = pm.Gamma('lambda_a', alpha=1.0, beta=1.0)
                    lambda_b = pm.Gamma('lambda_b', alpha=1.0, beta=1.0)
                
                # 尤度
                obs_a = pm.Poisson('obs_a', mu=lambda_a * time_period_a, observed=events_a)
                obs_b = pm.Poisson('obs_b', mu=lambda_b * time_period_b, observed=events_b)
                
                # 差とリフト
                diff = pm.Deterministic('diff', lambda_b - lambda_a)
                lift = pm.Deterministic('lift', (lambda_b - lambda_a) / lambda_a)
            
            # サンプリング
            trace = pm.sample(n_draws, tune=n_tune, chains=n_chains, return_inferencedata=True, 
                            random_seed=42, target_accept=target_accept)
            
        return trace, model
    
    
    # Streamlit UI部分
    st.subheader('2. ベイジアン分析結果')

    # モデルの計算を開始するボタンを作成
    button_clicked = st.button('モデルの計算を始める')
    
    if button_clicked:
        with st.spinner('モデルを計算中...'):
            trace, model = run_bayesian_model()
    
        # ここから先は先ほどと同様の処理
        # 確率モデルの構造を可視化
        st.markdown('<h4>確率モデル構造</h4>', unsafe_allow_html=True)
        g = pm.model_to_graphviz(model)
        st.graphviz_chart(g)
        
        # 事後分布の可視化
        col12, col13 = st.columns(2)
        
        if input_type == '訪問者数とCV数':
            # CVRベースの分析
            with col12:
                st.markdown('<h4>変換率の事後分布</h4>', unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                az.plot_posterior(trace, var_names=['p_a', 'p_b'], ax=ax1)
                plt.title('Conversion Rate Posterior Distributions')
                st.pyplot(fig1)
            
            with col13:
                st.markdown('<h4>差分の事後分布</h4>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                az.plot_posterior(trace, var_names=['diff'], ax=ax2)
                xx, yy = ax2.get_lines()[0].get_data()
                ax2.fill_between(xx[xx<0], yy[xx<0])
                plt.title('Difference (B - A) Posterior Distribution')
                st.pyplot(fig2)
            
            # トレースプロット
            st.markdown('<h4>トレースプロット</h4>', unsafe_allow_html=True)
            fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
            pm.plot_trace(trace, var_names=['p_a', 'p_b'], axes=axes, compact=False)
            plt.tight_layout()
            st.pyplot(fig3)
            
            # 統計的まとめ
            st.subheader('3. 統計的まとめ')
            prob_b_better = (trace.posterior['p_b'] > trace.posterior['p_a']).mean().item()
            expected_lift = trace.posterior['lift'].mean().item()
            
            col14, col15 = st.columns(2)
            with col14:
                st.metric("Bが優れている確率", f"{prob_b_better:.1%}", 
                         help="BのCVRがAのCVRを上回る確率")
            with col15:
                st.metric("期待されるリフト", f"{expected_lift:.1%}", 
                         help="BがAに対して期待される相対的な改善率")
            
            summary = az.summary(trace, var_names=['p_a', 'p_b', 'diff', 'lift'])
            
        else:
            # イベント発生率ベースの分析
            with col12:
                st.markdown('<h4>イベント発生率の事後分布</h4>', unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                az.plot_posterior(trace, var_names=['lambda_a', 'lambda_b'], ax=ax1)
                plt.title('Event Rate Posterior Distributions')
                st.pyplot(fig1)
            
            with col13:
                st.markdown('<h4>差分の事後分布</h4>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                az.plot_posterior(trace, var_names=['diff'], ax=ax2)
                xx, yy = ax2.get_lines()[0].get_data()
                ax2.fill_between(xx[xx<0], yy[xx<0])
                plt.title('Difference (B - A) Posterior Distribution')
                st.pyplot(fig2)
            
            # トレースプロット
            st.markdown('<h4>トレースプロット</h4>', unsafe_allow_html=True)
            fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
            pm.plot_trace(trace, var_names=['lambda_a', 'lambda_b'], axes=axes, compact=False)
            plt.tight_layout()
            st.pyplot(fig3)
            
            # 統計的まとめ
            st.subheader('3. 統計的まとめ')
            prob_b_better = (trace.posterior['lambda_b'] > trace.posterior['lambda_a']).mean().item()
            expected_lift = trace.posterior['lift'].mean().item()
            
            col14, col15 = st.columns(2)
            with col14:
                st.metric("Bが優れている確率", f"{prob_b_better:.1%}", 
                         help="Bの発生率がAの発生率を上回る確率")
            with col15:
                st.metric("期待されるリフト", f"{expected_lift:.1%}", 
                         help="BがAに対して期待される相対的な改善率")
            
            summary = az.summary(trace, var_names=['lambda_a', 'lambda_b', 'diff', 'lift'])
        
        st.markdown('<h4>パラメータの要約統計量</h4>', unsafe_allow_html=True)
        st.dataframe(summary)



else:
    st.sidebar.error("パスワードを入力してください。")
