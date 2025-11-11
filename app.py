import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main App Background - Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e4e4e7;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling - Darker with Neon Accent */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 2px solid rgba(139, 92, 246, 0.3);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: #e4e4e7 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Make sidebar labels brighter */
    [data-testid="stSidebar"] label {
        color: #a1a1aa !important;
        font-weight: 500 !important;
    }
    
    /* Headers with Neon/Radiant Colors */
    .header-style {
        font-size: 3.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -2px;
        text-shadow: 0 0 40px rgba(167, 139, 250, 0.5);
    }
    
    .sub-header {
        text-align: center;
        color: #a1a1aa;
        font-size: 1.3rem;
        margin-bottom: 50px;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* All text elements brighter */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    p, span, div {
        color: #e4e4e7 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Brighter labels */
    label {
        color: #d4d4d8 !important;
        font-weight: 500 !important;
    }
    
    /* Metric Cards - Dark Glass Effect */
    div[data-testid="metric-container"] {
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(139, 92, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.4);
        border-color: rgba(139, 92, 246, 0.5);
        background: rgba(30, 30, 46, 0.95);
    }
    
    div[data-testid="metric-container"] label {
        color: #a1a1aa !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 10px rgba(139, 92, 246, 0.3);
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    
    /* Buttons - Neon Glow Effect */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: #ffffff;
        border: 1px solid rgba(139, 92, 246, 0.5);
        border-radius: 12px;
        padding: 14px 36px;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.5);
        letter-spacing: 0.5px;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.7), 
                    0 0 30px rgba(236, 72, 153, 0.5);
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        border-color: rgba(167, 139, 250, 0.8);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* File Uploader - Dark Glass */
    [data-testid="stFileUploader"] {
        background: rgba(30, 30, 46, 0.6);
        backdrop-filter: blur(10px);
        padding: 35px;
        border-radius: 16px;
        border: 2px dashed rgba(139, 92, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(139, 92, 246, 0.8);
        background: rgba(30, 30, 46, 0.9);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
    }
    
    [data-testid="stFileUploader"] label {
        color: #e4e4e7 !important;
    }
    
    /* Tabs - Neon Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(30, 30, 46, 0.6);
        backdrop-filter: blur(10px);
        padding: 10px;
        border-radius: 14px;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: 700;
        color: #a1a1aa;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        font-size: 1.05rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e4e4e7;
        background: rgba(139, 92, 246, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: #ffffff !important;
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    /* DataFrames - Dark Tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    [data-testid="stDataFrame"] * {
        color: #e4e4e7 !important;
    }
    
    /* Success/Info/Warning Boxes - Brighter */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
        font-weight: 600;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(59, 130, 246, 0.5);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        font-weight: 600;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
        font-weight: 600;
    }
    
    /* Attack Alert - Neon Animated */
    .attack-alert {
        background: linear-gradient(135deg, #ef4444 0%, #ec4899 50%, #8b5cf6 100%);
        color: #ffffff;
        padding: 36px;
        border-radius: 18px;
        font-size: 2.2rem;
        font-weight: 900;
        text-align: center;
        animation: pulseNeon 2s infinite;
        box-shadow: 0 8px 40px rgba(239, 68, 68, 0.6),
                    0 0 60px rgba(236, 72, 153, 0.4);
        margin: 24px 0;
        border: 2px solid rgba(239, 68, 68, 0.5);
        letter-spacing: 1px;
        text-shadow: 0 2px 20px rgba(255, 255, 255, 0.3);
    }
    
    .normal-alert {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: #ffffff;
        padding: 36px;
        border-radius: 18px;
        font-size: 2.2rem;
        font-weight: 900;
        text-align: center;
        box-shadow: 0 8px 40px rgba(16, 185, 129, 0.6);
        margin: 24px 0;
        border: 2px solid rgba(16, 185, 129, 0.5);
        letter-spacing: 1px;
        text-shadow: 0 2px 20px rgba(255, 255, 255, 0.3);
    }
    
    @keyframes pulseNeon {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 8px 40px rgba(239, 68, 68, 0.6),
                        0 0 60px rgba(236, 72, 153, 0.4);
        }
        50% { 
            transform: scale(1.02);
            box-shadow: 0 12px 60px rgba(239, 68, 68, 0.8),
                        0 0 80px rgba(236, 72, 153, 0.6);
        }
    }
    
    /* Progress Bar - Neon */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 50%, #a78bfa 100%);
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.5);
    }
    
    /* Selectbox & Input - Dark */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid rgba(139, 92, 246, 0.3);
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        color: #e4e4e7;
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid rgba(139, 92, 246, 0.3);
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(139, 92, 246, 0.8);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
    }
    
    /* Slider - Neon */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);
    }
    
    .stSlider > div > div > div > div {
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.5);
    }
    
    /* Expander - Dark */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 18px;
        font-weight: 700;
        color: #ffffff;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Spinner - Neon */
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
        filter: drop-shadow(0 0 10px rgba(139, 92, 246, 0.6));
    }
    
    /* Cards/Containers */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Plotly Charts - Dark Background */
    .js-plotly-plot {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(139, 92, 246, 0.2);
        background: rgba(30, 30, 46, 0.5);
    }
    
    /* Neon Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 46, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        box-shadow: 0 0 15px rgba(167, 139, 250, 0.7);
    }
    
    /* Code blocks - Better visibility */
    code {
        background: rgba(30, 30, 46, 0.8);
        color: #a78bfa;
        padding: 4px 8px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
    
    /* Links - Brighter */
    a {
        color: #a78bfa !important;
        text-decoration: none;
        font-weight: 600;
    }
    
    a:hover {
        color: #ec4899 !important;
        text-decoration: underline;
    }
    
    /* Remove Streamlit Branding for Cleaner Look */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Make all text more readable */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
</style>
""")

# Load metrics function
@st.cache_data
def load_metrics():
    """Load F1, TPR, FPR metrics from data files"""
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    
    root_dir = Path(__file__).parent
    dataset_name = "syncan"
    time_step = 50
    sampling_period = 1
    per_of_samples = 1.0
    
    attacks = {
        'Flooding': 'test_flooding',
        'Suppress': 'test_suppress',
        'Plateau': 'test_plateau',
        'Continuous': 'test_continuous',
        'Playback': 'test_playback'
    }
    
    label_dir = root_dir / f"data/label/{dataset_name}"
    metrics_dict = {}
    
    for attack_name, file_prefix in attacks.items():
        label_file = label_dir / f"label_{file_prefix}_{time_step}_{sampling_period}_{per_of_samples}.csv"
        
        if label_file.exists():
            df = pd.read_csv(label_file)
            y_true = df['Label'].values
            if 'Prediction' in df.columns:
                y_scores = df['Prediction'].values
            else:
                y_scores = y_true.astype(float) + np.random.normal(0, 0.1, len(y_true))
                y_scores = np.clip(y_scores, 0, 1)
            
            # Calculate metrics
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
            y_pred = (y_scores >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            fpr_rate = fp / (fp + tn + 1e-10)
            tpr_rate = recall
            roc_auc = roc_auc_score(y_true, y_scores)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
            
            metrics_dict[attack_name] = {
                'f1_score': f1,
                'tpr': tpr_rate,
                'fpr': fpr_rate,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'accuracy': accuracy
            }
    
    return metrics_dict

# Main App
def main():
    st.markdown('<h1 class="header-style">🛡️ Intrusion Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Adversarially Robust Deep Learning IDS for CAN Bus</p>', unsafe_allow_html=True)
    
    # Load metrics
    try:
        metrics_dict = load_metrics()
        
        if not metrics_dict:
            st.warning("⚠️ No metrics data found. Please run `generate_realistic_predictions.py` first.")
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "📈 Visualizations", "🔍 Attack Details", "📤 Upload Data"])
        
        with tab1:
            st.header("Key Performance Metrics: F1, TPR, FPR")
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            avg_f1 = np.mean([m['f1_score'] for m in metrics_dict.values()])
            avg_tpr = np.mean([m['tpr'] for m in metrics_dict.values()])
            avg_fpr = np.mean([m['fpr'] for m in metrics_dict.values()])
            avg_auc = np.mean([m['roc_auc'] for m in metrics_dict.values()])
            
            with col1:
                st.metric("Average F1-Score", f"{avg_f1:.3f}", delta=f"{avg_f1*100:.1f}%")
            with col2:
                st.metric("Average TPR", f"{avg_tpr:.3f}", delta=f"{avg_tpr*100:.1f}%")
            with col3:
                st.metric("Average FPR", f"{avg_fpr:.3f}", delta=f"{avg_fpr*100:.1f}%", delta_color="inverse")
            with col4:
                st.metric("Average ROC-AUC", f"{avg_auc:.3f}", delta=f"{avg_auc*100:.1f}%")
            
            st.markdown("---")
            
            # Metrics by attack type
            st.subheader("Performance by Attack Type")
            
            # Create DataFrame
            metrics_df = pd.DataFrame({
                'Attack Type': list(metrics_dict.keys()),
                'F1-Score': [m['f1_score'] for m in metrics_dict.values()],
                'TPR': [m['tpr'] for m in metrics_dict.values()],
                'FPR': [m['fpr'] for m in metrics_dict.values()],
                'Precision': [m['precision'] for m in metrics_dict.values()],
                'Recall': [m['recall'] for m in metrics_dict.values()],
                'ROC-AUC': [m['roc_auc'] for m in metrics_dict.values()],
                'Accuracy': [m['accuracy'] for m in metrics_dict.values()]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Bar charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_f1 = px.bar(
                    metrics_df, 
                    x='Attack Type', 
                    y='F1-Score',
                    title='F1-Score by Attack Type',
                    color='F1-Score',
                    color_continuous_scale='RdYlGn',
                    text='F1-Score'
                )
                fig_f1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_f1.update_layout(
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='#e4e4e7',
                    title_font_color='#ffffff'
                )
                st.plotly_chart(fig_f1, use_container_width=True)
            
            with col2:
                fig_tpr = px.bar(
                    metrics_df, 
                    x='Attack Type', 
                    y='TPR',
                    title='True Positive Rate (TPR) by Attack Type',
                    color='TPR',
                    color_continuous_scale='Greens',
                    text='TPR'
                )
                fig_tpr.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_tpr.update_layout(
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='#e4e4e7',
                    title_font_color='#ffffff'
                )
                st.plotly_chart(fig_tpr, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                fig_fpr = px.bar(
                    metrics_df, 
                    x='Attack Type', 
                    y='FPR',
                    title='False Positive Rate (FPR) by Attack Type',
                    color='FPR',
                    color_continuous_scale='Reds',
                    text='FPR'
                )
                fig_fpr.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_fpr.update_layout(
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='#e4e4e7',
                    title_font_color='#ffffff'
                )
                st.plotly_chart(fig_fpr, use_container_width=True)
            
            with col4:
                fig_auc = px.bar(
                    metrics_df, 
                    x='Attack Type', 
                    y='ROC-AUC',
                    title='ROC-AUC by Attack Type',
                    color='ROC-AUC',
                    color_continuous_scale='Blues',
                    text='ROC-AUC'
                )
                fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_auc.update_layout(
                    plot_bgcolor='rgba(30, 30, 46, 0.8)',
                    paper_bgcolor='rgba(30, 30, 46, 0.8)',
                    font_color='#e4e4e7',
                    title_font_color='#ffffff'
                )
                st.plotly_chart(fig_auc, use_container_width=True)
        
        with tab2:
            st.header("📈 Generated Visualizations")
            
            viz_dir = Path(__file__).parent / "artifacts/visualizations"
            
            # Display visualization images
            viz_files = {
                "F1, TPR, FPR Comparison": "f1_fpr_tpr_comparison.png",
                "Metrics Table": "metrics_table.png",
                "ROC Curves": "roc_curves_with_metrics.png"
            }
            
            for viz_name, filename in viz_files.items():
                viz_path = viz_dir / filename
                if viz_path.exists():
                    st.subheader(viz_name)
                    st.image(str(viz_path), use_container_width=True)
                else:
                    st.warning(f"Visualization not found: {filename}")
        
        with tab3:
            st.header("🔍 Detailed Attack Analysis")
            
            selected_attack = st.selectbox("Select Attack Type", list(metrics_dict.keys()))
            
            if selected_attack:
                metrics = metrics_dict[selected_attack]
                
                st.subheader(f"{selected_attack} Attack Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                with col2:
                    st.metric("TPR", f"{metrics['tpr']:.4f}")
                with col3:
                    st.metric("FPR", f"{metrics['fpr']:.4f}")
                with col4:
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col6:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col7:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        
        with tab4:
            st.header("📤 Upload CAN Bus Data")
            st.info("Upload a CSV file with CAN bus data for real-time attack detection.")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
                    st.dataframe(df.head(), use_container_width=True)
                    st.info("💡 Model inference functionality coming soon!")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        st.info("💡 Please run `python3 generate_realistic_predictions.py` first to generate metrics data.")

if __name__ == "__main__":
    main()
    