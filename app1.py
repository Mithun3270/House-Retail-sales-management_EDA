# app.py - Royal Dark Luxury Sales Dashboard (Full with Animated Sidebar Navigation)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ====== Page config ======
st.set_page_config(page_title="Royal Luxury Sales Forecast", layout="wide", initial_sidebar_state="expanded")

# ====== Inline CSS: Royal Dark Luxury ======
LUX_CSS = r"""
<style>
:root{
  --bg-dark: #060507;
  --gold-1: #d4af37;
  --gold-2: #ffdd66;
  --accent-text: rgba(255,246,220,0.95);
}
html, body, .stApp {
  background-color: var(--bg-dark);
  color: var(--accent-text);
  font-family: "Segoe UI", "Inter", "Roboto", Arial, sans-serif;
}

/* Animated gold shimmer veins */
#lux-veins { position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: linear-gradient(20deg, rgba(212,175,55,0.02),
  rgba(212,175,55,0.01) 50%, rgba(212,175,55,0.02) 100%);
  background-size: 200% 200%; animation: veinsMove 18s linear infinite;
  mix-blend-mode: overlay; opacity: 0.7;}
@keyframes veinsMove {0% { background-position: 0% 0%; }
 50% { background-position: 100% 50%; } 100% { background-position: 0% 0%; }}

/* Floating particles */
#lux-particles { position: fixed; inset:0; z-index:1; pointer-events:none; }
.particle { position:absolute; width:6px; height:6px; border-radius:50%;
 background: radial-gradient(circle, var(--gold-2), var(--gold-1)); opacity:0.14;
 filter: blur(0.6px); animation: particleFloat linear infinite; }
@keyframes particleFloat { 0%{transform:translateY(0) translateX(0) scale(1); opacity:0.12;}
 50%{transform:translateY(-36px) translateX(12px) scale(1.15); opacity:0.22;}
 100%{transform:translateY(0) translateX(0) scale(1); opacity:0.12;} }

/* Card & header */
.card { background: linear-gradient(180deg, rgba(255,255,255,0.015),
 rgba(255,255,255,0.01)); border-radius:12px; padding:12px; margin-bottom:14px;
 border:1px solid rgba(212,175,55,0.06);}
.title { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.gold-dot { width:12px;height:12px;border-radius:50%;
 background: radial-gradient(circle, var(--gold-2), var(--gold-1)); }
[data-testid="stPlotlyChart"] > div { background: transparent !important; }
.plotly-graph-div .modebar { display: none !important; }

.clean-table-header { font-size:13px; opacity:0.9; margin-bottom:6px; }
.small { font-size:12px; opacity:0.8; }

@media (max-width:900px){ #lux-particles, #lux-veins { display:none; } }

/* Animated sidebar nav customizations */
[data-testid="stSidebar"] .stRadio > label { width:100%; }
.sidebar-card {
  padding:12px;
  border-radius:10px;
  margin-bottom:12px;
  background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));
  border:1px solid rgba(212,175,55,0.04);
}
.nav-item {
  display:flex;
  align-items:center;
  gap:8px;
  padding:8px 10px;
  border-radius:8px;
  transition: transform .14s ease, background .18s ease, color .18s ease;
  color: rgba(255,246,220,0.92);
  cursor: pointer;
  margin-bottom:6px;
}
.nav-item:hover {
  transform: translateX(6px);
  color: #ffdd66;
  background: linear-gradient(90deg, rgba(255,221,102,0.03), rgba(212,175,55,0.02));
}
.nav-active {
  color: #ffd966 !important;
  background: linear-gradient(90deg, rgba(255,221,102,0.05), rgba(212,175,55,0.03));
  box-shadow: 0 6px 14px rgba(212,175,55,0.03);
}
.nav-icon { font-size:14px; width:20px; text-align:center; opacity:0.95; }

/* small helper to compress sidebar radio look */
.stRadio { padding: 6px 0 !important; }
</style>
"""
st.markdown(LUX_CSS, unsafe_allow_html=True)

# decorative background
st.markdown('<div id="lux-veins"></div>', unsafe_allow_html=True)
st.markdown("""<div id="lux-particles">
  <div class="particle" style="left:6%; top:14%; animation-duration:20s;"></div>
  <div class="particle" style="left:18%; top:76%; animation-duration:18s; opacity:0.09;"></div>
  <div class="particle" style="left:32%; top:36%; animation-duration:22s; opacity:0.11;"></div>
  <div class="particle" style="left:44%; top:58%; animation-duration:19s; opacity:0.08;"></div>
  <div class="particle" style="left:62%; top:24%; animation-duration:21s; opacity:0.12;"></div>
  <div class="particle" style="left:78%; top:68%; animation-duration:16s; opacity:0.10;"></div>
  <div class="particle" style="left:88%; top:34%; animation-duration:24s; opacity:0.09;"></div>
</div>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <h1 style="background: linear-gradient(90deg, #ffdd66, #d4af37); -webkit-background-clip: text; color: transparent;">Royal Luxury Sales</h1>
      <p style="margin:0; color: rgba(255,246,220,0.85); font-size:13px;">Premium analytics • Golden insights • XGBoost forecasting</p>
    </div>
  </div>
  <div style="height:8px; margin-top:12px; border-radius:8px; background: linear-gradient(90deg, rgba(212,175,55,0.08), rgba(255,221,102,0.06), rgba(212,175,55,0.08));"></div>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar (animated nav + settings) ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; gap:10px; align-items:center;"><div class="gold-dot"></div><div><strong>Luxury Settings</strong></div></div>', unsafe_allow_html=True)
    anim_speed = st.slider("Veins animation speed (higher = slower)", 8, 40, 18)
    default_forecast_days = st.slider("Default forecast days", 7, 180, 60)
    use_quantity = st.checkbox("Enable Quantity column (if present)", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Animated nav (implemented using radio but styled)
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom:8px; font-weight:700;">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Upload", "Analysis", "Train & Evaluate", "Forecast", "About"], index=0, key="nav_radio")
    st.markdown('</div>', unsafe_allow_html=True)

# initialize session state keys we'll use
for k, v in {
    'df': None,
    'df_orig': None,
    'clean_cols': None,
    'clean_cols_selected': None,
    'prod_list': None,
    'sel_prod': None,
    'prod_daily': None,
    'series': None,
    'series_model': None,
    'model': None,
    'train_metrics': None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Helper utilities ----------------
def detect_columns(df):
    cols = list(df.columns)
    date_options = [c for c in cols if any(x in c.lower() for x in ["date","order","timestamp","time"]) ]
    product_options = [c for c in cols if any(x in c.lower() for x in ["product","item","category","name","sku"]) ]
    amount_options = [c for c in cols if any(x in c.lower() for x in ["price","revenue","rating","discount","final","sales"]) ]
    qty_options = [c for c in cols if any(x in c.lower() for x in ["quantity","qty","count","units"]) ]
    if not date_options: date_options = cols
    if not product_options: product_options = cols
    if not amount_options: amount_options = cols
    return date_options, product_options, amount_options, qty_options

# ---------------- Page: Upload ----------------
def page_upload():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title"><div class="gold-dot"></div><div><strong>Upload Sales CSV</strong><div style="font-size:12px; opacity:0.75;">Required columns: Date, Product, Sales. Optional: Quantity</div></div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Unable to read CSV: {e}")
            return
        st.session_state['df_orig'] = df.copy()
        date_options, product_options, amount_options, qty_options = detect_columns(df)
        st.session_state['clean_cols'] = (date_options, product_options, amount_options, qty_options)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="title"><div class="gold-dot"></div><div><strong>Column selection</strong></div></div>', unsafe_allow_html=True)
        date_col = st.selectbox("Date column (date/order/timestamp)", options=date_options)
        product_col = st.selectbox("Product column (product/item/category/name)", options=product_options)
        sales_col = st.selectbox("Amount column (Price / Revenue / Final Price / Sales)", options=amount_options)
        qty_col = st.selectbox("Quantity column (optional)", options=["None"] + qty_options, index=0)
        st.markdown('</div>', unsafe_allow_html=True)

        # store selection and cleaned df
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, product_col])
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0)
        if qty_col != "None" and use_quantity:
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)

        st.session_state['df'] = df
        st.session_state['clean_cols_selected'] = (date_col, product_col, sales_col, qty_col)
        st.success("File uploaded and cleaned — go to Analysis to continue.")
        st.markdown("<div style='margin-top:8px;'><strong>Sample of cleaned rows:</strong></div>", unsafe_allow_html=True)
        st.dataframe(df.sort_values(date_col).head(100))
    else:
        st.markdown('<div class="card"><div style="opacity:0.9;">Please upload a CSV to start. Example columns: <em>Order Date, Product, Sales, Quantity</em></div></div>', unsafe_allow_html=True)

# ---------------- Page: Analysis ----------------
def page_analysis():
    if st.session_state['df'] is None:
        st.info("No data loaded. Upload a CSV on the Upload page first.")
        return
    df = st.session_state['df']
    date_col, product_col, sales_col, qty_col = st.session_state['clean_cols_selected']

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title"><div class="gold-dot"></div><div><strong>Historical Analysis — Product Level</strong></div></div>', unsafe_allow_html=True)

    agg_prod = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
    st.markdown("<div class='small'><strong>Total sales by product</strong></div>", unsafe_allow_html=True)
    if agg_prod.empty:
        st.info("No product-level sales available.")
    else:
        fig_prod = px.bar(x=agg_prod.values[::-1], y=agg_prod.index[::-1], orientation='h',
                          color=agg_prod.values[::-1],
                          labels={'x':'Sales','y':product_col})
        fig_prod.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                               margin=dict(l=80,r=80,t=6,b=6))
        st.plotly_chart(fig_prod, use_container_width=True)

    products = df[product_col].unique().tolist()
    sel_prod = st.selectbox("Select product for detailed analysis & forecast", options=products, index=0)
    st.session_state['sel_prod'] = sel_prod

    prod_df_raw = df[df[product_col] == sel_prod].copy()
    prod_daily = prod_df_raw.groupby(df[date_col]).agg({sales_col: 'sum'}).reset_index().sort_values(date_col)
    if qty_col != "None" and use_quantity:
        prod_qty_daily = prod_df_raw.groupby(df[date_col]).agg({qty_col: 'sum'}).reset_index().sort_values(date_col)
        prod_daily = prod_daily.merge(prod_qty_daily, left_on=date_col, right_on=date_col, how='left')
    else:
        prod_daily['qty_unused'] = np.nan

    prod_daily = prod_daily.rename(columns={date_col: 'date', sales_col: 'sales'})
    prod_daily['month'] = prod_daily['date'].dt.to_period('M').dt.to_timestamp()
    prod_daily['year'] = prod_daily['date'].dt.year
    prod_daily['weekday'] = prod_daily['date'].dt.day_name()

    monthly = prod_daily.groupby('month')['sales'].sum().reset_index()
    st.markdown("<div style='margin-top:10px;' class='small'><strong>Monthly sales trend (selected product)</strong></div>", unsafe_allow_html=True)
    if monthly.empty:
        st.info("Not enough monthly data.")
    else:
        fig_m = px.line(monthly, x='month', y='sales', markers=True)
        fig_m.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=60,r=60,t=6,b=6))
        st.plotly_chart(fig_m, use_container_width=True)

    dow = prod_daily.groupby('weekday')['sales'].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
    st.markdown("<div style='margin-top:8px;' class='small'><strong>Day-of-week average sales (selected product)</strong></div>", unsafe_allow_html=True)
    if dow['sales'].isna().all():
        st.info("Not enough weekday data.")
    else:
        fig_d = px.bar(dow, x='weekday', y='sales')
        fig_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=40,r=40,t=6,b=6))
        st.plotly_chart(fig_d, use_container_width=True)

    st.session_state['prod_daily'] = prod_daily
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Page: Train & Evaluate ----------------
def page_train():
    if st.session_state.get('prod_daily') is None:
        st.info("No product selected. Go to Analysis and pick a product first.")
        return
    prod_daily = st.session_state['prod_daily'].copy()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title"><div class="gold-dot"></div><div><strong>Forecast configuration & training preview</strong></div></div>', unsafe_allow_html=True)

    forecast_target = 'Revenue'
    if use_quantity and st.session_state['clean_cols_selected'][3] != 'None':
        forecast_target = st.radio("Forecast target", options=['Revenue','Quantity'], index=0)
    else:
        st.markdown("<div class='small'>Forecasting Revenue (Quantity column not enabled or not present).</div>", unsafe_allow_html=True)

    forecast_days = st.slider("Forecast horizon (days)", 7, 180, int(default_forecast_days))
    st.markdown("<div class='small' style='margin-top:8px;'><strong>Training settings</strong> — lag features: 1, 7, 30 days; chronological split (80% train / 20% test)</div>", unsafe_allow_html=True)

    # prepare series
    prod_daily = prod_daily.rename(columns={prod_daily.columns[0]:'date','sales':'sales'}) if 'date' not in prod_daily.columns else prod_daily
    series = prod_daily[['date','sales']].copy().rename(columns={'date':'ds','sales':'y'})
    series = series.set_index('ds').asfreq('D').fillna(0).reset_index()

    # features
    series['day_num'] = (series['ds'] - series['ds'].min()).dt.days
    series['year'] = series['ds'].dt.year
    series['month'] = series['ds'].dt.month
    series['day'] = series['ds'].dt.day
    series['weekday'] = series['ds'].dt.weekday
    series['is_weekend'] = series['weekday'].isin([5,6]).astype(int)
    for lag in [1,7,30]:
        series[f'lag_{lag}'] = series['y'].shift(lag)
    series['roll_7'] = series['y'].rolling(7, min_periods=1).mean().shift(1)
    series_model = series.dropna().reset_index(drop=True)

    st.markdown("<div style='margin-top:6px;'><strong>Training data preview (features)</strong></div>", unsafe_allow_html=True)
    if series_model.empty or len(series_model) < 31:
        st.warning("Not enough data after lagging to train a model. Need at least 31 days to create lag_30.")
        return

    st.dataframe(series_model.head(40))
    n_rows = len(series_model)
    train_size = int(n_rows * 0.8)
    st.markdown(f"<div class='small'>Rows for modeling after lagging: <strong>{n_rows}</strong> — Train: <strong>{train_size}</strong>, Test: <strong>{n_rows-train_size}</strong></div>", unsafe_allow_html=True)

    feature_cols = ['day_num','year','month','day','weekday','is_weekend','lag_1','lag_7','lag_30','roll_7']
    X = series_model[feature_cols].copy()
    y = series_model['y'].copy()
    X_train = X.iloc[:train_size].copy()
    X_test = X.iloc[train_size:].copy()
    y_train = y.iloc[:train_size].copy()
    y_test = y.iloc[train_size:].copy()

    if st.button("Train model with XGBoost"):
        with st.spinner("Training model..."):
            model = XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=4, random_state=42)
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Training failed: {e}")
                return
            y_pred_test = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)

            st.session_state['model'] = model
            st.session_state['series'] = series
            st.session_state['series_model'] = series_model
            st.session_state['train_metrics'] = {'mae':mae,'mse':mse,'rmse':rmse,'train_size':train_size}

            st.markdown(f"<div class='small'><strong>Test metrics</strong> — MAE: <strong>{mae:.2f}</strong>, MSE: <strong>{mse:.2f}</strong>, RMSE: <strong>{rmse:.2f}</strong></div>", unsafe_allow_html=True)

            fig_tt = go.Figure()
            fig_tt.add_trace(go.Scatter(x=series_model['ds'].iloc[:train_size], y=y_train, mode='lines', name='Train', line=dict(color='#ffd966')))
            fig_tt.add_trace(go.Scatter(x=series_model['ds'].iloc[train_size:], y=y_test, mode='lines', name='Actual (Test)', line=dict(color='#ffd24a')))
            fig_tt.add_trace(go.Scatter(x=series_model['ds'].iloc[train_size:], y=y_pred_test, mode='lines', name='Predicted (Test)', line=dict(color='#d4af37', dash='dash')))
            fig_tt.update_layout(title="Train / Test visualization", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=60,r=60,t=36,b=36))
            st.plotly_chart(fig_tt, use_container_width=True)
    else:
        st.info("Press 'Train model with XGBoost' to start training. Model is required for forecasting.")

# ---------------- Page: Forecast ----------------
def page_forecast():
    if st.session_state.get('model') is None or st.session_state.get('series') is None:
        st.info("Model not available. Train a model on the Train & Evaluate page first.")
        return
    model = st.session_state['model']
    series = st.session_state['series'].copy()
    sel_prod = st.session_state.get('sel_prod', 'Selected Product')

    forecast_days = st.slider("Forecast horizon (days)", 7, 180, int(default_forecast_days), key='forecast_days')

    last_known = series[['ds','y']].set_index('ds').y.copy().sort_index()
    last_vals = last_known.values.tolist()

    def build_features_for_date(target_date, base_series_min_date, last_vals_list):
        day_num = (target_date - base_series_min_date).days
        year = target_date.year
        month = target_date.month
        day = target_date.day
        weekday = target_date.weekday()
        is_weekend = 1 if weekday in (5,6) else 0
        def get_lag(l):
            if len(last_vals_list) >= l:
                return last_vals_list[-l]
            else:
                return np.mean(last_vals_list) if len(last_vals_list) > 0 else 0
        lag_1 = get_lag(1)
        lag_7 = get_lag(7)
        lag_30 = get_lag(30)
        roll_7 = np.mean(last_vals_list[-7:]) if len(last_vals_list)>0 else 0
        return {
            'day_num': day_num, 'year': year, 'month': month, 'day': day,
            'weekday': weekday, 'is_weekend': is_weekend,
            'lag_1': lag_1, 'lag_7': lag_7, 'lag_30': lag_30, 'roll_7': roll_7
        }

    last_date = series['ds'].max()
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
    future_preds = []
    iter_history = last_vals.copy()
    feature_cols = ['day_num','year','month','day','weekday','is_weekend','lag_1','lag_7','lag_30','roll_7']

    for fd in future_dates:
        feats = build_features_for_date(fd, series['ds'].min(), iter_history)
        Xf = pd.DataFrame([feats])[feature_cols]
        pred = model.predict(Xf)[0]
        future_preds.append(pred)
        iter_history.append(pred)

    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(x=series['ds'], y=series['y'], mode='lines+markers', name='Historical', line=dict(color='#ffd966')))
    fig_fore.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name='Forecast', line=dict(color='#d4af37', dash='dash')))
    fig_fore.update_layout(title=f"Forecast — {sel_prod} (next {forecast_days} days)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(l=60,r=60,t=36,b=36))
    st.plotly_chart(fig_fore, use_container_width=True)

    total_forecast_val = float(np.sum(future_preds))
    total_forecast_display = f"₹{int(total_forecast_val):,}"
    st.markdown(f"""
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div style="font-size:13px; opacity:0.85;">Predicted revenue for</div>
          <div style="font-size:18px; font-weight:700;">{sel_prod}</div>
          <div style="font-size:12px; opacity:0.75;">Next {forecast_days} days</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:12px; opacity:0.85;">Total Forecast</div>
          <div style="font-size:20px; font-weight:800; background:linear-gradient(90deg,#ffd966,#d4af37); -webkit-background-clip:text; color:transparent;">{total_forecast_display}</div>
        </div>
      </div>
      <div style="margin-top:10px; font-size:12px; color:rgba(255,246,220,0.8);">
        Tip: production-ready forecasting needs cross-validation, promotions & holiday features, and hyperparameter tuning.
      </div>
    </div>
    """, unsafe_allow_html=True)
def page_about():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title"><div class="gold-dot"></div><div><strong>About — Royal Dark Luxury</strong></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="small">This dashboard provides premium, gold-themed analytics and quick XGBoost-based forecasting. Use Upload → Analysis → Train & Evaluate → Forecast. Train must be run to enable Forecast.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
page_map = {
    "Upload": page_upload,
    "Analysis": page_analysis,
    "Train & Evaluate": page_train,
    "Forecast": page_forecast,
    "About": page_about
}

page_map.get(page, page_about)()

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; opacity:0.75;">🔶 Royal Dark Luxury • Built with Streamlit & XGBoost</div>', unsafe_allow_html=True)