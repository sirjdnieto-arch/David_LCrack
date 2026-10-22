# app.py
# ─────────────────────────────────────────────────────────────────────────────
# David Indicators Confluence — Streamlit Dashboard
#   Tab 1 : Escáner estructural + seguimiento de siete indicadores
#   Tab 2 : Gráficos Estrategia  (9 panels + Estocástico) — ON DEMAND
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import io
import gc
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

from ta.trend    import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from indicators import (
    download_df, clean_yf_df,
    mcginley_dynamic, calculate_pvi, calculate_nvi,
    calc_mfi_blai5, calc_stoch, awesome_osc,
    calculate_bbwp, bbwp_signal,
    compute_blai5_koncorde, blai5_signals,
    clasificar_bitman, detectar_divergencia_simple,
    azul_z_score,
    semaforo_salida,
    calcular_señales_principales, clasificar_fase_v3,
    get_confluence_dashboard,
    calc_stochastic_full,
)
from config import ALL_TICKERS, TICKER_NAMES, GRUPOS
from structural_signals import structural_data, scanner_table, LABELS

import yfinance as yf

INTERVAL_CONFIG = {
    "1D": {"yf_interval":"1d",  "yf_period":"5y",  "resample":None, "label":"Diario"},
    "1W": {"yf_interval":"1wk", "yf_period":"5y",  "resample":None, "label":"Semanal"},
    "4h": {"yf_interval":"1h",  "yf_period":"60d", "resample":"4h", "label":"4 horas"},
    "1h": {"yf_interval":"1h",  "yf_period":"30d", "resample":None, "label":"1 hora"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="David Indicators Confluence",
    page_icon="🥒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    body, .stApp { background-color: #0d0f14; color: #ffffff; }
    .stDataFrame thead th { background: #13161e !important; color: #efb030 !important; }
    .stDataFrame tbody td { color: #ffffff !important; font-size: 0.78rem; }
    .stDataFrame tbody tr:hover { background: #1f2430 !important; }
    .stDataFrame { overflow-x: auto !important; }
    [data-testid="stMetricValue"]  { color: #ffffff !important; }
    [data-testid="stMetricDelta"]  { color: #c8cad0 !important; }
    [data-testid="stMetricLabel"]  { color: #efb030 !important; }
    label { color: #ffffff !important; }
    .stSlider label { color: #ffffff !important; }
    .stCaption { color: #aaaaaa !important; }
    div[data-baseweb="tab-list"] {
        background-color: #0d0f14 !important;
        border-bottom: 2px solid #efb030 !important;
        gap: 6px !important;
    }
    button[data-baseweb="tab"] {
        background-color: #1a1e2e !important;
        color: #aaaaaa !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        font-family: monospace !important;
        letter-spacing: 0.05em !important;
        padding: 10px 28px !important;
        border-radius: 6px 6px 0px 0px !important;
        border: 1px solid #2a2e45 !important;
        border-bottom: none !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #252a3a !important; color: #efb030 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #efb030 !important; color: #0d0f14 !important;
        border-color: #efb030 !important; font-size: 1.05rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"]::after { display: none !important; }
    @media (max-width: 768px) {
        .stDataFrame tbody td { font-size: 0.65rem !important; }
        h2 { font-size: 1.1rem !important; }
        button[data-baseweb="tab"] { font-size: 0.8rem !important; padding: 8px 12px !important; }
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════════════════════════════

STYLE = dict(
    bg="#0d0f14",        panel="#13161e",      border="#1f2430",
    bull="#26a65b",      bear="#e04040",
    bull_fade="#26a65b55",                      bear_fade="#e0404055",
    mcg="#efb030",       ema200="#6060dd",
    text="#ffffff",      muted="#aaaaaa",
    verde="#2ca85e",     marron="#a06432",
    azul="#4488e0",      media_k="#ffffff",
    pvi="#6090e0",       pvi_ema="#efb030",
    macd_line="#6090e0", macd_sig="#efb030",
    rsi="#a78bfa",       adx="#ffffff",
    pdi="#26a65b",       ndi="#e04040",
    ao_up="#26a65b",     ao_dn="#e04040",
    vol="#3a6ea8",       vol_ma="#efb030",
    grid="#1a1e28",      zero="#2a2e3a",
    stoch_k="#a78bfa",   stoch_d="#efb030",
)

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["panel"],
    "axes.edgecolor":   STYLE["border"],
    "axes.labelcolor":  STYLE["muted"],
    "xtick.color":      "#ffffff",
    "ytick.color":      "#ffffff",
    "text.color":       STYLE["text"],
    "grid.color":       STYLE["grid"],
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "font.size":        9,
})


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def download_ohlcv(ticker: str, interval_key: str = "1D") -> pd.DataFrame:
    cfg = INTERVAL_CONFIG[interval_key]
    df  = yf.download(
        ticker, period=cfg["yf_period"], interval=cfg["yf_interval"],
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    df = clean_yf_df(df)
    if df.empty: return df
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum",
        }).dropna()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cached_dashboard(tickers_tuple: tuple) -> pd.DataFrame:
    from indicators import get_confluence_dashboard
    return get_confluence_dashboard(list(tickers_tuple))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_chart_data(ticker: str, interval_key: str = "1D") -> dict:
    df = download_ohlcv(ticker, interval_key)
    if df.empty or len(df) < 60: return {}

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    mcg25     = mcginley_dynamic(close, 25)
    ema200    = EMAIndicator(close=close, window=200).ema_indicator()
    adx_ind   = ADXIndicator(high=high, low=low, close=close, window=14)
    adx_s     = adx_ind.adx()
    pdi_s     = adx_ind.adx_pos()
    ndi_s     = adx_ind.adx_neg()
    ao_s      = awesome_osc(high, low)
    bitman    = clasificar_bitman(df)
    div_df    = detectar_divergencia_simple(df)
    _, bbwp_s = calculate_bbwp(close, bb_len=13, lookback=252)
    konc      = compute_blai5_koncorde(df, m=15)
    pvi_s     = calculate_pvi(close, volume)
    pvi_ema   = pvi_s.ewm(span=25, adjust=False).mean()
    vol_ma    = volume.rolling(20).mean()
    macd_obj  = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    structural = structural_data(df,ticker)
    rsi_s = structural['rsi14']
    stoch_k, stoch_d = calc_stochastic_full(high, low, close, k_period=18, k_smooth=5, d_period=9)

    return dict(
        df=df, mcg25=mcg25, ema200=ema200,
        adx_s=adx_s, pdi_s=pdi_s, ndi_s=ndi_s,
        ao_s=ao_s, bitman=bitman, div_df=div_df,
        bbwp_s=bbwp_s, konc=konc,
        pvi_s=pvi_s, pvi_ema=pvi_ema, vol_ma=vol_ma,
        macd_line=macd_obj.macd(),
        macd_sig=macd_obj.macd_signal(),
        macd_hist=macd_obj.macd_diff(),
        rsi_s=rsi_s, structural=structural,
        stoch_k=stoch_k, stoch_d=stoch_d,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SYNC BAR HTML
# ══════════════════════════════════════════════════════════════════════════════

def sv(series: pd.Series, index) -> np.ndarray:
    return series.reindex(index).values

def format_xaxis(ax, index, n_labels: int = 8):
    step   = max(1, len(index) // n_labels)
    ticks  = list(range(0, len(index), step))
    labels = [index[i].strftime("%d %b") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8, color="#ffffff")

def panel_style(ax, ylabel="", yticks=5, zero_line=False):
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4, color="#ffffff")
    ax.yaxis.set_major_locator(plt.MaxNLocator(yticks, prune="both"))
    ax.tick_params(axis="both", labelsize=7, colors="#ffffff")
    ax.grid(True, axis="y", linewidth=0.4)
    ax.grid(True, axis="x", linewidth=0.2, alpha=0.4)
    if zero_line:
        ax.axhline(0, color=STYLE["zero"], linewidth=0.8, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["border"])


# ══════════════════════════════════════════════════════════════════════════════
# BUILD SIGNALS (for Tab 2)
# ══════════════════════════════════════════════════════════════════════════════

def build_signals(data: dict) -> list:
    df      = data["df"]; konc = data["konc"]; close = df["Close"]
    mcg25   = data["mcg25"]; ema200 = data["ema200"]; rsi_s = data["rsi_s"]
    pvi_s   = data["pvi_s"]; pvi_ema = data["pvi_ema"]; macd_h = data["macd_hist"]
    macd_l  = data["macd_line"]; bitman = data["bitman"]; div_df = data["div_df"]
    p       = close.iloc[-1]

    def sig(label, bull, neutral=False):
        return {"label":label, "state":"neutral" if neutral else ("bull" if bull else "bear")}

    sigs = [
        sig(f"precio {'>' if p >= mcg25.iloc[-1] else '<'} MCG25",   p >= mcg25.iloc[-1]),
        sig(f"precio {'>' if p >= ema200.iloc[-1] else '<'} EMA200", p >= ema200.iloc[-1]),
    ]
    r = rsi_s.iloc[-1]
    sigs.append(sig(f"RSI {r:.1f}", r > 50, neutral=45 < r < 55))
    sigs.append(sig(f"MACD hist {'↑' if macd_h.iloc[-1] >= 0 else '↓'}", macd_h.iloc[-1] >= 0))
    sigs.append(sig(f"MACD línea {'≥0' if macd_l.iloc[-1] >= 0 else '<0'}", macd_l.iloc[-1] >= 0))
    if not konc.empty:
        sigs.append(sig(f"Azul {'↑' if konc['azul'].iloc[-1] >= 0 else '↓'}", konc["azul"].iloc[-1] >= 0))
        sigs.append(sig(f"Verde {'>' if konc['verde'].iloc[-1] >= konc['marron'].iloc[-1] else '<'} M",
                        konc["verde"].iloc[-1] >= konc["marron"].iloc[-1]))
    sigs.append(sig(f"PVI {'>' if pvi_s.iloc[-1] >= pvi_ema.iloc[-1] else '<'} EMA25",
                    pvi_s.iloc[-1] >= pvi_ema.iloc[-1]))
    a = data["adx_s"].iloc[-1]
    sigs.append(sig(f"ADX {a:.1f}", a > 25, neutral=18 < a < 25))
    if bitman is not None and not bitman.empty:
        b_etiq = bitman["Bitman_Etiqueta"].iloc[-1]
        b_v    = int(bitman["Bitman_Velas"].iloc[-1])
        sigs.append(sig(f"Bitman {b_etiq[:8]} ({b_v}v)",
                        b_etiq in ("IMPULSO ALCISTA","RETROCESO ALCISTA"),
                        neutral=(b_etiq == "INDEFINICIÓN")))
    if div_df is not None:
        dt = div_df["divergencia_tipo"].iloc[-1]
        if dt == "alcista":   sigs.append({"label":"Div RSI alc","state":"bull"})
        elif dt == "bajista": sigs.append({"label":"Div RSI baj","state":"bear"})

    sk = data["stoch_k"].iloc[-1]
    sd = data["stoch_d"].iloc[-1]
    stoch_bull = sk > sd and sk < 40
    sigs.append(sig(f"Stoch {sk:.0f}/{sd:.0f}", stoch_bull, neutral=40 <= sk <= 60))

    return sigs

def score_signals(sigs: list) -> tuple:
    bulls = sum(1 for s in sigs if s["state"] == "bull")
    bears = sum(1 for s in sigs if s["state"] == "bear")
    total = bulls + bears
    pct   = round(bulls / total * 100) if total else 0
    if pct >= 80:   label = "CONFLUENCIA MÁXIMA"
    elif pct >= 60: label = "SETUP SÓLIDO"
    elif pct >= 40: label = "SEÑALES MIXTAS"
    else:           label = "PRESIÓN BAJISTA"
    return pct, label, bulls, len(sigs)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE — memory-optimized
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(data: dict, ticker: str, n_candles: int = 200) -> bytes:
    """Build figure and return PNG bytes. No st.pyplot — avoids memory leak."""
    df        = data["df"]
    close     = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
    mcg25     = data["mcg25"]; ema200 = data["ema200"]
    adx_s     = data["adx_s"]; pdi_s = data["pdi_s"]; ndi_s = data["ndi_s"]
    ao_s      = data["ao_s"]; bitman = data["bitman"]; div_df = data["div_df"]
    bbwp_s    = data["bbwp_s"]; konc = data["konc"]
    pvi_s     = data["pvi_s"]; pvi_ema = data["pvi_ema"]; vol_ma = data["vol_ma"]
    macd_line = data["macd_line"]; macd_sig = data["macd_sig"]; macd_hist = data["macd_hist"]
    rsi_s     = data["rsi_s"]
    stoch_k   = data["stoch_k"]; stoch_d = data["stoch_d"]

    sigs                              = build_signals(data)
    pct, score_label, bull_n, total_n = score_signals(sigs)
    score_color = STYLE["bull"] if pct >= 60 else (STYLE["bear"] if pct < 40 else STYLE["mcg"])

    n_max   = min(n_candles, len(df))
    df_plot = df.iloc[-n_max:]
    idx     = df_plot.index
    xs      = np.arange(len(idx))

    # 9 panels — smaller figure
    fig = plt.figure(figsize=(14, 24), facecolor=STYLE["bg"])
    heights = [5, 1.2, 3.2, 2, 2.2, 1.4, 1.6, 1.8, 1.6]
    gs  = gridspec.GridSpec(9, 1, figure=fig, height_ratios=heights, hspace=0.35)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.04)
    axes = [fig.add_subplot(gs[i]) for i in range(9)]
    for i in range(1, 9): axes[i].tick_params(labelbottom=False)

    last_p  = close.iloc[-1]; prev_p = close.iloc[-2]
    chg     = last_p - prev_p; pct_chg = chg / prev_p * 100
    chg_c   = STYLE["bull"] if chg >= 0 else STYLE["bear"]
    sign    = "+" if chg >= 0 else ""

    nombre = TICKER_NAMES.get(ticker, "")
    title_str = f"{ticker}  ({nombre})" if nombre else ticker

    fig.text(0.07, 0.965, title_str,         fontsize=18, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.30, 0.965, f"{last_p:.2f}",fontsize=16, fontweight="bold", color=STYLE["text"], va="bottom")
    fig.text(0.42, 0.965, f"{sign}{chg:.2f}  ({sign}{pct_chg:.2f}%)", fontsize=12, color=chg_c, va="bottom")
    fig.text(0.97, 0.965, f"{score_label}  ·  {bull_n}/{total_n}  ({pct}%)",
             fontsize=11, color=score_color, ha="right", va="bottom", style="italic")

    # PANEL 0 — Velas
    ax0 = axes[0]; w = 0.4
    for i, (_, row) in enumerate(df_plot.iterrows()):
        col = STYLE["bull"] if row["Close"] >= row["Open"] else STYLE["bear"]
        ax0.plot([i,i],[row["Low"],row["High"]], color=col, lw=0.8, zorder=2)
        ax0.add_patch(plt.Rectangle((i-w, min(row["Open"],row["Close"])),
            2*w, max(abs(row["Close"]-row["Open"]),0.001), color=col, zorder=3))
    ax0.plot(xs, sv(mcg25,idx),  color=STYLE["mcg"],   lw=1.4, label="MCG 25",  zorder=4)
    ax0.plot(xs, sv(ema200,idx), color=STYLE["ema200"], lw=1.4, label="EMA 200", zorder=4)
    structural=data['structural']; offset=len(df)-n_max
    wedge_ids={s['ID'] for s in structural['wedges']}
    visible=[]
    for f in structural['figures']:
        if f['tipo']!='CUÑA' or (f['estado']!='ACTIVA' and f['id'] not in wedge_ids): continue
        end=f['fin'] if f['fin'] is not None else len(df)-1
        if end<offset: continue
        visible.append((f,end))
    # Keep the drawing light: at most six recent shapes; all breakout markers remain.
    for f,end in sorted(visible,key=lambda item:item[1])[-6:]:
        xx=np.arange(max(offset,f['x0']),end+1)
        for m,b in [(f['mh'],f['bh']),(f['ml'],f['bl'])]:
            ax0.plot(xx-offset,m*xx+b,color='#b5c5c7',lw=.85,ls=(0,(5,4)),alpha=.5,zorder=2)
    for s in structural['wedges']:
        t=s['t']
        if t>=offset:
            ax0.annotate('▲',xy=(t-offset,float(low.iloc[t])),xytext=(0,-10),
                         textcoords='offset points',ha='center',va='top',fontsize=9,color='#26cf78')
    ax0.set_xlim(-1,len(idx))
    ax0.legend(loc="upper left",fontsize=8,frameon=False,labelcolor=[STYLE["mcg"],STYLE["ema200"]])
    panel_style(ax0,ylabel="Precio")
    ax0.set_title("Velas  ·  McGinley 25  ·  EMA 200",fontsize=9,color=STYLE["muted"],loc="right",pad=4)
    format_xaxis(ax0,idx); ax0.tick_params(labelbottom=True,bottom=True)
    ax0.spines["bottom"].set_linewidth(1.2); ax0.spines["bottom"].set_edgecolor("#ffffff")

    # PANEL 1 — Volumen
    ax1=axes[1]
    vol_v=sv(volume,idx); vol_m=sv(vol_ma,idx)
    vol_colors=[STYLE["bull"] if df_plot["Close"].iloc[i]>=df_plot["Open"].iloc[i] else STYLE["bear"] for i in range(len(df_plot))]
    ax1.bar(xs,vol_v,color=vol_colors,alpha=0.6,width=0.8,zorder=2)
    ax1.fill_between(xs,vol_m,alpha=0.25,color=STYLE["vol_ma"],zorder=1)
    ax1.plot(xs,vol_m,color=STYLE["vol_ma"],lw=1.2,label="Vol MA20",zorder=3)
    ax1.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["vol_ma"]])
    panel_style(ax1,ylabel="Vol")
    ax1.set_title("Volumen  ·  MA 20",fontsize=9,color=STYLE["muted"],loc="right",pad=4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x/1e6:.0f}M" if x>=1e6 else f"{x/1e3:.0f}K"))

    # PANEL 2 — RSI14 + RSI65 and original structural events
    ax2=axes[2]; rsi_v=sv(rsi_s,idx)
    ax2.plot(xs,rsi_v,color=STYLE['rsi'],lw=1.35,label='RSI14')
    ax2.plot(xs,sv(structural['rsi65'],idx),color='#689ed8',lw=1.0,alpha=.85,label='RSI65')
    for level in [30,45,60,70]:
        ax2.axhline(level,color=STYLE['muted'],lw=.6,ls='--',alpha=.5 if level in (45,60) else .25)
    events=structural['events']
    for kind,color,series,hollow in [('DIV_ALCISTA','#ff9c30',rsi_s,False),
                                    ('RSI_AZUL','#329dff',structural['rsi65'],False),
                                    ('DIV_FANTASMA','#ff9c30',rsi_s,True)]:
        positions=events.loc[events.Senal==kind,'signal_idx'].astype(int)
        positions=positions[positions>=offset]
        if len(positions):
            ax2.scatter(positions-offset,series.iloc[positions],s=36,
                        facecolors='none' if hollow else color,edgecolors=color,linewidths=1.3,zorder=6)
    slope=structural['slope'].iloc[-1]
    arrow='↑' if slope>0 else '↓' if slope<0 else '→'
    ax2.text(.99,.93,f'Media RSI65/8: {arrow} {slope:+.2f} (2v)',transform=ax2.transAxes,
             ha='right',va='top',fontsize=8,color=STYLE['muted'])
    panel_style(ax2,ylabel='RSI',yticks=5)
    ax2.set_ylim(0,100); ax2.set_yticks([30,45,60,70])
    ax2.legend(loc='upper left',fontsize=8,frameon=False,labelcolor='linecolor')
    ax2.set_title('Naranja: confirmada · círculo hueco: provisional · azul: RSI65/8',
                  fontsize=8,color=STYLE['muted'],loc='right',pad=4)

    # PANEL 3 — ADX + AO
    ax3=axes[3]; ax3r=ax3.twinx()
    ao_v=sv(ao_s,idx); ao_p=np.roll(ao_v,1); ao_p[0]=ao_v[0]
    ao_c=[STYLE["ao_up"] if ao_v[i]>=ao_p[i] else STYLE["ao_dn"] for i in range(len(ao_v))]
    ax3r.bar(xs,ao_v,color=ao_c,alpha=0.7,width=0.8,zorder=2)
    ax3r.axhline(0,color=STYLE["zero"],lw=0.7)
    ax3r.tick_params(labelsize=7,colors="#ffffff")
    ax3r.set_ylabel("AO",fontsize=7,color="#ffffff")
    ax3r.spines["right"].set_edgecolor(STYLE["border"])
    ax3.plot(xs,sv(adx_s,idx),color=STYLE["adx"],lw=1.6,label="ADX",zorder=3)
    ax3.plot(xs,sv(pdi_s,idx),color=STYLE["pdi"],lw=0.9,ls="--",label="+DI",zorder=3)
    ax3.plot(xs,sv(ndi_s,idx),color=STYLE["ndi"],lw=0.9,ls="--",label="-DI",zorder=3)
    ax3.axhline(25,color=STYLE["muted"],lw=0.6,ls=":")
    ax3.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["adx"],STYLE["pdi"],STYLE["ndi"]])
    panel_style(ax3,ylabel="ADX")
    ax3.set_title("ADX  ·  +DI / -DI  ·  Awesome Oscillator",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 4 — Koncorde
    ax4=axes[4]
    if not konc.empty:
        for key,col in [("verde",STYLE["verde"]),("marron",STYLE["marron"]),("azul",STYLE["azul"])]:
            v=sv(konc[key],idx)
            ax4.fill_between(xs,v,alpha=0.60,color=col,label=key.capitalize(),zorder=2)
            ax4.plot(xs,v,color=col,lw=1.0,zorder=3)
        ax4.plot(xs,sv(konc["media"],idx),color=STYLE["media_k"],lw=1.6,label="Media",zorder=4)
        ax4.legend(loc="upper left",fontsize=7,frameon=False,
                   labelcolor=[STYLE["verde"],STYLE["marron"],STYLE["azul"],STYLE["media_k"]])
    ax4.axhline(0,color=STYLE["zero"],lw=0.7)
    panel_style(ax4,ylabel="Koncorde")
    ax4.set_title("Blai5 Koncorde  ·  Verde / Marrón / Azul / Media",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 5 — BBWP
    ax5=axes[5]; bbwp_v=sv(bbwp_s,idx)
    ax5.fill_between(xs,bbwp_v,20,where=(~np.isnan(bbwp_v))&(bbwp_v<20),alpha=0.25,color=STYLE["azul"],zorder=1)
    ax5.fill_between(xs,bbwp_v,80,where=(~np.isnan(bbwp_v))&(bbwp_v>80),alpha=0.25,color=STYLE["bear"],zorder=1)
    for i in range(1,len(xs)):
        if np.isnan(bbwp_v[i]) or np.isnan(bbwp_v[i-1]): continue
        mid=(bbwp_v[i]+bbwp_v[i-1])/2
        lc=STYLE["azul"] if mid<20 else(STYLE["bear"] if mid>80 else STYLE["muted"])
        ax5.plot([xs[i-1],xs[i]],[bbwp_v[i-1],bbwp_v[i]],color=lc,lw=1.5,zorder=3)
    ax5.axhline(80,color=STYLE["bear"],lw=0.7,ls="--",alpha=0.6)
    ax5.axhline(20,color=STYLE["azul"],lw=0.7,ls="--",alpha=0.6)
    ax5.axhline(50,color=STYLE["muted"],lw=0.5,ls=":",alpha=0.4)
    ax5.set_ylim(-2,102); ax5.yaxis.set_ticks([0,20,50,80,100])
    panel_style(ax5,ylabel="BBWP")
    ax5.set_title("BBWP 13/252  ·  compresión<20  ·  expansión>80",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 6 — PVI
    ax6=axes[6]; pvi_v=sv(pvi_s,idx); pvi_e=sv(pvi_ema,idx)
    ax6.fill_between(xs,pvi_v,pvi_e,where=(pvi_v>=pvi_e),alpha=0.18,color=STYLE["bull"])
    ax6.fill_between(xs,pvi_v,pvi_e,where=(pvi_v<pvi_e), alpha=0.18,color=STYLE["bear"])
    ax6.plot(xs,pvi_v,color=STYLE["pvi"],    lw=1.4,label="PVI")
    ax6.plot(xs,pvi_e,color=STYLE["pvi_ema"],lw=1.4,ls="--",label="EMA 25")
    ax6.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["pvi"],STYLE["pvi_ema"]])
    panel_style(ax6,ylabel="PVI")
    ax6.set_title("PVI  ·  EMA 25",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 7 — MACD
    ax7=axes[7]; hist_v=sv(macd_hist,idx)
    hist_p=np.roll(hist_v,1); hist_p[0]=hist_v[0]
    bar_col=[]
    for i in range(len(hist_v)):
        v,p_v=hist_v[i],hist_p[i]
        if np.isnan(v): bar_col.append(STYLE["muted"]); continue
        bar_col.append(STYLE["bull"] if(v>=0 and v>=p_v) else
                       STYLE["bull_fade"] if v>=0 else
                       STYLE["bear"] if v<=p_v else STYLE["bear_fade"])
    ax7.bar(xs,hist_v,color=bar_col,width=0.8,alpha=0.9,zorder=2)
    ax7.plot(xs,sv(macd_line,idx),color=STYLE["macd_line"],lw=1.3,label="MACD",zorder=3)
    ax7.plot(xs,sv(macd_sig,idx), color=STYLE["macd_sig"], lw=1.3,ls="--",label="Señal",zorder=3)
    ax7.axhline(0,color=STYLE["zero"],lw=0.7)
    ax7.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["macd_line"],STYLE["macd_sig"]])
    panel_style(ax7,ylabel="MACD"); format_xaxis(ax7,idx)
    ax7.tick_params(labelbottom=True)
    ax7.set_title("MACD  12/26/9",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    # PANEL 8 — ESTOCÁSTICO (18, 5, 9)
    ax8=axes[8]; sk_v=sv(stoch_k,idx); sd_v=sv(stoch_d,idx)
    ax8.fill_between(xs,20,alpha=0.08,color=STYLE["bull"],zorder=0)
    ax8.fill_between(xs,80,alpha=0.08,color=STYLE["bear"],zorder=0)
    ax8.plot(xs,sk_v,color=STYLE["stoch_k"],lw=1.4,label="%K (18,5)",zorder=3)
    ax8.plot(xs,sd_v,color=STYLE["stoch_d"],lw=1.4,ls="--",label="%D (9)",zorder=3)
    ax8.axhline(80,color=STYLE["bear"],lw=0.7,ls="--",alpha=0.6)
    ax8.axhline(20,color=STYLE["bull"],lw=0.7,ls="--",alpha=0.6)
    ax8.axhline(50,color=STYLE["muted"],lw=0.5,ls=":",alpha=0.3)
    for i in range(1,len(sk_v)):
        if np.isnan(sk_v[i]) or np.isnan(sk_v[i-1]) or np.isnan(sd_v[i]) or np.isnan(sd_v[i-1]): continue
        if sk_v[i-1] <= sd_v[i-1] and sk_v[i] > sd_v[i] and sk_v[i] < 40:
            ax8.annotate("▲", xy=(i, sk_v[i]), fontsize=8, color="#00e676", ha="center",
                         fontweight="bold", xytext=(0,8), textcoords="offset points")
    ax8.set_ylim(-2,102); ax8.yaxis.set_ticks([0,20,40,60,80,100])
    ax8.legend(loc="upper left",fontsize=7,frameon=False,labelcolor=[STYLE["stoch_k"],STYLE["stoch_d"]])
    panel_style(ax8,ylabel="Stoch"); format_xaxis(ax8,idx)
    ax8.tick_params(labelbottom=True)
    ax8.set_title("Estocástico  (18, 5, 9)  ·  ▲ cruce alcista zona 20-40",fontsize=9,color=STYLE["muted"],loc="right",pad=4)

    for ax in axes: ax.set_xlim(-1,len(idx))

    x0=0.07; gap=(0.97-x0)/len(sigs)
    for i,s in enumerate(sigs):
        col=(STYLE["bull"] if s["state"]=="bull" else STYLE["bear"] if s["state"]=="bear" else STYLE["muted"])
        fig.text(x0+i*gap+(gap-0.003)/2, 0.012, s["label"],
                 fontsize=7.5,color=col,ha="center",va="center",
                 bbox=dict(boxstyle="round,pad=0.3",facecolor=STYLE["panel"],edgecolor=col+"66",linewidth=0.8))

    # Save to PNG bytes and close figure immediately
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE STYLES
# ══════════════════════════════════════════════════════════════════════════════

def style_df(df):
    return df.style.set_properties(**{'background-color':'#13161e','color':'#ffffff'})

with st.sidebar:
    st.markdown("## ⚙️ Confluence Cucumber")
    st.markdown("---")
    grupo_sel    = st.selectbox("Grupo de tickers",options=list(GRUPOS.keys()),index=0,key="grupo")
    custom_input = st.text_area("Tickers personalizados\n(uno por línea)",height=120,key="custom_tickers")
    force_refresh_tab1 = st.button("🔄 Recalcular dashboard",key="refresh1")
    st.markdown("---")
    st.caption('Datos: Yahoo Finance · Caché 1h · Escáner diario y siete indicadores')

tab1, tab2 = st.tabs([
    "📊  DASHBOARD SEÑALES",
    "📈  GRÁFICOS ESTRATEGIA",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dashboard Señales
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown('## 🥒 David Indicators Confluence V2')
    status_path=Path(__file__).parent/'snapshots/status.json'
    if status_path.exists():
        try:
            auto_status=json.loads(status_path.read_text(encoding='utf-8'))
            st.caption(f"Última revisión automática: {auto_status['checked_at']}")
            failed=[a for a in auto_status['assets'] if a['Estado']=='ERROR']
            if failed:
                st.warning(f"La última revisión no pudo comprobar {len(failed)} activos.")
                with st.expander('Errores de la revisión automática'):
                    st.dataframe(pd.DataFrame(failed),hide_index=True)
        except (ValueError,KeyError,OSError):
            st.warning('No se pudo leer el estado de la revisión automática.')

    tickers_tab1=list(dict.fromkeys([t.strip().upper() for t in custom_input.splitlines() if t.strip()]
                                   if custom_input.strip() else GRUPOS[grupo_sel]))
    if force_refresh_tab1:
        cached_dashboard.clear()
        cached_chart_data.clear()
    if st.button('🔎 Ejecutar escáner',type='primary') or force_refresh_tab1:
        with st.spinner(f'Calculando {len(tickers_tab1)} activos…'):
            st.session_state['dashboard_v2']=cached_dashboard(tuple(tickers_tab1))
            st.session_state['dashboard_tickers']=tuple(tickers_tab1)
    result=st.session_state.get('dashboard_v2')
    if result is None:
        st.info('Pulsa Ejecutar escáner para cargar el dashboard. Puedes abrir los gráficos directamente.')
    elif st.session_state.get('dashboard_tickers')!=tuple(tickers_tab1):
        st.info('Has cambiado los tickers. Ejecuta el escáner para actualizar los resultados.')
    else:
        st.caption(f"Calculado: {result['updated']:%d/%m/%Y %H:%M UTC} · Las velas en curso pueden cambiar.")
        st.subheader('Escáner estructural · señales recientes')
        window=st.slider('Antigüedad máxima (velas)',0,100,25)
        choices=st.multiselect('Mostrar activos con alguna de estas señales',list(LABELS.values()))
        table=scanner_table(result['events'],TICKER_NAMES,window)
        metrics=st.columns(5)
        metrics[0].metric('Procesados',len(result['indicators']))
        for box,(key,label) in zip(metrics[1:],LABELS.items()):
            box.metric(label,int(table['_'+key].sum()) if not table.empty else 0)
        if choices and not table.empty:
            selected=[key for key,label in LABELS.items() if label in choices]
            table=table[table[['_'+key for key in selected]].any(axis=1)]
        if table.empty: st.info('No hay señales para esta selección y ventana.')
        else:
            shown=table[[c for c in table if not c.startswith('_')]]
            st.dataframe(style_df(shown),width='stretch',hide_index=True)
            st.download_button('Descargar escáner CSV',shown.to_csv(index=False).encode('utf-8-sig'),
                               'scanner_estructural.csv','text/csv')
        st.caption('Orden: señal confirmada más reciente, después coincidencias. Fantasmas solos al final. '
                   'Coincidencias: tipos confirmados presentes en la ventana (0–3).')
        st.subheader('Seguimiento de indicadores')
        tracking=result['indicators']
        if not tracking.empty:
            phases=st.multiselect('Filtrar por fase 1D',sorted(tracking['Fase 1D'].unique()))
            if phases: tracking=tracking[tracking['Fase 1D'].isin(phases)]
            st.dataframe(style_df(tracking.drop(columns='Detalle')),width='stretch',hide_index=True)
            with st.expander('Detalle de los siete indicadores y divergencia naranja'):
                st.dataframe(tracking[['Ticker','Detalle']],width='stretch',hide_index=True)
            st.caption('MACD · PVI · Media K · AO · Bitman · Azul Koncorde · Estocástico. '
                       'Fresca: cruce hace 0–3 velas y condición todavía activa.')
        if not result['errors'].empty:
            with st.expander('Activos sin datos o con errores'):
                st.dataframe(result['errors'],hide_index=True)

with tab2:
    st.markdown("## 📈 Gráficos Estrategia")
    st.caption("Selecciona un ticker y pulsa 'Ver gráfico' para renderizar. Se genera 1 gráfico por vez para optimizar memoria.")

    col_a, col_b, col_c = st.columns([4, 2, 2])
    with col_a:
        chart_ticker = st.selectbox("📌 Ticker", options=sorted(ALL_TICKERS),
                                    index=sorted(ALL_TICKERS).index("BTC-USD") if "BTC-USD" in ALL_TICKERS else 0,
                                    key="chart_ticker")
    with col_b:
        chart_interval = st.selectbox("Intervalo",options=list(INTERVAL_CONFIG.keys()),index=0,key="chart_interval")
    with col_c:
        zoom_candles = st.selectbox("Zoom",options=[50,100,150,200,252],index=3,key="zoom_candles")

    if chart_interval in ("1h","4h"):
        st.warning(f"⚠️ Intervalo **{chart_interval}**: indicadores de largo plazo orientativos.")

    nombre = TICKER_NAMES.get(chart_ticker, "")
    titulo = f"{chart_ticker}  ({nombre})" if nombre else chart_ticker

    if st.button(f"📊 Ver gráfico de {chart_ticker}", key="render_chart", type="primary"):
        with st.spinner(f"Calculando {chart_ticker}…"):
            try:
                chart_data = cached_chart_data(chart_ticker, interval_key=chart_interval)
            except Exception as e:
                st.error(f"Error calculando **{chart_ticker}**: {e}")
                chart_data = {}

        if not chart_data:
            st.error(f"Sin datos para **{chart_ticker}**.")
        else:
            # Metrics row
            df_c = chart_data["df"]; close = df_c["Close"]
            m1,m2,m3,m4,m5 = st.columns(5)
            last_p = close.iloc[-1]; prev_p = close.iloc[-2]
            chg = last_p - prev_p; pct_c = chg / prev_p * 100
            m1.metric("Último precio",f"{last_p:.2f}",f"{chg:+.2f} ({pct_c:+.2f}%)")
            rsi_val = chart_data["rsi_s"].iloc[-1]
            m2.metric("RSI 14",f"{rsi_val:.1f}",
                      "Sobrecompra" if rsi_val>70 else("Sobreventa" if rsi_val<30 else "Neutral"))
            bbwp_v = chart_data["bbwp_s"].dropna()
            bbwp_l = bbwp_v.iloc[-1] if len(bbwp_v)>0 else np.nan
            m3.metric("BBWP 13/252",
                      f"{bbwp_l:.1f}%" if not np.isnan(bbwp_l) else "n/d",
                      "compresión" if bbwp_l<20 else("expansión" if bbwp_l>80 else "normal"))
            mcg_v = chart_data["mcg25"].iloc[-1]
            m4.metric("McGinley 25",f"{mcg_v:.2f}","↑ sobre MCG" if last_p>mcg_v else "↓ bajo MCG")
            e200_v = chart_data["ema200"].iloc[-1]
            m5.metric("EMA 200",f"{e200_v:.2f}","↑ sobre EMA" if last_p>e200_v else "↓ bajo EMA")

            sk_v = chart_data["stoch_k"].iloc[-1]
            sd_v = chart_data["stoch_d"].iloc[-1]
            st.caption(f"📈 Estocástico: %K={sk_v:.1f}  %D={sd_v:.1f}  "
                       f"{'🟢 Cruce alcista en zona baja' if sk_v > sd_v and sk_v < 40 else ''}")

            # Render chart to PNG bytes (no st.pyplot — saves memory)
            with st.spinner(f"Renderizando {chart_ticker}…"):
                png_bytes = build_figure(chart_data, chart_ticker, n_candles=zoom_candles)

            st.image(png_bytes, width="stretch")

            # Signal pills
            sigs = build_signals(chart_data)
            pct_s, label_s, bull_n, total_n = score_signals(sigs)
            cols_sig = st.columns(len(sigs))
            for col, s in zip(cols_sig, sigs):
                emoji = "🟢" if s["state"]=="bull" else("🔴" if s["state"]=="bear" else "⚪")
                col.markdown(f"<div style='text-align:center;font-size:.7rem;color:#fff'>{emoji}<br>{s['label']}</div>",
                             unsafe_allow_html=True)
            score_col = "green" if pct_s>=60 else("red" if pct_s<40 else "orange")
            st.markdown(f"<h4 style='color:{score_col};text-align:center'>{label_s}  ·  {bull_n}/{total_n}  ({pct_s}%)</h4>",
                        unsafe_allow_html=True)

            # Download button
            st.download_button(
                label=f"⬇️ Descargar PNG — {chart_ticker}",
                data=png_bytes,
                file_name=f"confluence_{chart_ticker.replace('=','').replace('-','_')}.png",
                mime="image/png",
                key=f"dl_{chart_ticker}_{chart_interval}")