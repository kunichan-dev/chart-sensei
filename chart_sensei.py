"""
くにちゃんのチャート先生
- トレンドライン: スイングハイ/ロー2点を結んでチャネル描画（プロ方式）
- AI結論: 「長期では〇〇＋短期では〇〇」形式
- claude-haiku でコスト最小化
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import pandas as pd
import numpy as np

# ─── ページ設定 ──────────────────────────────────────────
st.set_page_config(
    page_title="くにちゃんのチャート先生",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Nordic × Playful デザイン（Finland Blue）────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── 背景：Finland Blue ── */
.stApp {
    background: #0b2545;
}

/* ── サイドバー：少し明るめの青 ── */
section[data-testid="stSidebar"] {
    background: #13315c;
    border-right: 1px solid #1e4d8c;
}
/* サイドバー内テキストをすべて白に */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] td,
section[data-testid="stSidebar"] th {
    color: #e8f1ff !important;
}
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
}
/* サイドバーの入力欄 */
section[data-testid="stSidebar"] input {
    background: #1a4080 !important;
    border: 1px solid #4a90d9 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] input::placeholder {
    color: #a0c0e8 !important;
}

/* ── メインエリア ── */
section.main > div { padding-top: 1.5rem; }

/* ── 見出し ── */
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700;
}

/* ── 本文テキスト ── */
p, li, span, div {
    color: #e8f1ff;
}

/* ── ラジオボタン ── */
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
div[role="radiogroup"] { gap: 0.5rem; }
div[role="radiogroup"] label {
    background: #1a4080 !important;
    border: 1.5px solid #4a90d9 !important;
    border-radius: 20px !important;
    padding: 0.45rem 1rem !important;
    color: #ffffff !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: all 0.15s;
}
div[role="radiogroup"] label:hover {
    background: #2a5fa8 !important;
    border-color: #7ab8f5 !important;
    color: #ffffff !important;
}

/* ── プライマリボタン（コーラルオレンジ＝Playful） ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e85d26, #f07840) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.7rem 1.5rem !important;
    color: #ffffff !important;
    box-shadow: 0 3px 10px rgba(232,93,38,0.45);
    transition: transform 0.1s, box-shadow 0.1s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(232,93,38,0.55);
}

/* ── セカンダリボタン ── */
.stButton > button[kind="secondary"] {
    background: #1a4080 !important;
    border: 1.5px solid #4a90d9 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #2a5fa8 !important;
}

/* ── divider ── */
hr { border-color: #1e4d8c !important; }

/* ── caption ── */
div[data-testid="stCaptionContainer"] p {
    color: #a0c0e8 !important;
}

/* ── spinner ── */
div[data-testid="stSpinner"] { color: #7ab8f5 !important; }

/* ── step badge ── */
.step-badge {
    display: inline-block;
    background: #e85d26;
    color: white;
    border-radius: 50%;
    width: 26px; height: 26px;
    line-height: 26px;
    text-align: center;
    font-weight: 800;
    font-size: 0.85rem;
    margin-right: 8px;
}

/* ── info / error ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    background: #13315c !important;
    border-color: #4a90d9 !important;
    color: #ffffff !important;
}
div[data-testid="stAlert"] p { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ─── Anthropic クライアント ───────────────────────────────
@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


# ─── データ取得 ───────────────────────────────────────────
@st.cache_data(ttl=7200, show_spinner=False)
def fetch_stock(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    import time

    def clean(df):
        # yf.download はマルチレベルカラムになる場合がある
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(how="all")

    for attempt in range(4):
        try:
            monthly = clean(yf.download(
                ticker, period="max", interval="1mo",
                progress=False, auto_adjust=True
            ))
            time.sleep(1)
            daily = clean(yf.download(
                ticker, period="6mo", interval="1d",
                progress=False, auto_adjust=True
            ))
            # 銘柄名は取得失敗してもアプリは動くようにする
            try:
                time.sleep(1)
                info = yf.Ticker(ticker).fast_info
                name = getattr(info, "long_name", None) or ticker
            except Exception:
                name = ticker
            return monthly, daily, name
        except Exception as e:
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
            else:
                raise e


# ─── OHLC 要約（AI送信用）────────────────────────────────
def ohlc_summary(df: pd.DataFrame, label: str, n: int) -> str:
    d = df.tail(n).copy()
    close = d["Close"]
    cur = close.iloc[-1]
    chg = (cur - close.iloc[0]) / close.iloc[0] * 100
    hi = d["High"].max()
    lo = d["Low"].min()

    ma_strs = []
    for w in [5, 25, 75]:
        if len(close) >= w:
            mv = close.rolling(w).mean().iloc[-1]
            ma_strs.append(f"MA{w}:{mv:.0f}({'↑' if cur > mv else '↓'})")

    recent = "上昇" if close.iloc[-1] > close.iloc[-3] else "下降"

    x = np.arange(len(d))
    slope = np.polyfit(x, d["High"].values, 1)[0] + np.polyfit(x, d["Low"].values, 1)[0]
    tl = "上向き" if slope > 0 else "下向き"

    return "  ".join([
        f"[{label}/{n}本]", f"現値:{cur:.1f}", f"変化:{chg:+.1f}%",
        f"高:{hi:.1f} 安:{lo:.1f}", " ".join(ma_strs),
        f"直近3本:{recent}", f"トレンドライン:{tl}",
    ])


# ─── プロ方式トレンドライン ───────────────────────────────
def find_pivots(arr: np.ndarray, window: int, mode: str) -> list[int]:
    """スイングハイ / スイングローのインデックスを返す"""
    pivots = []
    for i in range(window, len(arr) - window):
        segment = arr[i - window: i + window + 1]
        if mode == "high" and arr[i] == segment.max():
            pivots.append(i)
        elif mode == "low" and arr[i] == segment.min():
            pivots.append(i)
    return pivots


def line_vals(x1, y1, x2, y2, n: int) -> np.ndarray:
    """2点を通る直線を n 点分計算（チャート全幅に延伸）"""
    slope = (y2 - y1) / (x2 - x1)
    return slope * np.arange(n) + (y1 - slope * x1)


def calc_channel(df: pd.DataFrame) -> dict | None:
    """
    プロ方式チャネルライン計算
    - 下降トレンド: 最高値スイングハイ → 次の有意な低いスイングハイ を結ぶ抵抗線
                  + そこから平行に引いた最安値通過の支持線
    - 上昇トレンド: 最安値スイングロー → 次の有意な高いスイングロー を結ぶ支持線
                  + そこから平行に引いた最高値通過の抵抗線
    """
    n = len(df)
    if n < 10:
        return None

    highs = df["High"].values
    lows = df["Low"].values
    close = df["Close"].values

    # ウィンドウサイズ（データ量に応じて調整）
    window = max(2, n // 15)
    min_gap = max(3, n // 10)   # 2点間の最小バー数
    min_pct = 0.03              # 最低限の値差（3%）

    ph = find_pivots(highs, window, "high")
    pl = find_pivots(lows,  window, "low")

    # 全体トレンド判定（後半60%の平均 vs 前半40%）
    cut = int(n * 0.4)
    is_up = np.mean(close[cut:]) > np.mean(close[:cut])

    if not is_up:
        # ── 下降トレンド ──
        # 最高値のスイングハイ(P1) → その後の有意な低いスイングハイ(P2)
        ph_by_val = sorted(ph, key=lambda i: highs[i], reverse=True)
        p1 = p2 = None
        for i1 in ph_by_val:
            for i2 in ph:
                if i2 > i1 + min_gap and highs[i2] < highs[i1] * (1 - min_pct):
                    p1, p2 = i1, i2
                    break
            if p1 is not None:
                break

        if p1 is None:
            return None

        resist = line_vals(p1, highs[p1], p2, highs[p2], n)
        # 平行サポート: 全期間の最安値を通るように平行移動
        lowest = int(np.argmin(lows))
        offset = resist[lowest] - lows[lowest]
        support = resist - offset

        return dict(
            trend="down",
            line1=resist,   line1_name="下降抵抗ライン", line1_color="#ff6b6b",
            line2=support,  line2_name="平行サポートライン", line2_color="#6bffb8",
            p1=p1, p2=p2,
        )

    else:
        # ── 上昇トレンド ──
        # 最安値のスイングロー(P1) → その後の有意な高いスイングロー(P2)
        pl_by_val = sorted(pl, key=lambda i: lows[i])
        p1 = p2 = None
        for i1 in pl_by_val:
            for i2 in pl:
                if i2 > i1 + min_gap and lows[i2] > lows[i1] * (1 + min_pct):
                    p1, p2 = i1, i2
                    break
            if p1 is not None:
                break

        if p1 is None:
            return None

        support = line_vals(p1, lows[p1], p2, lows[p2], n)
        # 平行レジスタンス: 全期間の最高値を通るように平行移動
        highest = int(np.argmax(highs))
        offset = highs[highest] - support[highest]
        resist = support + offset

        return dict(
            trend="up",
            line1=support,  line1_name="上昇支持ライン", line1_color="#6bffb8",
            line2=resist,   line2_name="平行抵抗ライン", line2_color="#ff6b6b",
            p1=p1, p2=p2,
        )


# ─── チャート描画 ─────────────────────────────────────────
def build_chart(df: pd.DataFrame, title: str, show_trendline: bool = False) -> go.Figure:
    n = len(df)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    # ロウソク足
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#ef5350",
        decreasing_line_color="#42a5f5",
        name="OHLC",
    ), row=1, col=1)

    # 移動平均
    for w, color, lw in [(5, "orange", 1.2), (25, "#da77f2", 1.2), (75, "#4dd0e1", 1.5)]:
        if n >= w:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Close"].rolling(w).mean(),
                name=f"MA{w}", line=dict(color=color, width=lw),
                hovertemplate=f"MA{w}: %{{y:.1f}}<extra></extra>",
            ), row=1, col=1)

    # プロ方式トレンドライン（答え合わせ時のみ表示）
    ch = calc_channel(df) if show_trendline else None
    if ch:
        for key, name_key, color_key in [
            ("line1", "line1_name", "line1_color"),
            ("line2", "line2_name", "line2_color"),
        ]:
            fig.add_trace(go.Scatter(
                x=df.index, y=ch[key],
                name=ch[name_key],
                line=dict(color=ch[color_key], width=1.8, dash="dash"),
                hovertemplate=f"{ch[name_key]}: %{{y:.1f}}<extra></extra>",
            ), row=1, col=1)

        # 2点にマーカー表示
        for pt_key, arr in [("p1", df["High"].values if ch["trend"] == "down" else df["Low"].values),
                             ("p2", df["High"].values if ch["trend"] == "down" else df["Low"].values)]:
            idx = ch[pt_key]
            fig.add_trace(go.Scatter(
                x=[df.index[idx]], y=[arr[idx]],
                mode="markers",
                marker=dict(color="yellow", size=8, symbol="circle"),
                name=f"基点{pt_key.upper()}", showlegend=False,
                hovertemplate=f"基点: %{{y:.1f}}<extra></extra>",
            ), row=1, col=1)

    # 出来高
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="出来高", marker_color="#546e7a", opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f0c27f")),
        height=460,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0d2d52", paper_bgcolor="#0b2545",
        font=dict(color="#d1d5db"),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=0, r=10, t=50, b=0),
    )
    fig.update_xaxes(gridcolor="#1e4d8c", showgrid=True, color="#a0c0e8")
    fig.update_yaxes(gridcolor="#1e4d8c", showgrid=True, color="#a0c0e8")
    return fig


# ─── AI 解説 ─────────────────────────────────────────────
def ask_haiku(name, monthly_sum, daily_sum, monthly_phase, daily_momentum) -> str:
    prompt = (
        f"銘柄:{name}\n"
        f"月足:{monthly_sum}\n  ユーザー月足判断:{monthly_phase}\n"
        f"日足:{daily_sum}\n  ユーザー日足判断:{daily_momentum}\n\n"
        "以下の形式で必ず答えよ。\n"
        "月足判定:[正解 or 不正解]\n"
        "日足判定:[正解 or 不正解]\n"
        "長期結論:[買い or 待ち or 売り]\n"
        "短期結論:[買い or 待ち or 売り]\n"
        "①長期（月足）：トレンドラインの状態と局面を1〜2文で。\n"
        "②短期（日足）：直近の勢いを1〜2文で。\n"
        "③根拠：上記2つの結論の理由を1〜2文で。"
    )
    resp = get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=(
            "あなたはテクニカル分析の先生くにちゃん。"
            "トレンドラインと局面を軸に、チャートを見ながら話すような自然な日本語で解説する。"
            "数値の羅列は避ける。"
            "出力の最初の4行は必ず指定の形式で出力すること。"
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def parse_judgments(text: str) -> tuple[str | None, str | None, str | None, str | None, str]:
    """
    AIレスポンスから正誤・長期/短期結論を抽出し、残りの解説本文を返す。
    Returns: (monthly_correct, daily_correct, long_term, short_term, body_text)
    """
    monthly_correct = daily_correct = long_term = short_term = None
    lines = text.splitlines()
    body_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("月足判定"):
            monthly_correct = "正解" if "正解" in s else "不正解"
        elif s.startswith("日足判定"):
            daily_correct = "正解" if "正解" in s else "不正解"
        elif s.startswith("長期結論"):
            for v in ["買い", "待ち", "売り"]:
                if v in s:
                    long_term = v
                    break
        elif s.startswith("短期結論"):
            for v in ["買い", "待ち", "売り"]:
                if v in s:
                    short_term = v
                    break
        else:
            body_lines.append(line)
    return monthly_correct, daily_correct, long_term, short_term, "\n".join(body_lines).strip()


# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:0.5rem 0 1.2rem;">
  <div style="font-size:1.6rem;font-weight:700;color:#e6edf3;letter-spacing:-0.5px;">
    📈 くにちゃんのチャート先生
  </div>
  <div style="color:#6e7681;font-size:0.9rem;margin-top:4px;">
    自分でチャートを読む → 先生に答え合わせ → 目が育つ
  </div>
</div>
""", unsafe_allow_html=True)

# ─── セッションステート初期化 ────────────────────────────
for key, val in [("show_answer", False), ("ai_result", None),
                 ("monthly_phase", None), ("daily_momentum", None),
                 ("last_ticker", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.markdown("### 🔍 銘柄を入力")
    ticker_input = st.text_input(
        "証券コード（4桁）",
        placeholder="例: 6232",
        help="4桁の証券コードをそのまま入力 → Enterで検索",
    )

    # スマホ：入力後にサイドバーを自動で閉じる
    if ticker_input:
        st.components.v1.html("""
<script>
(function() {
    function closeSidebar() {
        var btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
        if (!btn) btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if (btn) btn.click();
    }
    setTimeout(closeSidebar, 400);
})();
</script>
""", height=0)

    st.markdown("---")
    st.markdown("""
**使い方**
1. 証券コードを入力（4桁）
2. 月足で長期局面を選択
3. 日足で短期勢いを選択
4. 先生に確認 → 答え合わせ

---
**銘柄例**
| コード | 銘柄 |
|--------|------|
| 7203 | トヨタ |
| 6758 | ソニー |
| 9984 | ソフトバンクG |
| 6232 | ACSL |
| 4755 | 楽天 |
""")

if not ticker_input:
    st.markdown("""
<div style="background:#13315c;border:1px solid #2a6496;border-radius:14px;
            padding:2rem;text-align:center;margin-top:1rem;">
  <div style="font-size:2.5rem;margin-bottom:0.8rem;">📊</div>
  <div style="color:#e6edf3;font-size:1.1rem;font-weight:600;margin-bottom:0.4rem;">
    銘柄コードを入力してスタート
  </div>
  <div style="color:#6e7681;font-size:0.9rem;">
    左上の ≡ を押して、4桁の証券コードを入力してください
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

raw = ticker_input.strip()
ticker = (raw + ".T") if raw.isdigit() and len(raw) in (4, 5) else raw.upper()

# 銘柄が変わったらリセット
if ticker != st.session_state.last_ticker:
    st.session_state.show_answer = False
    st.session_state.ai_result = None
    st.session_state.last_ticker = ticker

with st.spinner(f"{ticker} のデータを取得中..."):
    try:
        monthly_df, daily_df, stock_name = fetch_stock(ticker)
    except Exception as e:
        if "rate" in str(e).lower() or "too many" in str(e).lower():
            st.error("データ取得に失敗しました。少し待ってから、ページを再読み込みしてください。（yfinanceのアクセス制限）")
        else:
            st.error(f"データ取得エラー: {e}")
        st.stop()

if monthly_df.empty or daily_df.empty:
    st.error("銘柄が見つかりません。コードをご確認ください。")
    st.stop()

years = (monthly_df.index[-1] - monthly_df.index[0]).days // 365
monthly_options = [
    "📈 上昇トレンド継続中",
    "📉 下降トレンド継続中",
    "🔼 下降→上昇への反転局面",
    "🔽 上昇→下降への反転局面",
    "↔️ レンジ（方向感なし）",
]
daily_options = [
    "🚀 強い上昇勢い（高値更新・大陽線）",
    "📗 緩やかな上昇（じわじわ上げ）",
    "➡️ 横ばい（売り買い拮抗）",
    "📕 緩やかな下降（じわじわ下げ）",
    "💥 強い下降勢い（安値更新・大陰線）",
]

# ══════════════════════════════════════════════════════════
# 答え合わせモード
# ══════════════════════════════════════════════════════════
if st.session_state.show_answer:
    # ページ先頭へスクロール
    st.components.v1.html("""
<script>
(function() {
    function scrollTop() {
        ['section.main','[data-testid="stAppViewContainer"]','.main','html','body'].forEach(function(sel){
            var el = window.parent.document.querySelector(sel);
            if (el) el.scrollTop = 0;
        });
        window.parent.scrollTo(0, 0);
    }
    scrollTop(); setTimeout(scrollTop, 150); setTimeout(scrollTop, 400);
})();
</script>
""", height=0)

    # 銘柄ヘッダー
    st.markdown(f"""
<div style="margin-bottom:1rem;">
  <span style="color:#6e7681;font-size:0.85rem;">答え合わせ</span>
  <div style="color:#e6edf3;font-size:1.3rem;font-weight:700;">{stock_name} <span style="color:#6e7681;font-size:1rem;">({ticker})</span></div>
</div>
""", unsafe_allow_html=True)

    monthly_correct, daily_correct, long_term, short_term, body_text = parse_judgments(st.session_state.ai_result)

    # ── 総合判断カード ────────────────────────────────────
    def vcolor(v):
        return {"買い":("#1a4731","#4ade80","#22c55e"), "待ち":("#3d2a00","#fcd34d","#f59e0b"), "売り":("#3d0f0f","#f87171","#ef4444")}.get(v, ("#1c2128","#8b949e","#6e7681"))

    lt_bg, lt_txt, lt_acc = vcolor(long_term)
    st_bg, st_txt, st_acc = vcolor(short_term)

    st.markdown(
        f'<div style="display:flex;gap:12px;margin-bottom:1rem;">'
        f'<div style="flex:1;background:{lt_bg};border:1px solid {lt_acc}33;border-radius:16px;padding:1.1rem;text-align:center;">'
        f'<div style="color:#8b949e;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">長期（月足）</div>'
        f'<div style="color:{lt_txt};font-size:2rem;font-weight:800;line-height:1;">{long_term or "―"}</div>'
        f'</div>'
        f'<div style="flex:1;background:{st_bg};border:1px solid {st_acc}33;border-radius:16px;padding:1.1rem;text-align:center;">'
        f'<div style="color:#8b949e;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">短期（日足）</div>'
        f'<div style="color:{st_txt};font-size:2rem;font-weight:800;line-height:1;">{short_term or "―"}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── あなたの判断 + 正誤 ───────────────────────────────
    def badge(label, result):
        if result == "正解":
            pill = '<span style="background:#1a4731;color:#4ade80;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:700;margin-right:6px;">✅ 正解</span>'
        elif result == "不正解":
            pill = '<span style="background:#3d0f0f;color:#f87171;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:700;margin-right:6px;">❌ 不正解</span>'
        else:
            pill = ""
        return f'<div style="color:#c9d1d9;font-size:0.88rem;line-height:1.5;">{pill}{label}</div>'

    col_m, col_d = st.columns(2)
    with col_m:
        st.markdown(
            f'<div style="background:#13315c;border:1px solid #2a6496;border-radius:12px;padding:0.8rem 1rem;">'
            f'<div style="color:#6e7681;font-size:0.72rem;font-weight:600;letter-spacing:0.04em;margin-bottom:5px;">あなたの長期判断</div>'
            f'{badge(st.session_state.monthly_phase, monthly_correct)}</div>',
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            f'<div style="background:#13315c;border:1px solid #2a6496;border-radius:12px;padding:0.8rem 1rem;">'
            f'<div style="color:#6e7681;font-size:0.72rem;font-weight:600;letter-spacing:0.04em;margin-bottom:5px;">あなたの短期判断</div>'
            f'{badge(st.session_state.daily_momentum, daily_correct)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── 先生の解説 ────────────────────────────────────────
    st.markdown(
        f'<div style="background:#13315c;border:1px solid #2a6496;border-left:4px solid #7ab8f5;'
        f'border-radius:12px;padding:1.2rem 1.4rem;font-size:0.95rem;line-height:1.85;color:#c9d1d9;">'
        f'<div style="color:#7ab8f5;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;margin-bottom:8px;">💬 くにちゃん先生の解説</div>'
        f'{body_text.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    # ── スクロール誘導 ────────────────────────────────────
    st.markdown("""
<div style="text-align:center;margin:1rem 0;padding:0.75rem;background:#161b22;
            border:1px solid #2d333b;border-radius:10px;color:#a0c0e8;font-size:0.88rem;">
    下にスクロールするとトレンドライン付きチャートで確認できます ↓
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── トレンドライン付きチャート ─────────────────────────
    st.markdown("#### 月足チャート（トレンドライン入り）")
    st.caption("黄丸＝基点 / 赤破線＝抵抗ライン / 緑破線＝支持ライン")
    monthly_fig = build_chart(monthly_df, f"月足（全期間 約{years}年）", show_trendline=True)
    st.plotly_chart(monthly_fig, use_container_width=True)

    st.markdown("#### 日足チャート（トレンドライン入り）")
    st.caption("黄丸＝基点 / 赤破線＝抵抗ライン / 緑破線＝支持ライン")
    daily_fig = build_chart(daily_df, "日足（6ヶ月）", show_trendline=True)
    st.plotly_chart(daily_fig, use_container_width=True)

    st.divider()
    if st.button("🔄 もう一度挑戦する", type="secondary", use_container_width=True):
        st.session_state.show_answer = False
        st.session_state.ai_result = None
        st.rerun()

# ══════════════════════════════════════════════════════════
# 問題モード
# ══════════════════════════════════════════════════════════
else:
    # 銘柄ヘッダー
    st.markdown(f"""
<div style="margin-bottom:1.2rem;">
  <div style="color:#e6edf3;font-size:1.3rem;font-weight:700;">
    {stock_name} <span style="color:#6e7681;font-size:1rem;">({ticker})</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # STEP 1: 月足
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.3rem;"><span class="step-badge">1</span><span style="color:#e6edf3;font-weight:600;font-size:1rem;">月足チャート — 長期トレンドを読む</span></div>', unsafe_allow_html=True)
    monthly_fig = build_chart(monthly_df, f"月足（全期間 約{years}年）", show_trendline=False)
    st.plotly_chart(monthly_fig, use_container_width=True)

    st.markdown('<div style="color:#c9d1d9;font-size:0.9rem;font-weight:600;margin-bottom:0.4rem;">📍 今の長期ポジション、どれだと思う？</div>', unsafe_allow_html=True)
    monthly_phase = st.radio("長期局面", monthly_options, key="monthly_phase_q",
                             horizontal=True, label_visibility="collapsed")

    st.divider()

    # STEP 2: 日足
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.3rem;"><span class="step-badge">2</span><span style="color:#e6edf3;font-weight:600;font-size:1rem;">日足チャート — 短期の勢いを読む</span></div>', unsafe_allow_html=True)
    daily_fig = build_chart(daily_df, "日足（6ヶ月）", show_trendline=False)
    st.plotly_chart(daily_fig, use_container_width=True)

    st.markdown('<div style="color:#c9d1d9;font-size:0.9rem;font-weight:600;margin-bottom:0.4rem;">⚡ 今の短期の勢い、どれだと思う？</div>', unsafe_allow_html=True)
    daily_momentum = st.radio("短期勢い", daily_options, key="daily_momentum_q",
                              horizontal=True, label_visibility="collapsed")

    st.divider()

    # STEP 3: ボタン
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.8rem;"><span class="step-badge">3</span><span style="color:#e6edf3;font-weight:600;font-size:1rem;">くにちゃん先生に確認する</span></div>', unsafe_allow_html=True)
    st.caption("自分の判断を先に決めてからボタンを押そう。それが一番勉強になる。")

    ask_btn = st.button("🎯 先生に聞いてみる！", type="primary", use_container_width=True)

    if ask_btn:
        with st.spinner("くにちゃん先生が分析中... ちょっと待ってね"):
            monthly_sum = ohlc_summary(monthly_df, "月足", n=12)
            daily_sum   = ohlc_summary(daily_df,   "日足", n=20)
            try:
                result = ask_haiku(stock_name, monthly_sum, daily_sum,
                                   monthly_phase, daily_momentum)
            except anthropic.AuthenticationError:
                st.error("ANTHROPIC_API_KEY が設定されていません。")
                st.stop()
            except Exception as e:
                st.error(f"AI 呼び出しエラー: {e}")
                st.stop()

        st.session_state.ai_result = result
        st.session_state.monthly_phase = monthly_phase
        st.session_state.daily_momentum = daily_momentum
        st.session_state.show_answer = True
        st.rerun()
