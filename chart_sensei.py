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
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
h1 { color: #f0c27f; }
</style>
""", unsafe_allow_html=True)


# ─── Anthropic クライアント ───────────────────────────────
@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


# ─── データ取得 ───────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    t = yf.Ticker(ticker)
    monthly = t.history(period="max", interval="1mo")
    daily = t.history(period="6mo", interval="1d")
    info = t.info
    name = info.get("longName") or info.get("shortName") or ticker
    return monthly, daily, name


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
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#d1d5db"),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=0, r=10, t=50, b=0),
    )
    fig.update_xaxes(gridcolor="#1f2937", showgrid=True)
    fig.update_yaxes(gridcolor="#1f2937", showgrid=True)
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

st.title("📈 くにちゃんのチャート先生")
st.caption("チャートを自分で読む → 先生に答え合わせ。繰り返しで目が育つ。")

# ─── セッションステート初期化 ────────────────────────────
for key, val in [("show_answer", False), ("ai_result", None),
                 ("monthly_phase", None), ("daily_momentum", None),
                 ("last_ticker", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.header("🔍 銘柄を選ぶ")
    ticker_input = st.text_input(
        "銘柄コード（4桁）",
        placeholder="例: 6232",
        help="4桁の証券コードをそのまま入力してください",
    )
    st.markdown("---")
    st.markdown("""
**使い方**
1. 銘柄コード入力
2. 月足で長期局面を選択
3. 日足で短期勢いを選択
4. 先生に確認 → トレンドライン表示

---
**銘柄例**
- `7203` トヨタ
- `6758` ソニー
- `9984` ソフトバンクG
- `6232` ACSL
- `4755` 楽天グループ
""")

if not ticker_input:
    st.info("← 左のサイドバーに銘柄コードを入力してください")
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
# 答え合わせモード（先生ボタンを押した後）
# ══════════════════════════════════════════════════════════
if st.session_state.show_answer:
    # ページ先頭へ強制スクロール
    st.components.v1.html("""
<script>
(function() {
    // Streamlit のスクロール可能コンテナを全パターンで試す
    var selectors = [
        'section.main',
        '[data-testid="stAppViewContainer"]',
        '[data-testid="stMain"]',
        '.main',
        'html', 'body'
    ];
    function scrollTop() {
        selectors.forEach(function(sel) {
            var el = window.parent.document.querySelector(sel);
            if (el) el.scrollTop = 0;
        });
        window.parent.scrollTo(0, 0);
    }
    scrollTop();
    setTimeout(scrollTop, 100);
    setTimeout(scrollTop, 300);
})();
</script>
""", height=0)

    st.header(f"📊 {stock_name}（{ticker}）— 答え合わせ")
    st.markdown("### 💬 くにちゃん先生の解説（テクニカル分析）")

    monthly_correct, daily_correct, long_term, short_term, body_text = parse_judgments(st.session_state.ai_result)

    # ── 総合判断を大きく最上部に ──────────────────────────
    def verdict_color(v):
        return {"買い": ("#166534", "#bbf7d0"), "待ち": ("#78350f", "#fef3c7"), "売り": ("#7f1d1d", "#fecaca")}.get(v, ("#374151", "#f3f4f6"))

    if long_term or short_term:
        lt_bg, lt_fg = verdict_color(long_term)
        st_bg, st_fg = verdict_color(short_term)
        st.markdown(
            f'<div style="display:flex;gap:1rem;margin-bottom:1rem;">'
            f'<div style="flex:1;background:{lt_bg};border-radius:10px;padding:1rem 1.2rem;text-align:center;">'
            f'<div style="color:#9ca3af;font-size:0.8rem;margin-bottom:6px;">長期（月足）</div>'
            f'<div style="color:{lt_fg};font-size:2rem;font-weight:bold;">{long_term or "―"}</div>'
            f'</div>'
            f'<div style="flex:1;background:{st_bg};border-radius:10px;padding:1rem 1.2rem;text-align:center;">'
            f'<div style="color:#9ca3af;font-size:0.8rem;margin-bottom:6px;">短期（日足）</div>'
            f'<div style="color:{st_fg};font-size:2rem;font-weight:bold;">{short_term or "―"}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 正誤バッジ ────────────────────────────────────────
    def badge(label, result):
        if result == "正解":
            tag = '<span style="background:#166534;color:#bbf7d0;padding:2px 8px;border-radius:10px;font-weight:bold;margin-right:6px;">✅ 正解</span>'
        elif result == "不正解":
            tag = '<span style="background:#7f1d1d;color:#fecaca;padding:2px 8px;border-radius:10px;font-weight:bold;margin-right:6px;">❌ 不正解</span>'
        else:
            tag = ""
        return f'<div style="color:#f1f5f9;font-size:0.9rem;">{tag}{label}</div>'

    col_m, col_d = st.columns(2)
    with col_m:
        st.markdown(
            f'<div style="background:#1f2937;padding:0.7rem 1rem;border-radius:8px;">'
            f'<div style="color:#9ca3af;font-size:0.75rem;margin-bottom:4px;">あなたの長期判断</div>'
            f'{badge(st.session_state.monthly_phase, monthly_correct)}</div>',
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            f'<div style="background:#1f2937;padding:0.7rem 1rem;border-radius:8px;">'
            f'<div style="color:#9ca3af;font-size:0.75rem;margin-bottom:4px;">あなたの短期判断</div>'
            f'{badge(st.session_state.daily_momentum, daily_correct)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 先生の解説本文 ────────────────────────────────────
    st.markdown(
        f'<div style="background:#1e3a5f;border-left:5px solid #f0c27f;padding:1.2rem 1.5rem;'
        f'border-radius:6px;font-size:1.05rem;line-height:1.9;color:#f1f5f9;">'
        f'{body_text.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    # ── スクロール誘導 ─────────────────────────────────────
    st.markdown("""
<div style="text-align:center;margin:1.2rem 0;padding:0.8rem;
            background:#1f2937;border-radius:8px;color:#9ca3af;font-size:0.95rem;">
    📉 下にスクロールすると、トレンドライン付きチャートで答え合わせができます ↓
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── トレンドライン付きチャート ─────────────────────────
    st.subheader("月足チャート（トレンドライン入り）")
    st.caption("黄丸＝基点 / 赤破線＝抵抗ライン / 緑破線＝支持ライン（平行チャネル）")
    monthly_fig = build_chart(monthly_df, f"月足（全期間 約{years}年）", show_trendline=True)
    st.plotly_chart(monthly_fig, use_container_width=True)

    st.subheader("日足チャート（トレンドライン入り）")
    st.caption("黄丸＝基点 / 赤破線＝抵抗ライン / 緑破線＝支持ライン（平行チャネル）")
    daily_fig = build_chart(daily_df, "日足（6ヶ月）", show_trendline=True)
    st.plotly_chart(daily_fig, use_container_width=True)

    st.divider()
    if st.button("🔄 もう一度挑戦する", type="secondary", use_container_width=True):
        st.session_state.show_answer = False
        st.session_state.ai_result = None
        st.rerun()

# ══════════════════════════════════════════════════════════
# 問題モード（通常表示、トレンドラインなし）
# ══════════════════════════════════════════════════════════
else:
    st.header(f"📊 {stock_name}（{ticker}）")

    # STEP 1: 月足
    st.subheader("STEP 1｜月足チャート — 長期トレンドを読む")
    monthly_fig = build_chart(monthly_df, f"月足（全期間 約{years}年）", show_trendline=False)
    st.plotly_chart(monthly_fig, use_container_width=True)

    st.markdown("#### 📍 今の長期ポジション、どれだと思う？")
    monthly_phase = st.radio("長期局面", monthly_options, key="monthly_phase_q",
                             horizontal=True, label_visibility="collapsed")

    st.divider()

    # STEP 2: 日足
    st.subheader("STEP 2｜日足チャート — 短期の勢いを読む")
    daily_fig = build_chart(daily_df, "日足（6ヶ月）", show_trendline=False)
    st.plotly_chart(daily_fig, use_container_width=True)

    st.markdown("#### ⚡ 今の短期の勢い、どれだと思う？")
    daily_momentum = st.radio("短期勢い", daily_options, key="daily_momentum_q",
                              horizontal=True, label_visibility="collapsed")

    st.divider()

    # STEP 3: ボタン
    st.subheader("STEP 3｜くにちゃん先生に確認する")
    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        ask_btn = st.button("🎯 先生に聞いてみる！", type="primary", use_container_width=True)
    with col_hint:
        st.caption("自分の判断を先に決めてからボタンを押そう。それが一番勉強になる。")

    if ask_btn:
        with st.spinner("くにちゃん先生が分析中..."):
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

        # セッションに保存してページ先頭へ
        st.session_state.ai_result = result
        st.session_state.monthly_phase = monthly_phase
        st.session_state.daily_momentum = daily_momentum
        st.session_state.show_answer = True
        st.rerun()
