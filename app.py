# ============================================================
# ARAM AI QUANT
# Persian Crypto Market Intelligence Dashboard
# ============================================================

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from datetime import datetime, timedelta, timezone
from plotly.subplots import make_subplots

# ---------------- CONFIG ----------------

APP_NAME = "ARAM AI QUANT"
NOBITEX = "https://apiv2.nobitex.ir"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
.block-container {
    padding-top: 1.5rem;
}
.signal-box {
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #777;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION ----------------

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []


# ---------------- API ----------------

@st.cache_data(ttl=15, show_spinner=False)
def api_get(path, params=None):
    try:
        response = requests.get(
            NOBITEX + path,
            params=params,
            timeout=12,
            headers={"User-Agent": "ARAM-AI-QUANT/1.0"}
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def get_market_stats():
    data = api_get("/market/stats")

    if not isinstance(data, dict):
        return {}

    stats = data.get("stats", {})

    return stats if isinstance(stats, dict) else {}


@st.cache_data(ttl=30, show_spinner=False)
def get_orderbook(symbol):
    data = api_get("/v3/orderbook/" + symbol)

    if not isinstance(data, dict):
        return {}

    return data


def get_candles(symbol, resolution="60", days=7):
    end = int(time.time())
    start = end - days * 86400

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start,
        "to": end
    }

    data = api_get("/market/udf/history", params)

    if not isinstance(data, dict):
        return pd.DataFrame()

    if data.get("s") != "ok":
        return pd.DataFrame()

    try:
        df = pd.DataFrame({
            "time": pd.to_datetime(
                data["t"], unit="s", utc=True
            ),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna().sort_values("time").reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# ---------------- INDICATORS ----------------

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_indicators(df):
    df = df.copy()

    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA21"] = df["close"].ewm(span=21).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    df["RSI"] = calculate_rsi(df["close"])

    df["VolumeMA"] = df["volume"].rolling(20).mean()

    df["Support"] = df["low"].rolling(20).min()
    df["Resistance"] = df["high"].rolling(20).max()

    df["Return"] = df["close"].pct_change()

    df["Volatility"] = (
        df["Return"].rolling(20).std()
    )

    return df


# ---------------- ORDER BOOK ----------------

def analyze_orderbook(book):
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    def total_value(rows):
        total = 0.0

        for row in rows:
            try:
                price = float(row[0])
                amount = float(row[1])
                total += price * amount
            except (ValueError, TypeError, IndexError):
                continue

        return total

    bid_value = total_value(bids)
    ask_value = total_value(asks)

    total = bid_value + ask_value

    imbalance = (
        (bid_value - ask_value) / total
        if total > 0 else 0
    )

    return {
        "bid_value": bid_value,
        "ask_value": ask_value,
        "imbalance": imbalance
    }


# ---------------- SMART MONEY ENGINE ----------------

def analyze_market(df, book):
    if df.empty or len(df) < 30:
        return {
            "signal": "صبر",
            "score": 0,
            "reason": "دادهٔ کافی برای تحلیل وجود ندارد",
            "entry": None,
            "stop": None,
            "targets": [],
            "risk_reward": None
        }

    df = calculate_indicators(df)
    last = df.iloc[-1]

    price = float(last["close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else 50

    support = float(last["Support"])
    resistance = float(last["Resistance"])

    volume_avg = last["VolumeMA"]

    volume_ratio = (
        float(last["volume"]) / float(volume_avg)
        if pd.notna(volume_avg) and volume_avg > 0
        else 1
    )

    ob = analyze_orderbook(book)

    score = 50
    reasons = []

    # Trend
    if ema9 > ema21:
        score += 10
        reasons.append("EMA9 بالاتر از EMA21 است")

    else:
        score -= 10
        reasons.append("EMA9 پایین‌تر از EMA21 است")

    if price > ema50:
        score += 8
        reasons.append("قیمت بالاتر از EMA50 است")

    else:
        score -= 8
        reasons.append("قیمت پایین‌تر از EMA50 است")

    # RSI
    if 50 <= rsi <= 68:
        score += 8
        reasons.append("مومنتوم RSI نسبتاً مثبت است")

    elif rsi > 75:
        score -= 8
        reasons.append("احتمال داغ‌شدن بیش از حد بازار وجود دارد")

    elif rsi < 30:
        reasons.append("RSI پایین است؛ برگشت قیمت قطعی نیست")

    # Volume
    if volume_ratio > 1.5:
        score += 8
        reasons.append("حجم معاملات از میانگین بیشتر است")

    # Orderbook
    if ob["imbalance"] > 0.15:
        score += 8
        reasons.append("ارزش سفارش‌های خرید بیشتر است")

    elif ob["imbalance"] < -0.15:
        score -= 8
        reasons.append("ارزش سفارش‌های فروش بیشتر است")

    score = max(0, min(100, int(score)))

    # Entry and exit calculations
    volatility = (
        float(last["Volatility"])
        if pd.notna(last["Volatility"])
        else 0.02
    )

    volatility = max(0.005, min(volatility, 0.10))

    support_distance = max(0, price - support)

    stop_distance = max(
        price * volatility * 1.5,
        support_distance * 0.5,
        price * 0.01
    )

    stop = price - stop_distance

    targets = [
        price + stop_distance * 1.5,
        price + stop_distance * 2.0,
        price + stop_distance * 3.0
    ]

    entry = price

    if score >= 75 and ema9 > ema21 and price > ema50:
        signal = "بررسی خرید"

    elif score <= 35:
        signal = "بررسی فروش یا کاهش ریسک"

    else:
        signal = "صبر"

    risk_reward = (
        (targets[0] - entry) / (entry - stop)
        if entry > stop else None
    )

    return {
        "signal": signal,
        "score": score,
        "reason": "؛ ".join(reasons),
        "entry": entry,
        "stop": stop,
        "targets": targets,
        "support": support,
        "resistance": resistance,
        "rsi": rsi,
        "volume_ratio": volume_ratio,
        "orderbook": ob,
        "risk_reward": risk_reward
    }


# ---------------- 12 ANALYST ROLES ----------------

AGENTS = [
    "تحلیل‌گر روند",
    "تحلیل‌گر مومنتوم",
    "تحلیل‌گر RSI",
    "تحلیل‌گر حجم معاملات",
    "تحلیل‌گر دفتر سفارش‌ها",
    "تحلیل‌گر حمایت و مقاومت",
    "تحلیل‌گر نوسان",
    "تحلیل‌گر نقدینگی",
    "تحلیل‌گر شکست قیمت",
    "تحلیل‌گر مدیریت ریسک",
    "تحلیل‌گر سناریوهای بازار",
    "ترکیب‌کنندهٔ نتایج"
]


def build_agent_report(df, analysis):
    if df.empty or len(df) < 30:
        return pd.DataFrame({
            "عامل": AGENTS,
            "وضعیت": ["دادهٔ ناکافی"] * len(AGENTS)
        })

    last = calculate_indicators(df).iloc[-1]

    reports = [
        "EMA9 و EMA21",
        f"RSI: {analysis.get('rsi', 0):.1f}",
        "بررسی RSI",
        f"نسبت حجم: {analysis.get('volume_ratio', 1):.2f}",
        "دفتر سفارش‌ها" if analysis.get("orderbook") else "بدون داده",
        "حمایت و مقاومت",
        "برآورد نوسان",
        "عدم اتصال به دادهٔ مستقل نقدینگی",
        "بررسی قیمت‌های اخیر",
        "محاسبهٔ حد ضرر",
        "سناریوی آماری؛ نه پیش‌بینی قطعی",
        f"امتیاز ترکیبی: {analysis.get('score', 0)}"
    ]

    return pd.DataFrame({
        "عامل": AGENTS,
        "وضعیت": reports
    })


# ---------------- CHART ----------------

def draw_chart(df, analysis, show_projection=True):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28]
    )

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="قیمت"
        ),
        row=1,
        col=1
    )

    indicators = calculate_indicators(df)

    fig.add_trace(
        go.Scatter(
            x=indicators["time"],
            y=indicators["EMA9"],
            name="EMA9"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=indicators["time"],
            y=indicators["EMA21"],
            name="EMA21"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["volume"],
            name="حجم"
        ),
        row=2,
        col=1
    )

    if analysis.get("entry") is not None:
        fig.add_hline(
            y=analysis["entry"],
            line_dash="dot",
            annotation_text="قیمت مرجع ورود",
            row=1,
            col=1
        )

        fig.add_hline(
            y=analysis["stop"],
            line_dash="dash",
            annotation_text="حد ضرر",
            row=1,
            col=1
        )

        for i, target in enumerate(analysis["targets"], 1):
            fig.add_hline(
                y=target,
                line_dash="dot",
                annotation_text=f"هدف {i}",
                row=1,
                col=1
            )

    # Statistical scenario only, not a guaranteed prediction.
    if show_projection and len(df) >= 20:
        returns = df["close"].pct_change().dropna().tail(20)

        if not returns.empty:
            drift = float(returns.mean())
            price = float(df["close"].iloc[-1])

            future_times = [
                df["time"].iloc[-1] + timedelta(hours=i)
                for i in range(1, 7)
            ]

            future_prices = [
                price * ((1 + drift) ** i)
                for i in range(1, 7)
            ]

            fig.add_trace(
                go.Scatter(
                    x=[df["time"].iloc[-1]] + future_times,
                    y=[price] + future_prices,
                    name="سناریوی آماری",
                    line=dict(dash="dash", width=3)
                ),
                row=1,
                col=1
            )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ---------------- MARKET GIANTS ----------------

GIANTS = [
    {
        "name": "BlackRock",
        "type": "مدیریت دارایی",
        "data": "نیازمند اتصال به منابع رسمی و داده‌های ETF"
    },
    {
        "name": "Franklin Templeton",
        "type": "مدیریت دارایی",
        "data": "نیازمند اتصال به گزارش‌ها و محصولات رسمی"
    },
    {
        "name": "Fidelity",
        "type": "مدیریت دارایی",
        "data": "نیازمند دادهٔ معتبر صندوق‌ها"
    },
    {
        "name": "Vanguard",
        "type": "مدیریت دارایی",
        "data": "نیازمند بررسی گزارش‌های رسمی"
    },
    {
        "name": "JPMorgan",
        "type": "بانک و خدمات مالی",
        "data": "نیازمند منابع خبری و گزارش‌های رسمی"
    },
    {
        "name": "Goldman Sachs",
        "type": "بانک سرمایه‌گذاری",
        "data": "نیازمند منابع خبری و گزارش‌های رسمی"
    },
    {
        "name": "Morgan Stanley",
        "type": "خدمات مالی",
        "data": "نیازمند گزارش‌های رسمی"
    },
    {
        "name": "Strategy",
        "type": "شرکت دارای خزانهٔ بیت‌کوین",
        "data": "نیازمند بررسی افشاگری‌های رسمی"
    },
    {
        "name": "Binance",
        "type": "صرافی رمزارز",
        "data": "نیازمند دادهٔ بازار و جریان‌های عمومی"
    },
    {
        "name": "Coinbase",
        "type": "صرافی رمزارز",
        "data": "نیازمند دادهٔ بازار و گزارش‌های عمومی"
    }
]


# ---------------- MAIN UI ----------------

st.title("📊 ARAM AI QUANT")
st.caption(
    "سامانهٔ تحلیل بازار، سناریوهای قیمت و مدیریت ریسک"
)

with st.sidebar:
    st.header("تنظیمات")

    refresh = st.checkbox(
        "تازه‌سازی خودکار هر ۳۰ ثانیه",
        value=False
    )

    days = st.selectbox(
        "تاریخچهٔ نمودار",
        [1, 3, 7, 14, 30],
        index=2
    )

    interval = st.selectbox(
        "بازهٔ کندل",
        ["15", "30", "60", "240", "1D"],
        index=2
    )

    show_projection = st.checkbox(
        "نمایش سناریوی احتمالی",
        value=True
    )

    st.warning(
        "سیگنال‌ها احتمالی‌اند و تضمین سود ندارند."
    )

if refresh:
    st.session_state.last_refresh = time.time()
    st.markdown(
        '<meta http-equiv="refresh" content="30">',
        unsafe_allow_html=True
    )

with st.spinner("دریافت اطلاعات بازار نوبیتکس..."):
    stats = get_market_stats()

if not stats:
    st.error(
        "دریافت داده از نوبیتکس ناموفق بود. "
        "اتصال اینترنت، دسترسی API یا محدودیت سرویس را بررسی کن."
    )

    st.stop()

market_names = sorted(stats.keys())

selected_symbol = st.selectbox(
    "بازار موردنظر را انتخاب کن",
    market_names
)

market = stats.get(selected_symbol, {})

# Find available last price
last_price = None

for key in ["latest", "lastTradePrice", "last"]:
    try:
        if market.get(key) is not None:
            last_price = float(market[key])
            break
    except (TypeError, ValueError):
        pass

change = market.get("dayChange", None)
low = market.get("dayLow", None)
high = market.get("dayHigh", None)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "قیمت آخر",
    f"{last_price:,.4f}" if last_price is not None else "نامشخص"
)

try:
    c2.metric("تغییر روزانه", f"{float(change):.2f}%")
except (TypeError, ValueError):
    c2.metric("تغییر روزانه", "نامشخص")

try:
    c3.metric("کمترین روز", f"{float(low):,.4f}")
except (TypeError, ValueError):
    c3.metric("کمترین روز", "نامشخص")

try:
    c4.metric("بیشترین روز", f"{float(high):,.4f}")
except (TypeError, ValueError):
    c4.metric("بیشترین روز", "نامشخص")

with st.spinner("دریافت کندل‌ها و تحلیل بازار..."):
    candles = get_candles(
        selected_symbol,
        resolution=interval,
        days=days
    )

if candles.empty:
    st.error(
        "دادهٔ نمودار دریافت نشد. ممکن است نماد یا بازهٔ زمانی "
        "در سرویس فعلی پشتیبانی نشود."
    )

    st.stop()

book = get_orderbook(selected_symbol)
analysis = analyze_market(candles, book)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 نمودار",
    "🎯 ورود و خروج",
    "🐋 سفارش‌ها",
    "🏦 بازار بزرگان",
    "🧠 تحلیل‌گرها"
])

# ---------------- CHART TAB ----------------

with tab1:
    st.subheader(f"نمودار {selected_symbol}")

    st.plotly_chart(
        draw_chart(candles, analysis, show_projection),
        use_container_width=True
    )

    st.info(
        "خط‌چین فقط برآورد آماری بر اساس بازده اخیر است؛ "
        "پیش‌بینی هوش مصنوعی آموزش‌دیده یا تضمین قیمت آینده نیست."
    )

# ---------------- ENTRY / EXIT TAB ----------------

with tab2:
    st.subheader("تحلیل ورود و خروج")

    st.markdown(
        f"### وضعیت: **{analysis['signal']}**"
    )

    st.metric(
        "امتیاز تحلیلی",
        f"{analysis['score']} از ۱۰۰"
    )

    if analysis["entry"] is not None:
        st.write(f"**قیمت مرجع ورود:** {analysis['entry']:,.6f}")
        st.write(f"**حد ضرر تخمینی:** {analysis['stop']:,.6f}")

        for i, target in enumerate(analysis["targets"], 1):
            st.write(f"**هدف خروج {i}:** {target:,.6f}")

        if analysis["risk_reward"] is not None:
            st.write(
                f"**نسبت سود به ریسک تا هدف اول:** "
                f"{analysis['risk_reward']:.2f}"
            )

        st.write(
            f"**حمایت تخمینی:** {analysis['support']:,.6f}"
        )

        st.write(
            f"**مقاومت تخمینی:** {analysis['resistance']:,.6f}"
        )

        st.write(
            f"**RSI:** {analysis['rsi']:.2f}"
        )

        st.write(
            f"**نسبت حجم به میانگین:** "
            f"{analysis['volume_ratio']:.2f}"
        )

        st.write("**دلایل تحلیل:**")
        st.write(analysis["reason"])

    st.warning(
        "این قیمت‌ها تخمینی هستند. پیش از معامله، "
        "کارمزد، نقدشوندگی، لغزش قیمت و شرایط بازار را بررسی کن."
    )

# ---------------- ORDERBOOK TAB ----------------

with tab3:
    st.subheader("فشار سفارش‌های خرید و فروش")

    ob = analysis.get("orderbook", {})

    a, b, c = st.columns(3)

    a.metric(
        "ارزش سفارش‌های خرید",
        f"{ob.get('bid_value', 0):,.2f}"
    )

    b.metric(
        "ارزش سفارش‌های فروش",
        f"{ob.get('ask_value', 0):,.2f}"
    )

    c.metric(
        "عدم تعادل سفارش‌ها",
        f"{ob.get('imbalance', 0):.2%}"
    )

    st.write(
        "این اطلاعات فقط دفتر سفارش فعلی را نشان می‌دهد. "
        "سفارش بزرگ الزاماً به معنی فعالیت نهنگ یا معاملهٔ واقعی نیست."
    )

    if book:
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if bids:
            st.write("### سفارش‌های خرید")
            st.dataframe(
                pd.DataFrame(
                    bids[:10],
                    columns=["قیمت", "مقدار"]
                ),
                use_container_width=True
            )

        if asks:
            st.write("### سفارش‌های فروش")
            st.dataframe(
                pd.DataFrame(
                    asks[:10],
                    columns=["قیمت", "مقدار"]
                ),
                use_container_width=True
            )

# ---------------- GIANTS TAB ----------------

with tab4:
    st.subheader("🏦 بازار بزرگان و سرمایه‌گذاران نهادی")

    st.info(
        "این بخش فهرست بازیگران مهم بازار را نشان می‌دهد. "
        "تا زمانی که منبع رسمی به برنامه متصل نشده باشد، "
        "هیچ خرید یا فروش زنده‌ای به آن‌ها نسبت داده نمی‌شود."
    )

    st.dataframe(
        pd.DataFrame(GIANTS),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("چه داده‌هایی باید ردیابی شوند؟")

    st.markdown("""
    - جریان ورودی و خروجی صندوق‌های ETF رمزارزی
    - گزارش‌های رسمی خریدوفروش دارایی‌های شرکت‌ها
    - تغییرات عمومی دارایی‌های خزانهٔ شرکت‌ها
    - انتقال‌های بزرگ روی زنجیرهٔ بلاک‌چین
    - ورود و خروج دارایی به صرافی‌ها، در صورت وجود دادهٔ معتبر
    - حجم معاملات و نقدینگی بازار
    - اخبار رسمی مؤسسات و شرکت‌های بزرگ
    """)

    st.warning(
        "انتقال دارایی به صرافی لزوماً به معنی فروش نیست. "
        "همچنین نمی‌توان هویت صاحب هر کیف‌پول را از روی مبلغ "
        "انتقال با اطمینان تشخیص داد."
    )

# ---------------- ANALYSTS TAB ----------------

with tab5:
    st.subheader("🧠 پنل ۱۲ عامل تحلیل‌گر")

    report = build_agent_report(candles, analysis)

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "این پنل از شاخص‌های آماری و قواعد برنامه استفاده می‌کند. "
        "۱۲ مدل مستقل هوش مصنوعی هنوز به آن متصل نشده‌اند."
    )

# ---------------- PERSIAN ASSISTANT ----------------

st.divider()
st.subheader("💬 دستیار فارسی بازار")

question = st.text_input(
    "دربارهٔ ارز انتخاب‌شده چه می‌خواهی بدانی؟",
    placeholder="مثلاً وضعیت روند و نقطهٔ ورود چیست؟"
)

if st.button("تحلیل کن"):
    if not question.strip():
        st.warning("سؤالت را وارد کن.")

    else:
        st.write(f"**بازار:** {selected_symbol}")
        st.write(f"**وضعیت فعلی:** {analysis['signal']}")
        st.write(f"**امتیاز تحلیلی:** {analysis['score']} از ۱۰۰")
        st.write(f"**دلایل:** {analysis['reason']}")

        if analysis["entry"] is not None:
            st.write(
                f"**قیمت مرجع:** {analysis['entry']:,.6f}"
            )
            st.write(
                f"**حد ضرر:** {analysis['stop']:,.6f}"
            )

            st.write(
                f"**اهداف:** "
                f"{', '.join(f'{x:,.6f}' for x in analysis['targets'])}"
            )

        st.info(
            "این دستیار در نسخهٔ فعلی از تحلیل محاسباتی استفاده می‌کند. "
            "برای گفت‌وگوی هوشمند آزاد، باید یک مدل زبانی واقعی "
            "به برنامه متصل شود."
        )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    f"{APP_NAME} | زمان نمایش: "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

st.caption(
    "این نرم‌افزار ابزار تحلیل است، نه تضمین سود یا توصیهٔ مالی قطعی."
)
