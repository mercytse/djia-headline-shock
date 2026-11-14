from pathlib import Path
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf

nltk.download("vader_lexicon", quiet=True)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Combined_News_DJIA.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets"
OUTPUT_DIR.mkdir(exist_ok=True)

news_df = pd.read_csv(DATA_PATH)
news_df["Date"] = pd.to_datetime(news_df["Date"])
news_df = news_df.sort_values("Date").reset_index(drop=True)

headline_cols = [f"Top{i}" for i in range(1, 26)]
for col in headline_cols:
    # Strip legacy Python 2 byte-string prefixes present in the raw Kaggle CSV
    news_df[col] = news_df[col].astype(str).str.replace(r'^b["\']|["\']$', "", regex=True)

sia = SentimentIntensityAnalyzer()
daily_stats = []

for _, row in news_df.iterrows():
    scores = [
        sia.polarity_scores(row[col])["compound"]
        for col in headline_cols
        if row[col] != "nan"
    ]
    if scores:
        daily_stats.append(
            {
                "sent_mean": np.mean(scores),
                "sent_dispersion": np.std(scores),
                "worst_headline": np.min(scores),
                "severe_neg_count": sum(s < -0.5 for s in scores),
            }
        )
    else:
        daily_stats.append(
            {
                "sent_mean": 0.0,
                "sent_dispersion": 0.0,
                "worst_headline": 0.0,
                "severe_neg_count": 0,
            }
        )

news_df = pd.concat([news_df, pd.DataFrame(daily_stats)], axis=1)

# Pull VIX and shift by 1 day so today's prediction only sees yesterday's close
vix_df = yf.download("^VIX", start="2008-08-01", end="2016-07-05", progress=False)
vix_close = vix_df["Close"].squeeze().reset_index()
vix_close.columns = ["Date", "vix_close"]
vix_close["Date"] = pd.to_datetime(vix_close["Date"]).dt.tz_localize(None)
vix_close["prev_vix"] = vix_close["vix_close"].shift(1)

market_df = pd.merge(news_df, vix_close[["Date", "prev_vix"]], on="Date", how="left").ffill().bfill()

# Markets price in chronic bad news; measure sudden deviations from a 5-day baseline
market_df["sent_shock"] = market_df["sent_mean"] - market_df["sent_mean"].rolling(5, min_periods=1).mean()
market_df["fear_amplified_shock"] = market_df["worst_headline"] * market_df["prev_vix"]
market_df["prev_day_label"] = market_df["Label"].shift(1).fillna(1)

# Chronological split prevents future market regimes from leaking into training
train_df = market_df[market_df["Date"] <= "2014-12-31"]
test_df = market_df[market_df["Date"] >= "2015-01-01"]

feature_cols = [
    "sent_mean",
    "sent_dispersion",
    "worst_headline",
    "severe_neg_count",
    "sent_shock",
    "prev_vix",
    "fear_amplified_shock",
]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
y_train = train_df["Label"].values

X_test = scaler.transform(test_df[feature_cols])
y_test = test_df["Label"].values

clf = LogisticRegression(C=0.1, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]

acc_always_up = accuracy_score(y_test, np.ones_like(y_test))
acc_momentum = accuracy_score(y_test, test_df["prev_day_label"].values)
acc_model = accuracy_score(y_test, preds)

print("--- Test Period Results (2015 - 2016) ---")
print(f"Always-Up Baseline:    {acc_always_up:.4f}")
print(f"Previous-Day Baseline: {acc_momentum:.4f}")
print(f"Shock + VIX Model:     {acc_model:.4f}\n")

weights = pd.Series(clf.coef_[0], index=feature_cols).sort_values()
print("--- Feature Weights ---")
print(weights.to_string())

# Selective abstention: evaluate accuracy when the model only trades high-conviction days
thresholds = np.linspace(0.50, 0.58, 9)
conv_acc = []
coverage = []

for thresh in thresholds:
    confident_mask = (probs >= thresh) | (probs <= (1 - thresh))
    coverage.append(confident_mask.mean() * 100)
    if confident_mask.sum() > 0:
        conv_acc.append(accuracy_score(y_test[confident_mask], preds[confident_mask]) * 100)
    else:
        conv_acc.append(np.nan)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(thresholds, conv_acc, marker="o", color="#1f77b4", label="Test Accuracy (%)")
ax1.axhline(acc_always_up * 100, color="gray", linestyle="--", label="Always-Up Baseline")
ax1.set_xlabel("Prediction Confidence Threshold")
ax1.set_ylabel("Accuracy (%)", color="#1f77b4")
ax1.set_ylim(45, 70)

ax2 = ax1.twinx()
ax2.bar(thresholds, coverage, width=0.006, alpha=0.25, color="#ff7f0e", label="Trading Days Covered (%)")
ax2.set_ylabel("Days Traded (%)", color="#ff7f0e")
ax2.set_ylim(0, 105)

plt.title("Selective Abstention: Accuracy vs. Trade Coverage (2015–2016)")
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_curve.png", dpi=200)
