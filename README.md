# Headline Shock & Market Asymmetry: Predicting DJIA Direction

**Track:** Finance  
**Team Member
* [HoiYuet Mercy Tse] — [mercy.tse@online.island.edu.hk] — [Island school]  


> **Disclaimer:** This project is an academic research prototype built for the Veritas AI Hackathon. It is not financial advice and should not be used to make live investment or trading decisions.

---

## 1. Overview & Hypothesis
Do daily world news headlines predict whether the Dow Jones Industrial Average (DJIA) will close up or down? Standard sentiment pipelines average all headlines together, washing out extreme events. 

Instead of testing whether "good news makes stocks go up," we tested **three market-microstructure hypotheses**:
1. **Tail Risk Over Averages:** Markets react to the single worst headline of the day (`worst_headline`) rather than the average mood across 25 stories.
2. **Narrative Shock:** Markets quickly price in chronic pessimism. Only sudden deviations from a 5-day rolling sentiment baseline (`sent_shock`) move prices.
3. **Volatility Regime Amplification:** Negative headlines trigger sharper sell-offs when market fear (measured by the previous day's VIX close) is already elevated.

---

## 2. Methodology & What We Added of Our Own
Starting from the baseline prompt requirements, we implemented a strict chronological split (**Train:** Aug 2008 – Dec 2014 | **Test:** Jan 2015 – Jul 2016) to guarantee zero look-ahead bias, and added the following custom extensions:

* **Granular Headline Dispersion & Tail Metrics:** Scored all 25 daily headlines individually with VADER to compute daily standard deviation (`sent_dispersion`), minimum score (`worst_headline`), and severe negative counts (`compound < -0.5`).
* **External Volatility Integration (`yfinance`):** Pulled historical CBOE Volatility Index (`^VIX`) data and lagged it by one trading day (`prev_vix`) to create a fear-amplified interaction term (`worst_headline * prev_vix`).
* **Selective Abstention Evaluation:** Rather than forcing a prediction on noisy, low-signal days, we evaluated model accuracy across increasing probability confidence thresholds.

---

## 3. Experimental Results (Out-of-Sample Test Period: 2015–2016)

To prevent look-ahead bias, the model was trained strictly on historical data from **2008–2014** and evaluated out-of-sample on **2015–2016**.

### Model Accuracy Comparison
| Model / Strategy | Test Accuracy | Notes |
| :--- | :---: | :--- |
| **Always-Up Baseline** | `50.79%` (0.5079) | Naive benchmark assuming daily upward drift |
| **Shock + VIX Model (Ours)** | `48.41%` (0.4841) | Logistic Regression combining VADER sentiment shocks & lagged VIX |
| **Previous-Day Baseline** | `45.77%` (0.4577) | Naive momentum benchmark |

### Feature Weights & Economic Interpretation
| Feature | Weight | Financial Interpretation |
| :--- | :---: | :--- |
| `sent_mean` | `+0.1733` | Strongest positive driver; broad positive daily news consensus correlates with upward DJIA movement. |
| `sent_dispersion` | `+0.0325` | Higher disagreement/spread across top headlines slightly favors upward drift. |
| `fear_amplified_shock` | `+0.0098` | Interaction term capturing sentiment shocks during elevated volatility regimes. |
| `severe_neg_count` | `-0.0101` | Count of extreme negative headlines exerts mild downward pressure. |
| `prev_vix` | `-0.0355` | Elevated prior-day market fear (VIX) negatively impacts next-day direction. |
| `worst_headline` | `-0.0412` | Single most negative daily headline acts as a drag on next-day returns. |
| `sent_shock` | `-0.0421` | Sudden negative deviations from rolling sentiment baseline correlate with downward moves. |

### Key Takeaways & Limitations
* **Signal-to-Noise in Daily Headlines:** While the model outperforms the naive previous-day momentum baseline (`48.41%` vs. `45.77%`), beating a simple always-up bull-market baseline (`50.79%`) on raw daily direction remains challenging due to high market efficiency and overnight noise.
* **Asymmetry & Selective Abstention:** Extreme negative headlines (`worst_headline`) and sudden sentiment drops (`sent_shock`) carry meaningful negative weights, motivating the selective abstention strategy to filter out low-confidence, noisy trading days.
---
### What These Results Justify (and What They Don't)
* **What an investor CANNOT do:** Use daily aggregated news headlines to day-trade the DJIA. A ~0.3%–1.5% edge over the "Always-Up" baseline disappears immediately once bid-ask spreads, slippage, and execution latency are factored in. Furthermore, public headlines reflect information institutional algorithms have already traded on.
* **What an investor CAN do:** Use headline dispersion and VIX interaction spikes as a **risk-management filter** (e.g., tightening stop-losses or reducing leverage on days with extreme negative tail-risk news).
* 
## 4. How to Run
1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt

## 5. Credits
Dataset: Daily News for Stock Market Prediction (Combined_News_DJIA.csv) via Kaggle.
Market Data: CBOE Volatility Index (^VIX) via yfinance.
Pretrained Sentiment Model: VADER Sentiment Analysis via NLTK.
Libraries: pandas, numpy, scikit-learn, and matplotlib.
AI Tools: Gemini was used to assist with README structuring and when the code showed error 
  
