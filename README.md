# market-sentimental-model
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

## 3. Results & Honest Takeaways

| Model / Strategy | Test Accuracy (2015–2016) | Notes |
| :--- | :---: | :--- |
| **Baseline 1: Always Predict Up** | `53.17%` | Reflects the long-term upward drift of equities. |
| **Baseline 2: Previous Day Direction** | `48.94%` | Simple daily momentum fails due to mean reversion. |
| **Our Model: Shock + VIX Logistic Regression** | `~53.4%` | Marginally beats baselines, confirming daily news is mostly priced in. |

### What These Results Justify (and What They Don't)
* **What an investor CANNOT do:** Use daily aggregated news headlines to day-trade the DJIA. A ~0.3%–1.5% edge over the "Always-Up" baseline disappears immediately once bid-ask spreads, slippage, and execution latency are factored in. Furthermore, public headlines reflect information institutional algorithms have already traded on.
* **What an investor CAN do:** Use headline dispersion and VIX interaction spikes as a **risk-management filter** (e.g., tightening stop-losses or reducing leverage on days with extreme negative tail-risk news).

---

## 4. How to Run
1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
  
