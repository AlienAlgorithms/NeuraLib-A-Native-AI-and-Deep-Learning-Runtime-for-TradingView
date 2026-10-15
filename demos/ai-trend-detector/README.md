# AI Trend Detector

A complete TradingView indicator built on NeuraLib and NeuraLib_Models: a small neural network trained on the chart that rates Bear, Neutral and Bull pressure. Back to the [NeuraLib README](../../README.md).

[![AI Trend Detector on TradingView](https://s3.tradingview.com/d/dfkynAUp.png?v=1780969232)](https://www.tradingview.com/script/dfkynAUp-AI-Trend-Detector-Adaptive-Signals-NeuraLib-Machine-Learning/)

- **TradingView publication:** [AI Trend Detector | Adaptive Signals [NeuraLib Machine Learning]](https://www.tradingview.com/script/dfkynAUp-AI-Trend-Detector-Adaptive-Signals-NeuraLib-Machine-Learning/). Add it to a chart from there.
- **Source:** [ai-trend-detector.pine](ai-trend-detector.pine), version 1, as published on TradingView.

---

AI Trend Detector is a NeuraLib-powered Machine Learning indicator. It trains a compact supervised neural model on confirmed historical movement, then uses the current market state to estimate **Bear**, **Neutral**, and **Bull** pressure.

The model output is converted into a clean visual system:

- **Trend Oscillator**: A 0-100 pressure gauge. Lower values suggest bullish pressure or oversold conditions. Higher values suggest bearish pressure or overbought conditions.
- **Adaptive MA Cloud**: A main-chart adaptive moving average with an AI-biased cloud that expands as model pressure moves away from neutral.
- **Confirmed Triangles**: Optional chart markers for overbought and oversold interactions, with modes for zone entry, zone exit, or confirmed rotation inside a zone.
- **Dashboard**: A compact readout showing the current state, signal value and confidence.
- **Triangle Alerts**: Alert conditions tied to the same confirmed marker logic shown on the chart.
- **Directional Confidence**: An optional 0-100 line showing the stronger directional model probability, calculated from the larger of Bull or Bear pressure. It does not include Neutral probability, so it reflects directional conviction rather than overall model certainty.

This is not a fixed crossover system. The signals are the visual layer of a model-driven trend pressure engine.

---

## How The Model Learns

Each bar contributes a compact feature row based on price movement, adaptive MA context, and distance from the adaptive baseline. NeuraLib stores these rows in a rolling dataset, normalizes the inputs, and trains the model on recent time-series windows.

The model is trained as a 3-class classifier:

- **Bear**
- **Neutral**
- **Bull**

Historical training examples use future-resolved movement to create their target class, but only after that movement has already occurred. This is the supervised learning setup: the model learns from completed historical outcomes, then applies its learned weights to the current live feature window.

The exposed settings allow users to experiment with model size, learning rate, training frequency, smoothing, trend horizon, and signal behavior.

---

## Model Architecture

The model uses a compact temporal classification architecture:

1. **Flattened state window**: Recent feature rows are combined into one temporal input.
2. **Temporal convolution stack**: Conv1D-style layers extract short-term structure from the recent market sequence.
3. **Global average pooling**: The temporal output is compressed into a compact state representation.
4. **Dense classifier head**: One or two dense layers process the pooled state.
5. **Three output logits**: The model produces Bear, Neutral, and Bull logits, which are converted into display probabilities.

This keeps the model small enough for Pine Script while still giving it a true sequence-learning structure rather than a simple crossover or rule-based signal engine.

---

## Reading The Signals

The oscillator is intentionally inverted for intuitive market reading:

- **Low values**: Oversold or bullish pressure.
- **Mid values**: Balanced or neutral pressure.
- **High values**: Overbought or bearish pressure.

Triangles can be configured through the **Triangle trigger** setting:

- **Crossing into**: Prints when the oscillator crosses into an overbought or oversold zone.
- **Going out of**: Prints when the oscillator exits an overbought or oversold zone.
- **Rotation inside zone**: Prints when the signal forms a confirmed turn while still inside the zone.

In rotation mode, **Rotation confirmation** controls how many bars must pass without breaking the candidate peak or trough before the marker is accepted. Rotation triangles print on the confirmation bar, not on the older pivot bar.

The adaptive MA cloud is visual only. The model is not trained on the shifted cloud edge. The cloud simply applies model pressure around the adaptive MA baseline.

---

## Repainting And Signal Timing

The training and signal system is designed around confirmed bars:

- Training rows are pushed on confirmed bars.
- Triangle signals are gated with `barstate.isconfirmed`.
- Rotation markers print on the confirmation bar.
- No negative plot offsets are used to move markers into the past.
- The smoothing path uses current and past values only.

Because this model does not train on the full price history, but instead learns from the most recent N bars, repainting may occur when the script is reloaded at a later date. This happens because the model may begin training from a different market environment.

To help preserve the original model state, adjust the **Historical Train Window** setting to account for any new bars that have been added since the original run.

---

## Limitations

Machine Learning inside Pine Script is powerful, but it is still bounded by TradingView's execution model.

- The model is compact by design.
- Training history is bounded for performance.
- Changing hyperparameters rebuilds the model.
- Signals depend on the chosen horizon, threshold, smoothing, and triangle mode.
- The model estimates directional pressure. It does not know your entries, exits, risk, fees, or position sizing.

This indicator is best treated as a model-based market pressure tool, not as a complete trading system by itself.

This indicator is powered by the [NeuraLib Deep Learning Runtime](https://www.tradingview.com/script/GewgOj30-NeuraLib-A-Native-AI-and-Deep-Learning-Runtime/)

*Disclaimer: This indicator is an analytical and educational tool. It does not guarantee future results, signal accuracy, or financial gain. Past behavior does not ensure future behavior. Use it as one component in a broader trading process, under your own responsibility. Conceptual architecture and quantitative development by Alien_Algorithms.*

---

## Licence

The AI Trend Detector is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). The full legal text is in [LICENSE](../../LICENSE).

- **Non-commercial use is free** when [Alien_Algorithms](https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts) is credited in the description of any script that uses it.
- **Share alike:** a published adaptation must carry the same licence.
- **Commercial use** needs a separate licence. Contact [Alien_Algorithms on TradingView](https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts).
