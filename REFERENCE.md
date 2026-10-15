# NeuraLib

> NeuraLib is an open-source machine learning runtime for TradingView, published as a Pine Script v6 library by Alien_Algorithms. It has tensors, automatic differentiation, layers, losses, optimizers, datasets with scaling, training, validation and inference, so a neural network can be built, trained and run inside an indicator or strategy. The companion library NeuraLib_Models adds convolution, recurrent, attention and Transformer blocks.

- Product page: https://alienalgorithms.com/neurallib
- Documentation: https://docs.alienalgorithms.com/Neuralib-AI-Framework-Tradingview
- Library and source code on TradingView: https://www.tradingview.com/script/GewgOj30-NeuraLib-A-Native-AI-and-Deep-Learning-Runtime/
- Import: `import Alien_Algorithms/NeuraLib/1 as nl`, and optionally `import Alien_Algorithms/NeuraLib_Models/1 as models` (use the latest version numbers shown on the TradingView publications)
- Language: Pine Script v6
- Author: Alien_Algorithms, https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts
- Licence: CC BY-NC-SA 4.0. Free for non-commercial use when Alien_Algorithms is credited. Commercial use needs a licence from Alien_Algorithms.

This file is the plain-text reference for the product page. It is written so that a person or an AI assistant can write correct NeuraLib code from it. Names and signatures are taken from the library source, NeuraLib 1.0 and NeuraLib_Models 1.0.

NeuraLib brings the mechanics behind modern neural networks, backpropagation, optimizers and data pipelines, onto a TradingView chart. Instead of a fixed formula or a static regression, you define an architecture, feed it market features and let the model learn. Its core maths has been parity-tested against established machine learning runtimes (Keras, TensorFlow and PyTorch): forward passes, gradients, losses and optimizer steps follow the same logic.

## Contents

1. Getting started
2. Core concepts in plain terms
3. Building a model
4. Configuring training
5. Feeding data: datasets and scaling
6. Training, evaluating and predicting
7. The two execution modes
8. Built-in data guardrails
9. Reference: activations, losses, metrics, optimizers
10. Advanced: custom graph blocks
11. The NeuraLib_Models expansion
12. Worked example: the AI Trend Detector
13. Best practices and limitations
14. API quick reference
15. Licence

## Getting started

Import NeuraLib at the top of any Pine Script v6 indicator or strategy:

```pine
import Alien_Algorithms/NeuraLib/1 as nl
```

For the advanced architecture blocks (LSTMs, Transformers, Q-heads; see The NeuraLib_Models expansion), import the expansion after the main library:

```pine
import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models
```

### Your first model

This complete indicator builds a small neural network with one hidden layer, dropout, Huber loss and the AdamW optimizer, then runs a prediction on four common market features:

```pine
//@version=6
indicator("NeuraLib Basic Model", overlay = false, calc_bars_count = 600)

import Alien_Algorithms/NeuraLib/1 as nl

var nl.Sequential model = nl.sequential("basic_model")
var float modelOutput = na

if barstate.isfirst
    nl.CompileConfig cfg = nl.compileConfig()
    cfg := cfg
      .optimizer(nl.adamW(0.001))
      .loss(nl.LossKind.huber)
      .metric(nl.MetricKind.mae)
      .withTrainingGate(true)

    model := model
      .input(array.from(4), "features")
      .dropout(0.15)
      .dense(8, nl.ActivationKind.relu, "hidden")
      .dense(1, nl.ActivationKind.linear, "output")
      .compile(cfg)

float rsiValue = ta.rsi(close, 14)
float emaValue = ta.ema(close, 21)
float atrValue = ta.atr(14)
float atrPct = close == 0.0 ? 0.0 : atrValue / close
float momentum = na(close[1]) or close[1] == 0.0 ? 0.0 : close / close[1] - 1.0

bool ready = not na(rsiValue) and not na(emaValue) and not na(atrPct) and not na(momentum)
if ready
    float priceVsEma = emaValue == 0.0 ? 0.0 : close / emaValue - 1.0
    nl.Tensor inputTensor = nl.vector(array.from(rsiValue, priceVsEma, atrPct, momentum), "features")
    nl.Tensor outputTensor = model.predict(inputTensor)
    modelOutput := outputTensor.get1d(0)

plot(modelOutput, "Untrained model output", color = color.aqua, linewidth = 2)
hline(0.0, "Zero", color = color.new(color.gray, 70))
```

This plots the untrained output: the model has not learned anything yet. The sections below show how to feed it data and train it.

## Core concepts in plain terms

If you are new to machine learning, these five ideas are enough to follow the rest of this file:

| Term | What it means |
|---|---|
| Feature | A number you give the model as input: an RSI value, a return, a distance from a moving average. A feature row is one set of these numbers for one moment in time. |
| Target | The answer you want the model to learn, for example the next bar's return, or a label such as Bull, Neutral or Bear. |
| Loss | A score of how wrong the model's output was compared with the target. Lower is better. |
| Training | The model adjusts its internal weights to reduce the loss next time. NeuraLib does this with real backpropagation. |
| Prediction (inference) | Running new features through the trained model to get its current output. |

The workflow is always the same loop: build a feature row, build a target row, push them into a dataset, train on a batch, predict on live data.

## Building a model

Models are defined with the chainable `Sequential` API. Each call adds one layer, so the chain reads top to bottom like a description of the network:

```pine
var nl.Sequential model = nl.sequential("my_model")

if barstate.isfirst
    model := model
      .input(array.from(4), "features")            // 4 input features
      .dense(8, nl.ActivationKind.relu, "hidden")  // hidden layer, 8 neurons
      .dropout(0.10)                               // regularization
      .dense(1, nl.ActivationKind.linear, "out")   // 1 output value
      .compile(cfg)
```

### Available layers

| Method | What it does |
|---|---|
| `.input(dimsArray, name)` | Defines the input shape. Always the first call. |
| `.dense(units, activation, name)` | Fully connected layer, the workhorse of most models. |
| `.dropout(rate, name)` | Randomly disables part of the layer during training to reduce overfitting. |
| `.layerNorm(name)` | Layer normalization. Stabilizes training in deeper models. |
| `.activation(kind, alpha, name)` | A standalone activation block. |
| `.qHead(actionCount, activation, name)` | Q-value output head for reinforcement-learning outputs. |
| `.flatten(name)` / `.reshape(dims, name)` | Adjust shape metadata between layers. |
| `.block(graphBlock)` | Inserts a custom GraphBlock (see Advanced: custom graph blocks). |

### Building and compiling

- `.compile(cfg)` builds the model if needed and applies a training configuration. Use it when you plan to train.
- `.build(nl.rng(seed))` builds the weights with a fixed random seed and no training configuration. Use it for inference-only demos, or for reproducible starting weights.

## Configuring training

Training is set through a `CompileConfig`, which also chains:

```pine
nl.CompileConfig cfg = nl.compileConfig()
cfg := cfg
  .optimizer(nl.adamW(0.003))          // how weights get updated
  .loss(nl.LossKind.huber)             // what "wrong" means
  .metric(nl.MetricKind.mae)           // extra tracked metric
  .clipNorm(1.0)                       // gradient clipping for stability
  .withTrainingGate(true)              // train only at the live edge (see The two execution modes)
```

### CompileConfig methods

| Method | Purpose |
|---|---|
| `.optimizer(state)` | Sets the optimizer (SGD, Momentum, RMSProp, Adam, AdamW). |
| `.schedule(state)` | Sets a learning-rate schedule (constant, step decay, cosine, warmup cosine). |
| `.loss(lossKind)` | Sets the training loss. |
| `.reduction(kind)` | Sets how the loss is reduced across a batch. |
| `.metric(metricKind)` | Adds a tracked metric. |
| `.batchSize(n)` / `.epochsPerBar(n)` / `.evalStride(n)` / `.historyLength(n)` | Batch and cadence preferences, and metric history length. |
| `.clipNorm(v)` / `.clipValue(v)` | Gradient clipping, which prevents unstable weight updates. |
| `.gradAccumSteps(n)` | Accumulates gradients across several calls before stepping. |
| `.withTrainingGate(true)` | Restricts training to the last confirmed historical bar and onward. |
| `.trainEveryCall()` | Allows training on every `.trainOnBatch()` call, on all bars. |

### Presets

Presets set a sensible loss and metric in one call:

| Preset | Use for |
|---|---|
| `.presetPriceRegression()` | Predicting a price level. |
| `.presetReturnRegression()` | Predicting a return. |
| `.presetBinaryDirection()` / `.presetBinaryDirectionLogits()` | Up or down classification. |
| `.presetQValues()` | Action-value (Q-learning) outputs. |
| `.presetSharpe()` | Sharpe-style risk-adjusted training. |

## Feeding data: datasets and scaling

Market data comes on very different scales: a price might be 60,000 while a return is 0.002. Models train badly on raw mixed scales, so NeuraLib datasets scale their data.

### The two dataset types

| Dataset | Shape of one example | Use when |
|---|---|---|
| `nl.windowDataset(featureCount, targetCount, maxRows, name)` | One flat feature row | Each bar's snapshot is enough: RSI, momentum, distance from a moving average. |
| `nl.rollingDataset(timeSteps, featureCount, targetCount, maxRows, name)` | A window of the last N rows | The sequence of recent bars matters, for Conv1D, LSTM or attention models. |

### Typical setup

```pine
var nl.WindowDataset dataset = nl.windowDataset(4, 1, 500, "returns_dataset")

if barstate.isfirst
    dataset := dataset
      .setInputScaler(nl.ScalerKind.zScore)
      .setTargetScaler(nl.ScalerKind.zScore)
```

Scalers (`nl.ScalerKind`): `none`, `zScore`, `minMax` and `runningZScore`.

### Pushing rows

```pine
dataset := dataset.pushRow(features, target)
```

Rows are checked on insert: their widths must match the configured feature and target counts, and rows that contain `na` are rejected.

Helpers for common targets:

- `pushNextReturnRow(featureBuilder, currentValue, futureValue)`: the target is the return between two values.
- `pushNextDirectionRow(featureBuilder, currentValue, futureValue, threshold, zeroOne)`: the target is a direction label.
- `nl.nextReturnValue(a, b)` and `nl.nextDirectionValue(a, b, threshold, zeroOne)` build these targets by hand.

### Getting batches out

```pine
if dataset.ready(64)                              // at least 64 usable rows
    nl.Batch train = dataset.trainBatch(16)       // hold out the last 16 rows for validation
    nl.Batch validation = dataset.validationBatch(16)
```

A `Batch` carries an `inputTensor` and a `targetTensor`, already scaled and ready for the model. Other ways to get batches: `lastBatch(batchSize)`, `toBatch()` (all rows) and `unrollBatch(targetOffset)` for rolling windows.

### FeatureBuilder

For readable feature assembly, `nl.featureBuilder(name)` lets you `.push(value, "feature_name")` one value at a time, then convert with `.toTensor()` or `.toArray()`.

## Training, evaluating and predicting

### Train

```pine
model := model.trainOnBatch(train.inputTensor, train.targetTensor)
float lastLoss = model.trainStats.lastLoss
```

`trainOnBatch()` runs a forward pass, computes the loss, backpropagates the gradients and applies an optimizer step, within the active training gate. `model.trainStats` tracks recent losses and metrics.

For manual control the step can be split: `.backward(target)` accumulates gradients without updating, and `.step()` applies the optimizer step.

### Evaluate without learning

```pine
nl.LossResult validationLoss = model.evaluate(validation.inputTensor, validation.targetTensor)
```

`evaluate()` measures the loss on data the model did not train on. Watching the validation loss, not only the training loss, is the standard way to catch overfitting. Pair it with `nl.earlyStopper(patience, minDelta)` to stop training when the validation loss stops improving.

### Predict

```pine
nl.Tensor scaledInput = dataset.scaleInput(liveInput)       // scale live features like the training data
nl.Tensor scaledPrediction = model.predict(scaledInput)
nl.Tensor rawPrediction = dataset.inverseScaleTarget(scaledPrediction)  // back to real units
float value = rawPrediction.get1d(0)
```

Two habits shown here:

1. Scale live inputs with the dataset's scaler (`dataset.scaleInput()`), so inference sees the same normalization as training.
2. Inverse-scale the output (`dataset.inverseScaleTarget()`) if you scaled the targets, to get answers back in real units.

### Uncertainty estimates

`.predictMC(inputTensor, samples)` runs several predictions with dropout active and returns their mean and variance: a practical way to see how confident the model is, not only what it predicts. `samples` defaults to 10.

### Saving and sharing weights

- `.getWeightsArray()` / `.setWeightsArray(array)` export and import flat weights.
- `.softUpdateFrom(sourceModel, tau)` blends weights from another model, as used for target networks in reinforcement learning.

## The two execution modes

Pine Script runs your script on every bar, so when training happens matters a lot for speed.

### 1. Live-edge training: .withTrainingGate(true)

The dataset collects rows across the whole chart history, but the expensive training step runs only on the last confirmed historical bar and then in real time. This avoids TradingView timeouts.

Use it for: larger models, more features, rolling sequence inputs, attention and recurrent architectures.

### 2. Full-history training: .trainEveryCall()

The model trains and predicts on every historical bar, so its behaviour evolves across the whole chart, for example an AI moving average that keeps adapting to local structure.

Use it for: tiny dense models, small batches, fast adaptive filters, light feature transforms. Start with 4 to 8 hidden units and batch sizes of 8 to 16.

Tip: also set `calc_bars_count` in `indicator()` (for example `calc_bars_count = 600`) to balance training history against execution time.

## Built-in data guardrails

NeuraLib prevents the most common machine learning mistakes by default:

- Invalid rows are rejected: rows of the wrong width and rows containing `na` never enter a dataset.
- Shape checks protect model calls: forward, training, backward and evaluation paths check input and target shapes before running expensive graph code.
- Training and validation stay separate: `trainBatch()` and `validationBatch()` use a holdout split instead of mixing everything together.
- Scaler leakage is controlled: validation batches are scaled with the training-side profile, so normalization never sees the holdout data.
- Rolling windows keep time order: `RollingDataset` reads chronologically even with wrapped ring buffers and target offsets.

Together these guard against data leakage, validation contamination, lookahead bias in scaling and silent shape bugs.

## Reference: activations, losses, metrics, optimizers

### Activations (nl.ActivationKind)

Linear, ReLU, Leaky ReLU, ELU, GELU (approximate), Sigmoid, Tanh, Softplus, Swish, Softmax.

### Losses (nl.LossKind)

| Loss | Typical use |
|---|---|
| MSE, MAE, Huber, LogCosh | General regression. Huber and LogCosh are robust to outliers. |
| Binary cross entropy (and from logits) | Two-class problems (up or down). |
| Categorical cross entropy, softmax cross entropy from logits | Multi-class problems (Bear, Neutral, Bull). |
| Directional | Penalizes getting the direction wrong. |
| Quantile | Predicts a chosen quantile instead of the mean. |
| Multi-horizon weighted | Weighted targets across several forecast horizons. |
| Sharpe | Optimizes a Sharpe-style risk-adjusted objective. |

### Metrics (nl.MetricKind)

MAE, RMSE, directional accuracy, binary accuracy (and from logits), categorical accuracy, cosine similarity.

### Optimizers and schedules

| Optimizer | Notes |
|---|---|
| `nl.sgd(learningRate)` | Plain gradient descent. |
| `nl.momentum(learningRate, momentum = 0.9)` | SGD with momentum. |
| `nl.rmsprop(learningRate, rho = 0.9, epsilon = 1e-7)` | Adaptive per-weight rates. |
| `nl.adam(learningRate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7)` | The most common modern default. |
| `nl.adamW(learningRate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7, weightDecay = 0.01)` | Adam with decoupled weight decay. The recommended default in NeuraLib. |

Learning-rate schedules: `constantSchedule()`, `stepDecay()`, `cosineDecay()`, `warmupCosine()`.

## Advanced: custom graph blocks

The `Sequential` API is the easy path. Under it sits a full automatic-differentiation graph engine you can use directly: create a `GraphBlock`, wire operations together, set an output, and add it to a model with `.block()`. The runtime handles the backward pass and the weight updates, so custom architectures train with the same `.trainOnBatch()` workflow as the built-in layers.

```pine
nl.GraphBlock myBlock = nl.graphBlock("custom")
// ... define inputs, params, operations and the output node ...
model := model.input(array.from(8)).block(myBlock).dense(1, nl.ActivationKind.linear).compile(cfg)
```

Graph operations include:

- Maths: matmul, transpose, add, subtract, multiply, divide, scale.
- Neural: activations, softmax, layer normalization, dropout.
- Structure: causal masking, slicing, concatenation, row and column reduction.
- Sequence: global average and max pooling, attention score and apply, Conv1D, LSTM scan, GRU scan.

This graph layer is what lets companion libraries such as NeuraLib_Models add whole families of architectures without changing the core runtime.

## The NeuraLib_Models expansion

NeuraLib_Models is the official companion library. It packages architectures that would be tedious to build from raw graph operations into single calls. Once imported, its methods are available directly on NeuraLib `Sequential` models, so the `models` alias never appears in the layer chain:

```pine
import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models   // import order matters: NeuraLib first

var nl.Sequential model = nl.sequential("advanced_model")

if barstate.isfirst
    model := model
      .input(array.from(8), "sequence")
      .temporalConvStack(4, 2, 3, 2, 2, 1, nl.ActivationKind.relu, 0.0, "temporal")
      .globalAvgPool1d(2, 3, "pool")
      .duelingQHead(4, 2, nl.ActivationKind.relu, "q_head")
      .build(nl.rng(7))
```

The result is still a normal NeuraLib model: `.compile()`, `.trainOnBatch()`, `.predict()`, `.evaluate()` and every dataset tool work unchanged.

### Model families

| Family | Methods | What they are for |
|---|---|---|
| Temporal convolution | `conv1d()`, `temporalConvStack()` | Local patterns in short rolling windows. A sequence of 4 time steps x 2 features is passed as 8 flattened inputs. |
| Pooling | `globalAvgPool1d()`, `globalMaxPool1d()` | Compressing a sequence into a summary. Average captures broad behaviour; max emphasizes the strongest signal. |
| Recurrent | `lstm()`, `gru()` | Sequence memory, when the order of recent states matters more than a snapshot. |
| Attention and Transformers | `selfAttention()`, `multiHeadSelfAttention()`, `crossAttention()`, `transformerEncoder()`, `transformerEncoderStack()`, `transformerDecoder()` | Token-style modelling where each row is a time step and each column a feature. `modelDim` must be divisible by `headCount`, with at most 8 heads. |
| Residual | `residualDense()` | Deeper feedforward stacks that keep information across layers. |
| Reinforcement learning | `qHead()`, `duelingQHead()` | Action-value outputs. Dueling heads split state value and action advantage, then recombine them. |
| Replay utilities | `prioritizedReplayBuffer()` with `pushExperience()`, `sampleBatch()`, `updatePriority()` | Prioritized experience replay: revisit high-error examples more often. Sampling is deterministic per seed, so results are reproducible. |
| Sequence helpers | `pushPositionalEncoding()` | Adds sinusoidal position features, so attention models know where in the sequence a step sits. |

### Shape rules

- Conv1D, temporal stacks, LSTM and GRU expect flattened sequences of size `timeSteps x featureCount`.
- Attention methods expect each input row to have `modelDim` columns; cross-attention and decoders use packed row layouts.
- The builders check feature counts and head compatibility and stop with an error instead of misbehaving silently.

## Worked example: the AI Trend Detector

The published indicator "AI Trend Detector | Adaptive Signals [NeuraLib Machine Learning]" is a complete NeuraLib application and a good template for your own scripts. This is how it maps onto everything above.

The goal: classify current market pressure as Bear, Neutral or Bull with a small neural network trained live on the chart.

### 1. Features: three per bar

Each bar contributes a percent return, an adaptive (Kaufman-style) moving average value, and the percent distance of price from that average. Thirty consecutive bars are stacked into one state window, so the model input is 30 x 3 = 90 values.

### 2. Labels: created only after the future resolves

The hard part of machine learning on charts is avoiding lookahead. The Trend Detector labels a bar only after a fixed horizon of 8 bars has passed: it measures where price actually went, normalizes that move by local volatility, and assigns Bear `[1,0,0]`, Neutral `[0,1,0]` or Bull `[0,0,1]`. The feature row belongs to the past anchor bar; the label comes from the move that has since completed. No future data leaks into a feature.

### 3. Dataset: a RollingDataset with z-score scaling

```pine
var nl.RollingDataset dataset = nl.rollingDataset(sequenceLength, featureCount, CLASS_COUNT, datasetRows, "trend_detector_rows")
dataset := dataset.setInputScaler(nl.ScalerKind.zScore).setTargetScaler(nl.ScalerKind.none)
```

One resolved example is pushed per confirmed bar. Inputs are z-scored column by column, so the model behaves the same on a $2 altcoin and a $60,000 index; the one-hot class targets are left unscaled.

### 4. Model: temporal convolution into a softmax classifier

```pine
model := nl.sequential("trend_detector_nn").input(array.from(modelInputCount), "state_window")
model := model.temporalConvStack(sequenceLength, featureCount, temporalFilters, temporalKernel, temporalLayers, 1, nl.ActivationKind.relu, dropoutRate, "temporal")
model := model.globalAvgPool1d(temporalStepsOut, temporalFilters, "temporal_avg")
model := model.dense(hiddenUnitsInput, nl.ActivationKind.relu, "detector_1")
model := model.dense(CLASS_COUNT, nl.ActivationKind.linear, "trend_logits").build(nl.rng(seedInput)).compile(cfg)
```

The temporal convolution stack (from NeuraLib_Models) extracts short-term patterns from the state window, pooling compresses the sequence, and dense layers produce three logits. The configuration uses softmax cross entropy from logits as the loss, categorical accuracy as the metric, AdamW, and `clipNorm(1.0)` for stability.

### 5. Training: controlled cadence

Training uses `trainEveryCall()` but is gated by hand: it runs only on confirmed bars, only every N bars (a stride input), and only within the most recent N historical bars, while the dataset keeps collecting rows everywhere. This pattern suits history-wide data collection with budget-limited training.

### 6. Inference: same layout, same scaler

Every bar, the current 30-bar state is built in the same feature order as in training, scaled through `dataset.scaleInput()`, and passed to `model.predict()`. The three logits become Bear, Neutral and Bull probabilities through a temperature-controlled softmax, and are drawn as an oscillator, a confidence line, an AI-biased moving average cloud and overbought and oversold triangles.

The takeaway: NeuraLib does not have to be the indicator. Here the model drives the decisions, while the visible chart layer is ordinary Pine plotting.

## Best practices and limitations

- Start small. Pine Script is not a GPU environment. A shallow model with 4 to 8 hidden units and a batch size of 8 to 16 is the right start; grow it only if validation results justify it.
- Control chart history. Use `calc_bars_count` in `indicator()` to balance training history against execution time.
- Use the training gate for heavy models. `.withTrainingGate(true)` keeps backpropagation at the confirmed live edge and avoids timeouts.
- Always scale inputs. Raw market features differ by orders of magnitude; use the dataset scalers unless you have a deliberate reason not to.
- Validate separately. Watch the `validationBatch()` loss, not only the training loss, and consider `earlyStopper()`.
- Avoid lookahead. Build feature rows only from information available at that moment, and create targets only after the future has resolved (the Trend Detector pattern above).
- Treat outputs as research signals. NeuraLib provides the model mechanics. Strategy design, risk management and market assumptions remain your responsibility. A model that fits history does not guarantee future performance.

## API quick reference

### Model setup
`sequential(name)`, `compileConfig()`, `build(rng)`, `compile(config)`, `rng(seed, streamId)`

### Sequential methods
`input()`, `dense()`, `qHead()`, `activation()`, `dropout()`, `layerNorm()`, `flatten()`, `reshape()`, `block()`, `trainOnBatch()`, `backward()`, `step()`, `predict()`, `predictMC()`, `evaluate()`, `fitDataset()`, `fitRollingDataset()`, `getWeightsArray()`, `setWeightsArray()`, `softUpdateFrom()`

### CompileConfig
`optimizer()`, `schedule()`, `loss()`, `reduction()`, `metric()`, `batchSize()`, `epochsPerBar()`, `evalStride()`, `historyLength()`, `clipNorm()`, `clipValue()`, `gradAccumSteps()`, `withTrainingGate()`, `trainEveryCall()`, and the presets `presetPriceRegression()`, `presetReturnRegression()`, `presetBinaryDirection()`, `presetBinaryDirectionLogits()`, `presetQValues()`, `presetSharpe()`

### Datasets
`windowDataset()`, `rollingDataset()`, `pushRow()`, `pushBuilderRow()`, `pushNextReturnRow()`, `pushNextDirectionRow()`, `ready()`, `size()`, `lastBatch()`, `toBatch()`, `unrollBatch()`, `trainBatch()`, `validationBatch()`, `setInputScaler()`, `setTargetScaler()`, `scaleInput()`, `scaleTarget()`, `inverseScaleTarget()`, `clear()`

### Tensors and features
- Constructors: `scalar()`, `vector()`, `matrix2d()`, `zeros()`, `ones()`, `full()`, `shapeFromDims()`
- Tensor methods: `reshape()`, `flatten()`, `row()`, `get1d()`, `sum()`, `mean()`, `variance()`, `normL2()`, `argmax()`, `dot()`
- Matrix methods: `matmul()`, `transpose()`, `add()`, `subtract()`, `multiply()`, `divide()`, `scale()`, `activate()`, `softmax()`, `sliceRows()`, `sliceCols()`, `concatRows()`, `concatCols()`, `globalAvgPool1d()`, `globalMaxPool1d()`
- `featureBuilder()` with `push()`, `addFeature()`, `toTensor()`, `toArray()`, `size()`, `clear()`

### Scalers, optimizers, schedules
`zScoreScaler()`, `minMaxScaler()`, `runningZScoreScaler()`, `noneScaler()` (with `fit()`, `partialFit()`, `transform()`, `inverseTransform()`), `sgd()`, `momentum()`, `rmsprop()`, `adam()`, `adamW()`, `constantSchedule()`, `stepDecay()`, `cosineDecay()`, `warmupCosine()`, `currentRate()`

### Losses, metrics, helpers
`mse()`, `mae()`, `huber()`, `logCosh()`, `binaryCrossEntropy()`, `binaryCrossEntropyFromLogits()`, `categoricalCrossEntropy()`, `softmaxCrossEntropyFromLogits()`, `directionalLoss()`, `quantileLoss()`, `multiHorizonWeighted()`, `sharpeLoss()`, `metricValue()`, `earlyStopper()`, `nextReturnValue()`, `nextDirectionValue()`

### GraphBlock
- `graphBlock()`
- Structure: `input()`, `param()`, `constScalar()`, `constMatrix()`, `output()`
- Maths: `matmul()`, `add()`, `subtract()`, `multiply()`, `divide()`, `scale()`, `activate()`, `softmax()`, `transpose()`, `layerNorm()`, `dropout()`
- Structure operations: `causalMask()`, `sliceRows()`, `concatRows()`, `sliceCols()`, `concatCols()`, `reduceRows()`, `reduceCols()`
- Sequence operations: `globalAvgPool1d()`, `globalMaxPool1d()`, `attentionScore()`, `attentionApply()`, `conv1d()`, `scanLstm()`, `scanGru()`

### NeuraLib_Models
- Sequential methods: `residualDense()`, `duelingQHead()`, `conv1d()`, `temporalConvStack()`, `globalAvgPool1d()`, `globalMaxPool1d()`, `lstm()`, `gru()`, `selfAttention()`, `multiHeadSelfAttention()`, `crossAttention()`, `transformerEncoder()`, `transformerEncoderStack()`, `transformerDecoder()`
- Block factories: `qHeadBlock()`, `duelingQHeadBlock()`, `residualDenseBlock()`, `conv1dBlock()`, `temporalConvStackBlock()`, and pooling, recurrent, attention and Transformer block variants
- Replay: `prioritizedReplayBuffer()`, `pushExperience()`, `sampleBatch()`, `updatePriority()`, `toBatch()`, `ready()`, `size()`, `clear()`
- Helpers: `pushPositionalEncoding()`

## Licence

NeuraLib and NeuraLib_Models are licensed under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/). Use is free for non-commercial purposes when Alien_Algorithms (https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts) is credited in the description of any script that uses them. For commercial licensing, contact Alien_Algorithms on TradingView.
