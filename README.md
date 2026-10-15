# NeuraLib

A machine learning runtime for TradingView, written in Pine Script v6: tensors, automatic differentiation, layers, losses, optimizers, datasets with scaling, training, validation and inference, inside an indicator or strategy. This repository also holds **NeuraLib_Models**, the expansion with convolution, recurrent, attention and Transformer blocks.

[![NeuraLib on TradingView](https://s3.tradingview.com/g/GewgOj30.png?v=1777899672)](https://www.tradingview.com/script/GewgOj30-NeuraLib-A-Native-AI-and-Deep-Learning-Runtime/)

- **TradingView publications:** [NeuraLib: A Native AI and Deep Learning Runtime](https://www.tradingview.com/script/GewgOj30-NeuraLib-A-Native-AI-and-Deep-Learning-Runtime/) and [NeuraLib Expansion: Advanced Model Layers](https://www.tradingview.com/script/drHsBXE4-NeuraLib-Expansion-Advanced-Model-Layers/)
- **Documentation:** [docs.alienalgorithms.com](https://docs.alienalgorithms.com/Neuralib-AI-Framework-Tradingview)
- **Product page:** [alienalgorithms.com/neurallib](https://alienalgorithms.com/neurallib)
- **Version:** NeuraLib 1, NeuraLib_Models 1. See [CHANGELOG.md](CHANGELOG.md).
- **Licence:** CC BY-NC-SA 4.0, free for non-commercial use with credit. See [Licence](#licence).

## Use it in a script

```pine
//@version=6
indicator("My NeuraLib Model", overlay = false, calc_bars_count = 600)
import Alien_Algorithms/NeuraLib/1 as nl
// Optional, for the advanced model layers. Import it after NeuraLib.
import Alien_Algorithms/NeuraLib_Models/1 as models
```

Pine Script loads libraries from TradingView, not from GitHub. The import line above uses the published library; the source in this repository is the same code, here to read, search and reference. When NeuraLib is updated on TradingView, the version number in the import line goes up.

## Coding with an AI assistant

Pine Script is rare in AI training data, so assistants tend to guess library calls. Give the assistant [REFERENCE.md](REFERENCE.md) before it writes NeuraLib code. It documents the calls with their parameters, the rules for correct scripts and complete examples. The same file is online at https://alienalgorithms.com/neurallib.md.

## In this repository

| File | What it is |
|---|---|
| [NeuraLib.pine](NeuraLib.pine) | The NeuraLib runtime, version 1, as published on TradingView |
| [NeuraLib_Models.pine](NeuraLib_Models.pine) | The NeuraLib_Models expansion, version 1, as published on TradingView |
| [MODELS.md](MODELS.md) | The NeuraLib_Models guide: model families, examples and API |
| [REFERENCE.md](REFERENCE.md) | Both libraries in one file, for people and AI assistants |
| [CHANGELOG.md](CHANGELOG.md) | What each version changed |
| [LICENSE](LICENSE) | CC BY-NC-SA 4.0 legal text |
| [examples/basic-model.pine](examples/basic-model.pine) | NeuraLib Basic Model |
| [examples/return-validation.pine](examples/return-validation.pine) | NeuraLib Return Validation Example |
| [examples/models-extension-demo.pine](examples/models-extension-demo.pine) | NeuraLib Models Extension Demo |
| [examples/models-quick-start.pine](examples/models-quick-start.pine) | NeuraLib Models Quick Start |
| [examples/models-temporal-q.pine](examples/models-temporal-q.pine) | NeuraLib Models Temporal Q Example |
| [examples/models-transformer-encoder.pine](examples/models-transformer-encoder.pine) | NeuraLib Models Transformer Encoder Example |
| [examples/models-prioritized-replay.pine](examples/models-prioritized-replay.pine) | NeuraLib Models PER Example |

---

**NeuraLib** is a tensor-based, auto-differentiating Machine Learning runtime built natively for Pine Script™.

It brings real Deep Learning mechanisms that power modern Artificial Intelligence systems into TradingView. Instead of relying on fixed formulas, static regressions, or rigid structures, NeuraLib gives Pine developers a different tool: a compact neural runtime that can learn from the features you feed it, using the architecture you define.

This means users are no longer limited to classical methods like Linear Regression, Logistic Regression, KNN, Naive Bayes, Kalman Filters, or Markov Chains. One can build adaptive architectures perfectly suited for custom indicators, strategies, regime detection, directional prediction, price transforms, and AI-assisted signal generation.

Using NeuraLib, one can build a model, collect market data, normalize it, run predictions, train through backpropagation, track validation behavior, and update weights directly inside TradingView.

Furthermore, it is not necessary to directly display trained variables. The process can be a part of a larger script functionality, where AI-powered decision making changes how an indicator behaves.

[![NeuraLib on a TradingView chart](https://s3.tradingview.com/snapshots/t/tJSDIp4a.png)](https://www.tradingview.com/x/tJSDIp4a/)

The goal is to make real neural network workflows usable in Pine Script without hiding the important controls, being scalable with evolving market dynamics, and abstracting away the complexity that comes with such software. The provided API is highly modular and intuitive, using chained object-oriented programming for easy readability and use. The backend is engineered with fault-tolerance in mind, providing users with sanity checks and preventing common pitfalls by default.

Think of NeuraLib as a comprehensive machine learning ecosystem, containing:

- **A Model Builder**: Define neural networks with readable chained calls like `.input()`, `.dense()`, and `.dropout()`.
- **An In-Pine Training Engine**: Models calculate losses, backpropagate gradients, update weights, and produce predictions directly on chart data.
- **Automated Data Pipelines**: Built-in datasets handle feature collection, robust scaling (Z-Score, Min-Max), validation holdout splits, and time-series rolling windows.
- **Finance-Native Loss Functions**: Beyond standard error metrics, the engine includes Directional, Quantile, Multi-Horizon Weighted, and Sharpe-style losses tailored for trading.
- **Practical Training Controls**: Layer Normalization, AdamW weight decay, gradient clipping, gradient accumulation, and early stopping are built in to prevent overfitting.
- **Advanced Optimizers**: Train networks using RMSProp, Adam, or AdamW, paired with learning rate schedules like Warmup Cosine and Step Decay.

For newer users, this means you can start with a simple dense model. For advanced users, the same runtime exposes graph operations, custom blocks, tensors, matrix operations, optimizers, schedules, losses, and extension hooks.

In plain terms, a model receives a row of numbers called features, compares its output against a target, measures the error with a loss function, and then adjusts its internal weights to reduce that error next time.

---

## What makes it different

### Parity-tested neural math

NeuraLib’s core operations have been tested against established Machine Learning Runtimes outside of TradingView (Such as Keras / TensorFlow / PyTorch).

The goal was not to imitate the appearance of Machine Learning, but to reproduce the math that is proven to work. Standard forward passes, gradients, losses, and optimizer behavior were checked for 1:1 algorithmic parity, with negligible differences coming from normal floating-point behavior.

That means the matrix math, backpropagation, and gradient updates running on your chart follow the same underlying logic expected from professional Machine Learning environments.

### Matrix-first computation

NeuraLib uses tensor and matrix abstractions as the foundation of the runtime. Under the hood, it supports the operations needed for neural computation, including matrix multiplication, broadcasting, activation functions, softmax, slicing, concatenation, reductions, normalization, attention scoring, convolution-style operations, and recurrent scan blocks.

### Auto-differentiating graph engine

NeuraLib makes the computational graph a first-class object.

You can use high-level Sequential models, or build custom GraphBlocks from lower-level operations. Once a custom block is connected to a model, the same runtime handles the backward pass. That means your custom architecture can be trained with the same `.trainOnBatch()` workflow as standard layers.

---

## Custom graphs

The Sequential API is the easiest way to start, but NeuraLib is not just a list of built-in layers.

You can create a `GraphBlock`, add operations, set an output node, and plug that block into a model. Once connected, the runtime handles the backward pass and parameter updates.

Useful graph operations include:

- Matrix multiplication, transpose, add, subtract, multiply, divide, and scale.
- Activation functions and softmax.
- Layer Normalization and Dropout.
- Causal masking, slicing, concatenation, row reduction, and column reduction.
- Global average pooling and global max pooling for 1D sequences.
- Attention score and attention apply operations.
- Conv1D, LSTM scan, and GRU scan primitives.

This is the foundation that allows companion model libraries to add advanced AI and Machine Learning architectures without changing the main NeuraLib runtime.

---

## Built-in data guardrails

NeuraLib is not only a training mechanism. It also includes guardrails for cleaner research:

- **Invalid rows are rejected**: Dataset rows must match the configured feature and target counts, and rows containing `na` values are not inserted.
- **Shape checks protect model calls**: Forward, training, backward, and evaluation paths validate input and target shapes before running expensive graph code.
- **Train and validation splits are separated**: `trainBatch()` and `validationBatch()` use holdout rows instead of blending all rows into one batch.
- **Scaler leakage is controlled**: Validation batches are scaled from the training-side profile where the dataset split requires it, so validation normalization does not learn from the holdout slice.
- **Rolling windows respect time order**: `RollingDataset` supports target offsets and wrapped ring buffers while preserving chronological reads.

These checks help reduce common data poisoning and data leakage mistakes: wrong row widths, missing values, validation contamination, target-offset leakage, and accidental overtraining across every historical bar.

---

## A first model

The basic API is intentionally readable. This creates a small model with dropout, one hidden layer, Huber loss, AdamW optimization, and MAE tracking.

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

The same model can then receive scaled batches from a dataset and train with `.trainOnBatch()`. The plot in this first example is the untrained forward output, included so the block can be pasted directly into an indicator.

---

## A practical data flow

Machine Learning models usually fail when the data pipeline is careless. Price, volume, volatility, and oscillators often live on very different scales. NeuraLib includes dataset and scaling helpers so the common workflow stays explicit:

1. Build a feature row.
2. Build a target row.
3. Push the row into a dataset.
4. Request a training batch.
5. Request a validation batch when needed.
6. Train, evaluate, predict, and inverse-scale targets when appropriate.

```pine
//@version=6
indicator("NeuraLib Return Validation Example", overlay = false, calc_bars_count = 600)

import Alien_Algorithms/NeuraLib/1 as nl

var nl.Sequential model = nl.sequential("returns_model")
var nl.WindowDataset dataset = nl.windowDataset(4, 1, 500, "returns_dataset")
var float predictedReturn = na
var float validationLossValue = na
var float trainingLossValue = na

if barstate.isfirst
    nl.CompileConfig cfg = nl.compileConfig()
    cfg := cfg
      .optimizer(nl.adamW(0.003))
      .loss(nl.LossKind.huber)
      .metric(nl.MetricKind.mae)
      .trainEveryCall()

    model := model
      .input(array.from(4), "features")
      .dense(8, nl.ActivationKind.relu, "hidden")
      .dropout(0.10, "dropout")
      .dense(1, nl.ActivationKind.linear, "next_return")
      .compile(cfg)

    dataset := dataset
      .setInputScaler(nl.ScalerKind.zScore)
      .setTargetScaler(nl.ScalerKind.zScore)

float rsiValue = ta.rsi(close, 14)
float emaValue = ta.ema(close, 21)
float atrValue = ta.atr(14)
float atrPct = close == 0.0 ? 0.0 : atrValue / close
float momentum = na(close[1]) or close[1] == 0.0 ? 0.0 : close / close[1] - 1.0
float realizedReturn = na(close[1]) ? na : nl.nextReturnValue(close[1], close)

bool rowReady = not na(rsiValue[1]) and not na(emaValue[1]) and not na(atrPct[1]) and not na(momentum[1]) and not na(close[1])

if rowReady
    float prevEma = emaValue[1]
    float priceVsEma = prevEma == 0.0 ? 0.0 : close[1] / prevEma - 1.0

    array<float> features = array.from(
      rsiValue[1],
      priceVsEma,
      atrPct[1],
      momentum[1])

    array<float> target = array.from(nl.nextReturnValue(close[1], close))

    dataset := dataset.pushRow(features, target)

if dataset.ready(64)
    nl.Batch train = dataset.trainBatch(16)
    nl.Batch validation = dataset.validationBatch(16)

    model := model.trainOnBatch(train.inputTensor, train.targetTensor)
    trainingLossValue := model.trainStats.lastLoss

    nl.LossResult validationLoss = model.evaluate(validation.inputTensor, validation.targetTensor)
    validationLossValue := validationLoss.value

    bool liveReady = not na(rsiValue) and not na(emaValue) and not na(atrPct) and not na(momentum)
    if liveReady
        float livePriceVsEma = emaValue == 0.0 ? 0.0 : close / emaValue - 1.0
        array<float> liveFeatures = array.from(rsiValue, livePriceVsEma, atrPct, momentum)
        nl.Tensor liveInput = nl.vector(liveFeatures, "live_features")
        nl.Tensor scaledInput = dataset.scaleInput(liveInput)
        nl.Tensor scaledPrediction = model.predict(scaledInput)
        nl.Tensor rawPrediction = dataset.inverseScaleTarget(scaledPrediction)
        predictedReturn := rawPrediction.get1d(0)

plot(realizedReturn, "Last realized return", color = color.gray)
plot(predictedReturn, "Predicted next return", color = color.aqua, linewidth = 2)
plot(validationLossValue, "Validation loss", color = color.orange)
plot(trainingLossValue, "Training loss", color = color.new(color.blue, 35))
hline(0.0, "Zero", color = color.new(color.gray, 70))
```

This example trains from completed historical pairs. The feature row comes from the previous bar, and the target is the return from that previous bar to the current bar. That keeps the example easy to inspect and avoids using future information in the feature row. When pasted into an indicator, it plots the last realized return, the model's predicted next return, training loss, and validation loss.

---

## Two practical execution modes

Deep Learning in Pine requires careful execution control. NeuraLib supports two main workflows.

### Live-edge training

Use this when you want safer execution for larger models.

The dataset can collect rows across the chart, while the expensive training step only runs on the last confirmed historical bar. This helps avoid timeouts while still allowing the model to learn from recent prepared data.

```pine
cfg := cfg.withTrainingGate(true)
```

Use this for:

- Larger models
- More features
- Rolling sequence inputs
- Heavier architectures
- Safer live-edge updates

### Full-history training and inference

Use this when the model is intentionally small.

The model can train and infer across historical bars, which makes it possible to create lightweight adaptive indicators, such as an AI Moving Average that learns from recent local structure instead of using a fixed smoothing formula.

```pine
cfg := cfg.trainEveryCall()
```

Use this for:

- Tiny dense models
- Small batches
- Fast adaptive filters
- AI-assisted moving averages
- Lightweight feature transforms

For full-history workflows, start small. A shallow model with 4 to 8 hidden units and a batch size of 8 or 16 is usually a better starting point than a deep architecture.

---

## Advanced model expansion

NeuraLib is designed to act as the foundation for larger model libraries and community-built extensions.

To demonstrate this, **NeuraLib Expansion: Advanced Model Layers** is built entirely on top of the public NeuraLib API and is launched in parallel on day one. The expansion library is published as **NeuraLib_Models**. It extends the runtime with higher-level builders for LSTMs, GRUs, temporal convolution stacks, residual dense blocks, dueling Q-heads for Reinforcement Learning, Transformer-style attention blocks, and Prioritized Experience Replay utilities.

The important part is architectural: advanced models plug into the same runtime. NeuraLib remains the foundation for tensors, graph execution, optimization, training, inference, datasets, and scaling. After importing `NeuraLib_Models`, its fluent methods become available on NeuraLib `Sequential` models, so the expansion alias does not need to be referenced directly in the layer chain. Its source is in this repository as [NeuraLib_Models.pine](NeuraLib_Models.pine), and its full description is in [MODELS.md](MODELS.md).

```pine
//@version=6
indicator("NeuraLib Models Extension Demo", overlay = false, calc_bars_count = 600)

import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models

var nl.Sequential model = nl.sequential("advanced_demo")

if barstate.isfirst
    model := model
      .input(array.from(8), "sequence")
      .temporalConvStack(4, 2, 3, 2, 2, 1, nl.ActivationKind.relu, 0.0, "temporal")
      .globalAvgPool1d(2, 3, "pool")
      .duelingQHead(4, 2, nl.ActivationKind.relu, "q_head")
      .build(nl.rng(7))
```

---

## Feature quick reference

- **Runtime**: Matrix-first auto-differentiating neural graph runtime for Pine Script.
- **Model API**: Chainable `Sequential` builder with `input`, `dense`, `dropout`, `layerNorm`, `activation`, `flatten`, `reshape`, and custom `block` support.
- **Training**: Forward pass, loss calculation, backpropagation, gradient accumulation, optimizer steps, train stats, and history buffers.
- **Inference**: `.predict()` for deterministic inference and `.predictMC()` for dropout-based uncertainty sampling.
- **Datasets**: `WindowDataset` for flat rows and `RollingDataset` for time-series windows.
- **Scaling**: None, Z-Score, Min-Max, Running Z-Score scalers, dataset input scaling, target scaling, and inverse target scaling.
- **Optimizers**: SGD, Momentum, RMSProp, Adam, and AdamW.
- **Schedulers**: Constant, Step Decay, Cosine Decay, and Warmup Cosine.
- **Activations**: Linear, ReLU, Leaky ReLU, ELU, GELU Approx, Sigmoid, Tanh, Softplus, Swish, and Softmax.
- **Losses**: MSE, MAE, Huber, LogCosh, Binary Cross Entropy, Binary Cross Entropy From Logits, Categorical Cross Entropy, Softmax Cross Entropy From Logits, Directional, Quantile, Multi-Horizon Weighted, and Sharpe.
- **Metrics**: MAE, RMSE, Directional Accuracy, Binary Accuracy, Binary Accuracy From Logits, Categorical Accuracy, and Cosine Similarity.
- **Guardrails**: Shape validation, invalid-row rejection, train/validation split helpers, leakage-aware scaler profiles, training gates, gradient clipping, and EarlyStopper.
- **Advanced expansion**: Conv1D, temporal stacks, recurrent blocks, attention, Transformers, dueling Q-heads, positional encodings, and Prioritized Experience Replay.

---

## Important considerations

- **Start small**: Pine Script is not a GPU training environment. Compact models are the right starting point.
- **Control chart history**: Use `calc_bars_count = 600` in `indicator()` when needed to balance available training history against model size and execution time.
- **Use the training gate**: For heavier models, use `.withTrainingGate(true)` so backpropagation runs only at the confirmed historical edge.
- **Scale your inputs**: Raw market features often differ by orders of magnitude. Use dataset scalers unless you have a deliberate reason not to.
- **Validate separately**: Use `trainBatch()` and `validationBatch()` to monitor generalization instead of only watching training loss.
- **Avoid lookahead**: Build feature rows only from information available at the time of the row. Use completed target rows for training.
- **Treat outputs as research signals**: NeuraLib provides model mechanics. Strategy design, risk management, and market assumptions remain the user's responsibility.

---

## API reference

### Model Setup

- `sequential(name)`: Creates an empty `Sequential` model.
- `compileConfig()`: Creates a model configuration object.
- `build(rng)`: Builds model parameters with a deterministic random stream.
- `compile(config)`: Builds the model when needed and applies the training configuration.
- `rng(seed, streamId)`: Creates a deterministic random stream.

### Sequential Methods

- `input(dimsArray, name)`: Defines the input shape.
- `dense(units, activation, name)`: Adds a fully connected layer.
- `qHead(actionCount, activation, name)`: Adds a Q-value output head.
- `activation(activationKind, alpha, name)`: Adds an activation block.
- `dropout(rate, name)`: Adds dropout regularization.
- `layerNorm(name)`: Adds layer normalization.
- `flatten(name)` and `reshape(outputDimsArray, name)`: Adjust model shape metadata.
- `block(graphBlock)`: Adds a custom `GraphBlock`.
- `trainOnBatch(inputTensor, targetTensor)`: Runs training when the active gate allows it.
- `backward(targetTensor)`: Accumulates gradients from the last forward pass without stepping.
- `step()`: Applies the optimizer step to accumulated gradients.
- `predict(inputTensor)`: Runs inference.
- `predictMC(inputTensor, samples)`: Runs dropout-enabled Monte Carlo prediction and returns mean and variance.
- `evaluate(inputTensor, targetTensor)`: Calculates loss without updating weights.
- `fitDataset(dataset)` and `fitRollingDataset(dataset, targetOffset)`: Train through dataset adapters.
- `getWeightsArray()` and `setWeightsArray(weightsArray)`: Export and import flat model weights.
- `softUpdateFrom(sourceModel, tau)`: Soft-update parameters from another model.

### CompileConfig Methods

- `optimizer(optimizerState)`: Sets the optimizer.
- `schedule(scheduleState)`: Sets the learning-rate schedule.
- `loss(lossKind)`: Sets the training loss.
- `reduction(reductionKind)`: Sets loss reduction behavior.
- `metric(metricKind)`: Adds a metric.
- `batchSize(size)`, `epochsPerBar(count)`, `evalStride(stride)`, and `historyLength(length)`: Store batch and cadence preferences, and set the metric history length.
- `clipNorm(value)` and `clipValue(value)`: Apply gradient clipping.
- `gradAccumSteps(steps)`: Accumulates gradients before stepping.
- `withTrainingGate(enabled)`: Restricts training to the last confirmed historical bar when enabled.
- `trainEveryCall()`: Allows training whenever `.trainOnBatch()` is called.
- `presetPriceRegression()`, `presetReturnRegression()`, `presetBinaryDirection()`, `presetBinaryDirectionLogits()`, `presetQValues()`, and `presetSharpe()`: Apply common loss and metric presets.

### Datasets

- `windowDataset(featureCount, targetCount, maxRows, name)`: Stores flat feature and target rows.
- `rollingDataset(timeSteps, featureCount, targetCount, maxRows, name)`: Stores time-series windows.
- `pushRow(featureArray, targetArray)`: Adds one validated row.
- `pushBuilderRow(featureBuilder, targetArray)`: Adds a row from a `FeatureBuilder`.
- `pushNextReturnRow(featureBuilder, currentValue, futureValue)`: Adds a next-return target.
- `pushNextDirectionRow(featureBuilder, currentValue, futureValue, threshold, zeroOne)`: Adds a direction target.
- `ready(minRows or minWindows, targetOffset)` and `size()`: Check dataset readiness.
- `lastBatch(batchSize)`: Returns the most recent scaled rows from a `WindowDataset`.
- `toBatch()`: Returns all rows from a `WindowDataset`.
- `unrollBatch(targetOffset)`: Returns all rolling windows from a `RollingDataset`.
- `trainBatch(validationRows or validationWindows, targetOffset)`: Returns the training side of the split.
- `validationBatch(validationRows or validationWindows, targetOffset)`: Returns the validation side of the split.
- `setInputScaler(kind)`, `setTargetScaler(kind)`, `scaleInput(tensor)`, `scaleTarget(tensor)`, and `inverseScaleTarget(tensor)`: Configure and apply scaling.
- `clear()`: Clears stored rows.

### Tensor, Matrix, and Feature Helpers

- `scalar(value)`, `vector(valuesArray)`, `matrix2d(rows, cols, fillValue)`, `zeros(shape)`, `ones(shape)`, and `full(shape, fillValue)`: Create tensors.
- `shapeFromDims(dimsArray)`: Creates a shape.
- `matrixTensor(tensor)`, `matrixTensor2d(rows, cols, fillValue)`, and `matrixTensorFromMatrix(sourceMatrix)`: Create matrix tensors.
- `reshape(dimsArray)`, `flatten()`, `row(rowIndex)`, `get1d(index)`, `sum()`, `mean()`, `variance()`, `normL2()`, `argmax()`, and `dot(other)`: Tensor methods.
- `matmul()`, `transpose()`, `add()`, `subtract()`, `multiply()`, `divide()`, `scale()`, `activate()`, `softmax()`, `sliceRows()`, `sliceCols()`, `concatRows()`, `concatCols()`, `globalAvgPool1d()`, and `globalMaxPool1d()`: MatrixTensor methods.
- `featureBuilder(name)`, `push(value, featureName)`, `addFeature(value, featureName)`, `toTensor(tensorName)`, `toArray()`, `size()`, and `clear()`: Feature row helpers.

### Scalers, Optimizers, and Schedules

- `zScoreScaler()`, `minMaxScaler()`, `runningZScoreScaler()`, and `noneScaler()`: Standalone scaler states.
- `fit(tensor)`, `partialFit(tensor)`, `transform(tensor)`, and `inverseTransform(tensor)`: Scaler methods.
- `sgd(learningRate)`, `momentum(learningRate, momentum)`, `rmsprop(learningRate, rho, epsilon)`, `adam(learningRate, beta1, beta2, epsilon)`, and `adamW(learningRate, beta1, beta2, epsilon, weightDecay)`: Optimizers.
- `constantSchedule(learningRate)`, `stepDecay(baseLearningRate, decaySteps, gamma)`, `cosineDecay(baseLearningRate, minLearningRate, decaySteps)`, and `warmupCosine(baseLearningRate, minLearningRate, warmupSteps, decaySteps)`: Schedules.
- `currentRate(stepCount)`: Reads a schedule's learning rate at a step.
- `paramBank()`, `append()`, `zeroGrad()`, `globalGradNorm()`, `step(optimizerState)`, and `softUpdateFrom(sourceBank, tau)`: Low-level parameter bank utilities.

### Losses and Metrics

- `mse()`, `mae()`, `huber()`, `logCosh()`, `binaryCrossEntropy()`, `binaryCrossEntropyFromLogits()`, `categoricalCrossEntropy()`, `softmaxCrossEntropyFromLogits()`, `directionalLoss()`, `quantileLoss()`, `multiHorizonWeighted()`, and `sharpeLoss()`: Direct loss helpers.
- `metricValue(metricKind, predictionTensor, targetTensor)`: Direct metric helper.
- `earlyStopper(patience, minDelta)`, `update(validationLoss)`, and `reset()`: Validation stopping helper.
- `nextReturnValue(currentValue, futureValue)` and `nextDirectionValue(currentValue, futureValue, threshold, zeroOne)`: Common target helpers.

### GraphBlock Operations

- `graphBlock(name)`: Creates a custom trainable graph block.
- `input()`, `param()`, `constScalar()`, `constMatrix()`, and `output()`: Define graph inputs, parameters, constants, and output metadata.
- `matmul()`, `add()`, `subtract()`, `multiply()`, `divide()`, `scale()`, `activate()`, `softmax()`, `transpose()`, `layerNorm()`, and `dropout()`: NeuraLib graph math.
- `causalMask()`, `sliceRows()`, `concatRows()`, `sliceCols()`, `concatCols()`, `reduceRows()`, and `reduceCols()`: Structural graph operations.
- `globalAvgPool1d()`, `globalMaxPool1d()`, `attentionScore()`, `attentionApply()`, `conv1d()`, `scanLstm()`, and `scanGru()`: Sequence and architecture primitives.

### NeuraLib_Models API

- `prioritizedReplayBuffer(featureCount, targetCount, maxRows, name)`: Creates a replay buffer.
- `pushExperience(featureRowArray, targetRowArray, priority)`, `sampleBatch(batchSize, alpha, beta, seed)`, `updatePriority(index, priority)`, `toBatch()`, `ready(minRows)`, `size()`, and `clear()`: Prioritized Experience Replay helpers.
- `pushPositionalEncoding(position, dimensions, maxPeriod, featurePrefix)`: Adds positional encoding values to a `FeatureBuilder`.
- `residualDense()`, `duelingQHead()`, `conv1d()`, `temporalConvStack()`, `globalAvgPool1d()`, `globalMaxPool1d()`, `lstm()`, `gru()`, `selfAttention()`, `multiHeadSelfAttention()`, `crossAttention()`, `transformerEncoder()`, `transformerEncoderStack()`, and `transformerDecoder()`: NeuraLib_Models `Sequential` methods.

*NeuraLib is for Pine Script developers who want to move beyond fixed formulas and experiment with real neural network workflows directly inside TradingView. It is a research framework, not a guarantee of market performance. Use validation, avoid lookahead, control risk, and keep models small enough for Pine's execution limits.*

*All the diagrams in this publication are rendered natively on TradingView using [Pine3D](https://www.tradingview.com/script/I2K4hIdP-Pine3D-A-Native-3D-Graphical-Rendering-Engine/)*

---

## Related libraries

- [Pine3D](https://github.com/AlienAlgorithms/Pine3D-A-Native-3D-Graphical-Rendering-Engine): 3D rendering in Pine Script. It draws the diagrams in the NeuraLib publication.
- [PineCanvas](https://github.com/AlienAlgorithms/PineCanvas-A-Native-High-Resolution-Graphics-Engine-for-TradingView): heatmaps, high-resolution images and filled shapes in Pine Script.

## Licence

NeuraLib and NeuraLib_Models are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). The full legal text is in [LICENSE](LICENSE).

- **Non-commercial use is free** when [Alien_Algorithms](https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts) is credited in the description of any script that uses it.
- **Share alike:** a published adaptation must carry the same licence.
- **Commercial use** needs a separate licence. Contact [Alien_Algorithms on TradingView](https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts).

## Author

Built by [Alien_Algorithms on TradingView](https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts) and [AlienAlgorithms on GitHub](https://github.com/AlienAlgorithms). More indicators and libraries at [alienalgorithms.com](https://alienalgorithms.com).
