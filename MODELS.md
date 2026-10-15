# NeuraLib_Models

The companion model expansion for NeuraLib: temporal convolution, LSTM and GRU, attention and Transformer blocks, residual dense blocks, dueling Q-heads and prioritized experience replay. Back to the [NeuraLib README](README.md).

[![NeuraLib_Models on TradingView](https://s3.tradingview.com/d/drHsBXE4.png?v=1777329636)](https://www.tradingview.com/script/drHsBXE4-NeuraLib-Expansion-Advanced-Model-Layers/)

- **TradingView publication:** [NeuraLib Expansion: Advanced Model Layers](https://www.tradingview.com/script/drHsBXE4-NeuraLib-Expansion-Advanced-Model-Layers/)
- **Source:** [NeuraLib_Models.pine](NeuraLib_Models.pine), version 1

```pine
import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models
```

---

**NeuraLib_Models** is the companion model expansion for **NeuraLib**.

NeuraLib provides the runtime: tensors, graph execution, datasets, scalers, losses, optimizers, training, inference, and validation tools. NeuraLib_Models builds on that foundation with higher-level neural architectures that are difficult and repetitive to write by hand.

The purpose of this expansion is to keep the main NeuraLib runtime clean, compact, and general, while giving researchers ready-to-use model families for sequence learning, attention, temporal pattern extraction, and Reinforcement Learning workflows.

---

## How it fits into NeuraLib

NeuraLib_Models is built entirely on top of the public NeuraLib API. It does not replace the main runtime and it does not introduce a separate training engine.

After importing NeuraLib_Models, its fluent methods become available directly on NeuraLib `Sequential` models. The expansion alias can remain unused in the layer chain.

```pine
//@version=6
indicator("NeuraLib Models Quick Start", overlay = false, calc_bars_count = 600)

import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models

var nl.Sequential model = nl.sequential("advanced_model")
var float qLong = na
var float qFlat = na
var float qShort = na

if barstate.isfirst
    model := model
      .input(array.from(8), "sequence")
      .temporalConvStack(4, 2, 2, 2, 1, 1, nl.ActivationKind.relu, 0.0, "temporal")
      .globalAvgPool1d(3, 2, "pool")
      .duelingQHead(4, 3, nl.ActivationKind.relu, "dueling_head")
      .build(nl.rng(7))

float ret0 = na(close[1]) or close[1] == 0.0 ? 0.0 : close / close[1] - 1.0
float ret1 = na(close[2]) or close[2] == 0.0 ? 0.0 : close[1] / close[2] - 1.0
float ret2 = na(close[3]) or close[3] == 0.0 ? 0.0 : close[2] / close[3] - 1.0
float ret3 = na(close[4]) or close[4] == 0.0 ? 0.0 : close[3] / close[4] - 1.0

float atrValue = ta.atr(14)
float atr0 = close == 0.0 ? 0.0 : atrValue / close
float atr1 = close[1] == 0.0 ? 0.0 : atrValue[1] / close[1]
float atr2 = close[2] == 0.0 ? 0.0 : atrValue[2] / close[2]
float atr3 = close[3] == 0.0 ? 0.0 : atrValue[3] / close[3]

bool ready = not na(ret3) and not na(atr3)
if ready
    nl.Tensor state = nl.vector(array.from(ret3, atr3, ret2, atr2, ret1, atr1, ret0, atr0), "state_window")
    nl.Tensor qValues = model.predict(state)
    qLong := qValues.get1d(0)
    qFlat := qValues.get1d(1)
    qShort := qValues.get1d(2)

plot(qLong, "Q long", color = color.lime, linewidth = 2)
plot(qFlat, "Q flat", color = color.gray)
plot(qShort, "Q short", color = color.red, linewidth = 2)
hline(0.0, "Zero", color = color.new(color.gray, 70))
```

The model is still a normal NeuraLib model. You still call `.compile()`, `.trainOnBatch()`, `.predict()`, `.evaluate()`, `.getWeightsArray()`, and `.softUpdateFrom()` from the main library.

---

## Why this expansion exists

The main NeuraLib library is the foundation. It exposes a graph engine powerful enough to create custom architectures, but repeatedly building LSTM gates, attention projections, residual blocks, Conv1D stacks, or Transformer paths from raw graph operations would be too verbose for everyday research.

NeuraLib_Models packages those patterns into readable blocks:

- **Temporal models**: Conv1D blocks, temporal convolution stacks, global average pooling, and global max pooling for flattened sequence inputs.
- **Recurrent models**: LSTM and GRU blocks for compact sequence memory.
- **Attention models**: Self-attention, multi-head self-attention, cross-attention, Transformer encoder blocks, Transformer encoder stacks, and Transformer decoder blocks.
- **Residual models**: Residual dense blocks for deeper feedforward paths.
- **Reinforcement Learning heads**: Q-head blocks and dueling Q-heads for action-value style outputs.
- **Replay utilities**: Deterministic Prioritized Experience Replay for reproducible Pine research.
- **Sequence helpers**: Positional encoding for token, sequence, and attention workflows.

---

## Practical examples

### Temporal Conv Model With Dueling Q-Head

This pattern is useful when a flattened sequence contains recent market states and the output represents action values.

```pine
//@version=6
indicator("NeuraLib Models Temporal Q Example", overlay = false, calc_bars_count = 600)

import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models

var nl.Sequential qModel = nl.sequential("temporal_q_model")
var nl.WindowDataset qDataset = nl.windowDataset(8, 3, 400, "q_rows")
var float qDown = na
var float qNeutral = na
var float qUp = na
var float qLoss = na

if barstate.isfirst
    nl.CompileConfig cfg = nl.compileConfig()
    cfg := cfg
      .presetQValues()
      .optimizer(nl.adamW(0.001))
      .withTrainingGate(true)

    qModel := qModel
      .input(array.from(8), "state_window")
      .temporalConvStack(4, 2, 2, 2, 1, 1, nl.ActivationKind.relu, 0.0, "temporal")
      .globalAvgPool1d(3, 2, "pool")
      .duelingQHead(4, 3, nl.ActivationKind.relu, "dueling_head")
      .compile(cfg)

    qDataset := qDataset
      .setInputScaler(nl.ScalerKind.zScore)
      .setTargetScaler(nl.ScalerKind.none)

float ret0 = na(close[1]) or close[1] == 0.0 ? 0.0 : close / close[1] - 1.0
float ret1 = na(close[2]) or close[2] == 0.0 ? 0.0 : close[1] / close[2] - 1.0
float ret2 = na(close[3]) or close[3] == 0.0 ? 0.0 : close[2] / close[3] - 1.0
float ret3 = na(close[4]) or close[4] == 0.0 ? 0.0 : close[3] / close[4] - 1.0
float ret4 = na(close[5]) or close[5] == 0.0 ? 0.0 : close[4] / close[5] - 1.0

float atrValue = ta.atr(14)
float atr0 = close == 0.0 ? 0.0 : atrValue / close
float atr1 = close[1] == 0.0 ? 0.0 : atrValue[1] / close[1]
float atr2 = close[2] == 0.0 ? 0.0 : atrValue[2] / close[2]
float atr3 = close[3] == 0.0 ? 0.0 : atrValue[3] / close[3]
float atr4 = close[4] == 0.0 ? 0.0 : atrValue[4] / close[4]

bool rowReady = not na(ret4) and not na(atr4)
if rowReady
    array<float> features = array.from(ret4, atr4, ret3, atr3, ret2, atr2, ret1, atr1)
    float downTarget = math.max(-ret0, 0.0)
    float neutralTarget = math.max(0.002 - math.abs(ret0), 0.0)
    float upTarget = math.max(ret0, 0.0)
    qDataset := qDataset.pushRow(features, array.from(downTarget, neutralTarget, upTarget))

if qDataset.ready(48)
    if barstate.islastconfirmedhistory
        nl.Batch train = qDataset.trainBatch(12)
        qModel := qModel.trainOnBatch(train.inputTensor, train.targetTensor)
        qLoss := qModel.trainStats.lastLoss

    nl.Tensor liveState = nl.vector(array.from(ret3, atr3, ret2, atr2, ret1, atr1, ret0, atr0), "live_state")
    nl.Tensor scaledState = qDataset.scaleInput(liveState)
    nl.Tensor qValues = qModel.predict(scaledState)
    qDown := qValues.get1d(0)
    qNeutral := qValues.get1d(1)
    qUp := qValues.get1d(2)

plot(qDown, "Q down", color = color.red, linewidth = 2)
plot(qNeutral, "Q neutral", color = color.gray)
plot(qUp, "Q up", color = color.lime, linewidth = 2)
plot(qLoss, "Training loss", color = color.orange)
hline(0.0, "Zero", color = color.new(color.gray, 70))
```

Input shape `array.from(8)` represents a flattened 4 step by 2 feature sequence. The temporal stack extracts short sequence structure, pooling compresses the sequence, and the dueling head separates value and advantage paths before producing action scores. The example trains only on the last confirmed historical bar so it remains safe to paste onto long charts.

### Transformer Encoder For Token Rows

Attention models are useful when each row is a token or time step, and each column is a feature dimension.

```pine
//@version=6
indicator("NeuraLib Models Transformer Encoder Example", overlay = false, calc_bars_count = 600)

import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models

var nl.Sequential encoder = nl.sequential("encoder_model")
var float tokenSignal = na
var float tokenContext = na
var float tokenVolatility = na

if barstate.isfirst
    encoder := encoder
      .input(array.from(4), "tokens")
      .multiHeadSelfAttention(4, 2, true, "mha")
      .transformerEncoder(4, true, 2, nl.ActivationKind.geluApprox, "encoder", 0.05, 2)
      .build(nl.rng(11))

float emaValue = ta.ema(close, 21)
float atrValue = ta.atr(14)

float ret0 = na(close[1]) or close[1] == 0.0 ? 0.0 : close / close[1] - 1.0
float ret1 = na(close[2]) or close[2] == 0.0 ? 0.0 : close[1] / close[2] - 1.0
float ret2 = na(close[3]) or close[3] == 0.0 ? 0.0 : close[2] / close[3] - 1.0
float ret3 = na(close[4]) or close[4] == 0.0 ? 0.0 : close[3] / close[4] - 1.0

float emaGap0 = emaValue == 0.0 ? 0.0 : close / emaValue - 1.0
float emaGap1 = emaValue[1] == 0.0 ? 0.0 : close[1] / emaValue[1] - 1.0
float emaGap2 = emaValue[2] == 0.0 ? 0.0 : close[2] / emaValue[2] - 1.0
float emaGap3 = emaValue[3] == 0.0 ? 0.0 : close[3] / emaValue[3] - 1.0

float atr0 = close == 0.0 ? 0.0 : atrValue / close
float atr1 = close[1] == 0.0 ? 0.0 : atrValue[1] / close[1]
float atr2 = close[2] == 0.0 ? 0.0 : atrValue[2] / close[2]
float atr3 = close[3] == 0.0 ? 0.0 : atrValue[3] / close[3]

bool ready = not na(ret3) and not na(emaGap3) and not na(atr3)
if ready
    nl.Tensor tokens = nl.vector(array.from(
      ret3, emaGap3, atr3, -1.0,
      ret2, emaGap2, atr2, -0.33,
      ret1, emaGap1, atr1, 0.33,
      ret0, emaGap0, atr0, 1.0), "tokens").reshape(array.from(4, 4))

    nl.Tensor encoded = encoder.predict(tokens)
    tokenSignal := encoded.get1d(12)
    tokenContext := encoded.get1d(13)
    tokenVolatility := encoded.get1d(14)

plot(tokenSignal, "Latest token signal", color = color.aqua, linewidth = 2)
plot(tokenContext, "Latest token context", color = color.purple)
plot(tokenVolatility, "Latest token volatility", color = color.orange)
hline(0.0, "Zero", color = color.new(color.gray, 70))
```

In this example, each input row has 4 features. `headCount` is 2, so the model dimension is split into two attention heads.

*Attention rule: `modelDim` must be divisible by `headCount`, and the current implementation supports up to 8 heads.*

### Prioritized Experience Replay

Prioritized Experience Replay stores examples with priorities, then returns reproducible weighted samples. This is especially useful for Reinforcement Learning experiments where high-error transitions should be revisited more often.

```pine
//@version=6
indicator("NeuraLib Models PER Example", overlay = false, calc_bars_count = 1200)

import Alien_Algorithms/NeuraLib/1 as nl
import Alien_Algorithms/NeuraLib_Models/1 as models

var models.PrioritizedReplayBuffer replay = models.prioritizedReplayBuffer(4, 2, 300, "replay")
var nl.Sequential replayModel = nl.sequential("replay_q_model")
var float replayLoss = na
var float firstImportanceWeight = na
var float replayRows = na

if barstate.isfirst
    nl.CompileConfig cfg = nl.compileConfig()
    cfg := cfg
      .presetQValues()
      .optimizer(nl.adamW(0.001))
      .trainEveryCall()

    replayModel := replayModel
      .input(array.from(4), "state")
      .dense(8, nl.ActivationKind.relu, "hidden")
      .qHead(2, nl.ActivationKind.linear, "q_values")
      .compile(cfg)

float rsiValue = ta.rsi(close, 14)
float emaValue = ta.ema(close, 21)
float atrValue = ta.atr(14)
float atrPct = close == 0.0 ? 0.0 : atrValue / close
float momentum = na(close[1]) or close[1] == 0.0 ? 0.0 : close / close[1] - 1.0
float nextReturn = na(close[1]) ? 0.0 : nl.nextReturnValue(close[1], close)

bool rowReady = not na(rsiValue[1]) and not na(emaValue[1]) and not na(atrPct[1]) and not na(momentum[1])
if rowReady
    float prevEma = emaValue[1]
    float priceVsEma = prevEma == 0.0 ? 0.0 : close[1] / prevEma - 1.0
    array<float> stateFeatures = array.from(rsiValue[1] / 100.0, priceVsEma, atrPct[1], momentum[1])
    array<float> targetValues = array.from(math.max(-nextReturn, 0.0), math.max(nextReturn, 0.0))
    float priority = math.abs(nextReturn) + 0.0001
    replay := replay.pushExperience(stateFeatures, targetValues, priority)
    replayRows := float(replay.size())

if replay.ready(32)
    models.PrioritizedReplaySample sample = replay.sampleBatch(32, 0.6, 0.4, 17)
    replayModel := replayModel.trainOnBatch(sample.batch.inputTensor, sample.batch.targetTensor)
    replayLoss := replayModel.trainStats.lastLoss
    firstImportanceWeight := sample.weightArray.size() > 0 ? sample.weightArray.get(0) : na
    if sample.indexArray.size() > 0
        replay := replay.updatePriority(sample.indexArray.get(0), replayLoss + 0.0001)

plot(replayLoss, "Replay training loss", color = color.orange, linewidth = 2)
plot(firstImportanceWeight, "First sample weight", color = color.aqua)
```

The returned sample includes:

- **batch**: A normal NeuraLib `Batch` containing sampled inputs and targets.
- `indexArray`: Logical replay indices that can be passed back to `updatePriority()`.
- `weightArray`: Normalized importance weights for custom loss weighting or diagnostics.
- `sampleRows`: Number of sampled rows.

PER sampling is deterministic for a given buffer, `batchSize`, and `seed`. That makes Pine tests and live research easier to reproduce.

---

## Model families

### Residual Dense Blocks
`residualDense()` adds a feedforward residual block. Residual paths help preserve information through deeper models and reduce the chance that a dense stack destroys useful features too early.

### Conv1D And Temporal Convolution Stacks
`conv1d()` and `temporalConvStack()` operate on flattened sequence inputs. A sequence with `timeSteps = 4` and `featureCount = 2` is represented as 8 input features. These blocks are useful for local temporal structure, short rolling windows, feature rhythm, and compact pattern extraction.

### Global Pooling
`globalAvgPool1d()` and `globalMaxPool1d()` compress flattened sequence outputs into feature-level summaries. Average pooling captures broad sequence behavior, while max pooling emphasizes the strongest activation per feature.

### LSTM And GRU Blocks
`lstm()` and `gru()` provide recurrent sequence memory over flattened time-series inputs. They are useful when the order of recent states matters more than a single snapshot.

### Attention And Transformers
`selfAttention()`, `multiHeadSelfAttention()`, `crossAttention()`, `transformerEncoder()`, `transformerEncoderStack()`, and `transformerDecoder()` bring attention-style modeling into Pine. They are designed for compact token matrices, packed target-memory layouts, and small Transformer-style research models that fit TradingView limits.

### Q-Heads And Dueling Q-Heads
`qHeadBlock()` creates action-value style outputs. `duelingQHead()` splits the model into value and advantage branches, then recombines them into Q-values. This is useful when you want the model to estimate both the overall state value and the relative value of each action.

### Positional Encoding
`pushPositionalEncoding()` adds sinusoidal position features to a NeuraLib `FeatureBuilder`. This helps attention-style models distinguish where a token or time step sits in a sequence.

---

## Feature quick reference

- **Built on NeuraLib**: Uses the main NeuraLib graph, tensor, training, optimizer, dataset, and inference runtime.
- **Fluent API**: Adds methods directly to NeuraLib `Sequential` models after import.
- **Block factories**: Provides standalone `GraphBlock` factories for users who want lower-level composition.
- **Temporal modeling**: Conv1D, temporal convolution stacks, and 1D pooling.
- **Recurrent modeling**: LSTM and GRU sequence blocks.
- **Attention modeling**: Self-attention, multi-head self-attention, cross-attention, encoders, encoder stacks, and decoders.
- **Reinforcement Learning support**: Q-heads, dueling Q-heads, target-model soft updates through NeuraLib, and Prioritized Experience Replay.
- **Reproducible replay**: PER sampling is deterministic for a given seed.
- **Shape guardrails**: Advanced builders validate expected model feature counts and attention head compatibility.

---

## Important usage notes

- **Import order matters**: Import `NeuraLib` first, then `NeuraLib_Models`.
- **The alias can be unused**: The imported expansion registers methods on NeuraLib types, so `.lstm()`, `.gru()`, `.transformerEncoder()`, and similar methods can be called in the model chain.
- **Keep models compact**: Pine Script has execution limits. Start with small hidden sizes, short sequences, and low head counts.
- **Control chart history**: Use `calc_bars_count = 600` in `indicator()` when needed to balance available training history against model size and execution time.
- **Respect sequence shapes**: Conv1D, temporal stacks, LSTM, and GRU methods expect flattened sequence sizes of `timeSteps \* featureCount`.
- **Respect attention shapes**: Attention methods expect each input row to have `modelDim` columns. Cross-attention and decoder blocks use packed rows.
- **Use NeuraLib guardrails**: Train/validation splits, scalers, EarlyStopper, training gates, and gradient clipping remain part of the main NeuraLib workflow.

---

## API reference

### Sequential Methods

- `residualDense(hiddenUnits, activationKind, dropoutRate, name)`: Adds a residual dense block.
- `duelingQHead(hiddenUnits, actionCount, activationKind, name)`: Adds a dueling value/advantage Q-head.
- `conv1d(timeSteps, featureCount, filters, kernelSize, stride, activationKind, name)`: Adds a Conv1D block for flattened sequences.
- `temporalConvStack(timeSteps, featureCount, filters, kernelSize, layers, stride, activationKind, dropoutRate, name)`: Adds stacked temporal Conv1D layers.
- `globalAvgPool1d(timeSteps, featureCount, name)`: Adds global average pooling over a flattened 1D sequence.
- `globalMaxPool1d(timeSteps, featureCount, name)`: Adds global max pooling over a flattened 1D sequence.
- `lstm(timeSteps, featureCount, units, activationKind, name)`: Adds an LSTM scan block.
- `gru(timeSteps, featureCount, units, activationKind, name)`: Adds a GRU scan block.
- `selfAttention(modelDim, causal, name)`: Adds row-wise self-attention.
- `multiHeadSelfAttention(modelDim, headCount, causal, name)`: Adds multi-head self-attention.
- `crossAttention(queryRows, memoryRows, modelDim, headCount, name)`: Adds packed query-memory cross-attention.
- `transformerEncoder(modelDim, causal, ffMultiplier, activationKind, name, dropoutRate, headCount)`: Adds one Transformer encoder block.
- `transformerEncoderStack(modelDim, layers, causal, ffMultiplier, activationKind, dropoutRate, headCount, name)`: Adds repeated Transformer encoder blocks.
- `transformerDecoder(targetRows, memoryRows, modelDim, headCount, ffMultiplier, activationKind, dropoutRate, name)`: Adds a packed target-memory Transformer decoder.

### GraphBlock Factories

- `qHeadBlock(inputFeatures, actionCount, activationKind, name)`: Creates a Q-head block.
- `duelingQHeadBlock(inputFeatures, hiddenUnits, actionCount, activationKind, name)`: Creates a dueling Q-head block.
- `residualDenseBlock(inputFeatures, hiddenUnits, activationKind, dropoutRate, name)`: Creates a residual dense block.
- `conv1dBlock(timeSteps, featureCount, filters, kernelSize, stride, activationKind, name)`: Creates a Conv1D block.
- `temporalConvStackBlock(timeSteps, featureCount, filters, kernelSize, layers, stride, activationKind, dropoutRate, name)`: Creates a temporal convolution stack.
- `globalAvgPool1dBlock(timeSteps, featureCount, name)` and `globalMaxPool1dBlock(timeSteps, featureCount, name)`: Create pooling blocks.
- `lstmBlock(timeSteps, featureCount, units, activationKind, name)` and `gruBlock(timeSteps, featureCount, units, activationKind, name)`: Create recurrent blocks.
- `selfAttentionBlock(modelDim, causal, name)`, `multiHeadSelfAttentionBlock(modelDim, headCount, causal, name)`, and `crossAttentionBlock(queryRows, memoryRows, modelDim, headCount, name)`: Create attention blocks.
- `transformerEncoderBlock(modelDim, causal, ffMultiplier, activationKind, name, dropoutRate, headCount)` and `transformerDecoderBlock(targetRows, memoryRows, modelDim, headCount, ffMultiplier, activationKind, dropoutRate, name)`: Create Transformer blocks.

### Prioritized Experience Replay

- `prioritizedReplayBuffer(featureCount, targetCount, maxRows, name)`: Creates a replay buffer.
- `pushExperience(featureRowArray, targetRowArray, priority)`: Adds or overwrites one replay row.
- `sampleBatch(batchSize, alpha, beta, seed)`: Returns a deterministic weighted sample.
- `updatePriority(index, priority)`: Updates a sampled row priority.
- `toBatch()`: Returns all replay rows in chronological order.
- `ready(minRows)`, `size()`, and `clear()`: Replay buffer utilities.

### Feature Helpers

- `pushPositionalEncoding(position, dimensions, maxPeriod, featurePrefix)`: Appends sinusoidal positional encoding values to a NeuraLib `FeatureBuilder`.

*NeuraLib_Models is for Pine Script developers who want higher-level neural architecture blocks without leaving the NeuraLib runtime. It is built for compact research models inside TradingView's execution limits, not for oversized GPU-style networks.*

*All the diagrams in this publication are rendered natively on TradingView using [Pine3D](https://www.tradingview.com/script/I2K4hIdP-Pine3D-A-Native-3D-Graphical-Rendering-Engine/)*

---

## Licence

NeuraLib_Models is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). The full legal text is in [LICENSE](LICENSE).

- **Non-commercial use is free** when [Alien_Algorithms](https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts) is credited in the description of any script that uses it.
- **Share alike:** a published adaptation must carry the same licence.
- **Commercial use** needs a separate licence. Contact [Alien_Algorithms on TradingView](https://www.tradingview.com/u/Alien_Algorithms/?aff_id=169033#published-scripts).
