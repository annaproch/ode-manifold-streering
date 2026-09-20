# Colour-manifold pilot implementation

**Implementation status: written, not executed or tested.** No installation, model download, model load, numerical preflight, unit test, or experiment has been performed for this implementation. The commands below are for a later explicit run.

The research design remains in [../experiment-plan.md](../experiment-plan.md). This file documents operating the code; it is not a second experiment plan. The implementation covers the active colour pilot and its optional steering ablations. Deferred emotion, SAE, geography and alignment-benchmark studies are not silently included.

## Devices and your 128 GB machine

| Profile | Model inference | Numerical steering/fitting |
|---|---|---|
| `configs/cpu-smoke.json` | CPU, float32, batch 1 | CPU NumPy/SciPy |
| `configs/cpu-full.json` | CPU, float32, batch 2; full evaluation exceeds the default budget | CPU NumPy/SciPy |
| `configs/mps-128gb.json` | Apple Silicon MPS, float16, batch 4 | CPU NumPy/SciPy |
| `configs/cuda.json` | NVIDIA CUDA, bfloat16, batch 8 | CPU NumPy/SciPy |

Use MPS if the unified-memory machine is an Apple Silicon Mac; use CUDA if it is an NVIDIA system. Both profiles are provided because unified memory does not identify the accelerator. For CUDA hardware without bfloat16 support, change `model.dtype` to `float16` before starting a new run. `model.device: "auto"` is also supported but explicit profiles make runs easier to compare.

The 8B model's weights alone are approximately 16 GB in float16/bfloat16 or 32 GB in float32, with additional inference/loading overhead. The profiles use small batches and no training optimizer. CPU inference can still be slow despite sufficient memory. No quantization, device sharding or automatic CPU offloading is used. Adjust batch size before a run rather than changing precision midway through an experiment.

The model adapter targets Hugging Face decoder models with `model.model.layers`, including Llama-style models. The code deliberately keeps PCA, KDE, local SVD, graphs and the independent polynomial sketch on CPU. CUDA/MPS accelerate the expensive model forwards; they do not change the reference geometry implementation. This avoids requiring MPS support for double-precision linear algebra or FFTs.

## Installation, when ready

Python 3.11 or 3.12 is recommended. From the repository root, create a dedicated environment. These are instructions only; they have not been run:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r pilot/requirements.txt
```

For Linux CPU-only or a particular CUDA runtime, install the appropriate PyTorch wheel using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the requirements. The Transformers adapter is written against [version 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model), which is pinned in requirements. Dependency compatibility is unverified until the deferred tests run.

The default model is the **base** `meta-llama/Llama-3.1-8B`, with raw prompts. All profiles default to `local_files_only: true`, so missing model files cause an error rather than a download. Set `model.name` to a local immutable checkpoint directory, or cache an accessible checkpoint yourself. To allow the CLI to download later, explicitly change `local_files_only` to false and configure normal Hugging Face access. No access token is recorded in artifacts.

For a remote model, preferably set `model.revision` to a commit hash. If it is left as `main`, the first model load records the resolved revision, and subsequent stages request that revision. Local weight directories must remain immutable; the manifest records their path and config hash, not a full hash of multi-GB weights. If using an instruct checkpoint, set both its model name and `chat_template: true` before starting a new run.

## Configuration

JSON files override the defaults in [config.py](config.py); unknown keys are rejected. Copy a supplied profile, set model location/revision and output path, and retain the effective settings with the run. All layer numbers are **zero-based decoder output indices**. The default candidate indices are 18, 24 and 27; these are candidates, not asserted replications of a paper's numbering.

The grid uses HSL, not HSV. The primary curve (saturation 0.8, lightness 0.5) samples hue every **0.25°**, deduplicated after rounding to hex. Other saturation/lightness slices (0.65/0.8/0.95 × 0.35/0.5/0.65) use a separate coarse 6° grid for context plots. Gray, brown and unrelated controls are included. Every template for a given hex has the same split. At fixed saturation/lightness, 8-bit RGB quantization limits distinct hexes: a finer hue grid cannot create 2,000–3,000 independent colours on this curve. Inspect actual counts rather than counting requested grid points or repeated templates.

The smoke profile reduces the grid to 90 primary-slice hue inputs, plus controls, uses two layers/two PCA dimensions and one seed. Readout validation is limited to 100 labeled examples. The subsequent smoke collection is larger than 100 because it also includes controls and held-out splits. It is not a confirmatory three-seed study, and a missing reference class or insufficient split coverage is a real gate failure rather than an excuse to use training examples as test data.

Data are split by hex identity into training (45%), independent natural reference (10%), validation (22.5%), and test (22.5%). `coverage.json` reports actual primary starts by band after deduplication and subsampling. Full profiles require at least 20 per 90°/120°/150° band and 10 at 180° in each evaluation split; collection refuses insufficient coverage. These are minimum coverage safeguards, not a power calculation. Selection checks counts again after gating for each fitted seed; test reports cannot claim a criterion pass with insufficient eligible main-band counts. At least eight model-labeled target references are required for full evaluation. The smoke profile lowers these minima and cannot establish a research result.

The classifier, PCA, KDE and spline are fitted on **main-template primary-slice training data**. Other slices are plotted, not pooled into the claimed evaluation count; change the primary values and output directory for another conditional experiment. The classifier target is the unsteered model's argmax label. A paraphrase is a held-out transfer test and does not multiply the independent sample size.

Strengths are dimensionless multipliers `[0, 0.25, 0.5, 1, 2, 4]`. Before validation, each layer's training data fixes a scale: by default, the median full-activation distance between the target centroid and each available non-target chromatic centroid. `strength_scale: "activation_norm"` instead uses the median training activation norm; `"raw"` disables scaling. Saved rows contain both multiplier and raw T, and fits retain the calibration scales. The solver's T always means raw activation length.

Choose practical effect thresholds *before inspecting outcome data*. The defaults `endpoint_practical_improvement` and `path_practical_improvement` are null; those runs remain exploratory. For example, a value of 0.2 means a required 20% relative reduction, but this is a project choice, not a scientifically established threshold. The success level defaults to 0.8 and the between-method test success tolerance to 0.05. Freeze these before `validate`, not after seeing its curves.

## Explicit stages

There is no `run-all` command or import-time work. Run one stage, inspect its output, then decide whether the next stage is justified. Use the same configuration path throughout. The examples below use the MPS profile; replace that path with a CPU/CUDA profile as appropriate.

### 1. Inspect settings and prepare data

```sh
python -m pilot --config pilot/configs/mps-128gb.json plan
python -m pilot --config pilot/configs/mps-128gb.json data
python -m pilot --config pilot/configs/mps-128gb.json estimate
python -m pilot --config pilot/configs/mps-128gb.json estimate --split test
```

`plan` prints configuration and stage order without loading a model or creating artifacts. `data` writes `stimuli.jsonl`, its manifest and `coverage.json`. `estimate` counts prompt-state decodes and batched forwards without loading a model. It is a conservative upper bound before caching, not a timing prediction. Evaluation checks `maximum_decode_examples` before loading weights. The CPU profiles cap it at 1,000: fitting in RAM does not make tens of thousands of 8B forwards practical. Use CPU for preflight, readout and small checks; use MPS/CUDA for the full evaluation. Raising the cap requires an explicit config edit after reviewing the estimate. All paths are relative to the repository working directory unless absolute.

### 2. Numerical preflight and tests

```sh
python -m unittest discover -s pilot/tests -v
python -m pilot --config pilot/configs/mps-128gb.json preflight
```

These tests create only small synthetic arrays/toy networks. They never load pretrained weights. The preflight checks analytic probe derivatives, exact zero-strength identity, a vanishing small-strength bound, residual preservation, spherical norm preservation, and support rejection. It deliberately exposes R0's nonzero movement at zero directional strength. Its generated world differs from the historical `toy/` sweeps and does not regenerate their numbers.

Optional checks, also not run:

```sh
PILOT_TEST_ACCELERATORS=1 python -m unittest discover -s pilot/tests -v
python -m pilot --config pilot/configs/cpu-smoke.json preflight --odesteer-src toy/third_party/odesteer/src
```

The second command requires an existing pinned upstream checkout. It imports that checkout read-only, loads the same sketch hashes/weights into our independent implementation, and compares features and the normalized Euler trajectory. It does not fetch or vendor ODESteer. With independent seeds, our NumPy-generated hashes are not expected to match upstream's Torch-generated hashes bit-for-bit.

### 3. Validate the task and collect activations

```sh
python -m pilot --config pilot/configs/mps-128gb.json readout-check
python -m pilot --config pilot/configs/mps-128gb.json collect
python -m pilot --config pilot/configs/mps-128gb.json calibrate-noop
```

Readout validation compares the six- and eight-option prompts using non-test labeled examples. It checks chromatic accuracy, answer probability mass, and gray/brown controls. All outputs have bins for six hues, gray (including grey spelling), brown and other; the six-option check counts only its six listed answers toward allowed-answer mass. If the task fails, fitting refuses to proceed. Reconsider prompt/model/readout in a new experiment directory rather than lowering criteria after test inspection.

Collection hooks the decoder outputs at the last prompt token, using left padding with explicit position IDs. All requested layers are harvested in one forward pass. It writes resumable per-batch NPZ shards. Test activations may be cached here, but downstream fitting, layer selection and strength selection do not use them.

`calibrate-noop` replays cached activations on validation prompts using both singletons and full repeated-prompt batches, compared with the mixed batches used for collection. Inspect the saved maximum answer-bin probability difference on the actual device and precision. `model.noop_tolerance` remains a provisional 0.02 and is configurable; it is never widened automatically. A failed calibration blocks steering. Diagnose batching/precision first and document any revised tolerance before evaluating steering; recalibrate after a device or batch-size change. This check is especially relevant to CUDA bf16.

### 4. Fit, inspect geometry and test causal control

```sh
python -m pilot --config pilot/configs/mps-128gb.json fit
python -m pilot --config pilot/configs/mps-128gb.json diagnostics
python -m pilot --config pilot/configs/mps-128gb.json patch
```

`fit` writes each layer/dimension/seed artifact and reports skipped configurations with insufficient samples. The probe uses degree-two polynomial count sketch, gamma 0.1, coef0 1, 8000 features and logistic C=1. Its diagnostics distinguish raw-posterior classification from class-prior-corrected density ratios. KDE bandwidth uses training-only cross-validation; support thresholds use primary validation examples. The gate requires both full-space and projected nearest-neighbor distance thresholds. Its acceptance rate on validation controls is reported, not assumed to be good.

`diagnostics` saves PCA plots colored by hue, saturation and lightness, restricted score curves, and zero/small-strength audits on real validation activations without model forwards. This is also where to inspect a gate that accepts unrelated prompts too often. There is no claim that distance thresholds alone establish general token support.

`patch` compares identity, full donor replacement and subspace donor replacement on non-target validation prompts. Donors are real training target activations selected by nearest preserved residual. Full replacement is a diagnostic control, not a low-side-effect competitor. The selection rule is smallest PCA dimension passing the predeclared full/subspace success gate, then strongest subspace success and lowest layer index. It requires all requested fitted seeds to exist. An identity-patch discrepancy above `model.noop_tolerance` prevents a configuration passing the gate; this tolerance needs the preceding first-run calibration.

If no causal configuration passes, validation steering refuses to run. Diagnose the readout/hook or reconsider the representation before expanding the search.

### 5. Validation sweep, frozen operating points, test

```sh
python -m pilot --config pilot/configs/mps-128gb.json validate
python -m pilot --config pilot/configs/mps-128gb.json report --split validation
python -m pilot --config pilot/configs/mps-128gb.json select
python -m pilot --config pilot/configs/mps-128gb.json estimate --paths
python -m pilot --config pilot/configs/mps-128gb.json validate-paths
python -m pilot --config pilot/configs/mps-128gb.json report --split validation-paths
python -m pilot --config pilot/configs/mps-128gb.json test
python -m pilot --config pilot/configs/mps-128gb.json report
```

The validation sweep uses the selected layer/dimension and every requested seed. It decodes only the initial and final states across the strength grid. After `select`, `validate-paths` decodes all checkpoints at the frozen strengths; test does the same. Endpoint-only sweeps have null path metrics, never fabricated path scores. Saved geometric states remain available throughout.

Evaluation records 90°/120°/150° and separately 180° starts, plus gray/brown/unrelated controls. **A, B and S run ungated** on the full set, as does the R0 diagnostic. The geometry methods use the declared identity fallback on rejected starts. All methods are also compared on the same eligible subset. Thus full-set results measure actual off-support baseline effects. Unrelated prompts retain full-vocabulary probabilities and report endpoint KL from their original distributions in addition to displacement and bin changes.

`select` chooses each method's smallest validation T reaching the configured success level, averaging success over fitted seeds on eligible non-antipodal non-target starts. If no T reaches it, the highest-success T is saved and explicitly marked unmatched. The selection is written to `operating-points.json`. Test evaluation refuses changed selection/settings. No method-specific retuning on test is performed.

The full profiles also collect a paraphrase template. After freezing settings, test transfer without refitting:

```sh
python -m pilot --config pilot/configs/mps-128gb.json test --template paraphrase
python -m pilot --config pilot/configs/mps-128gb.json report --template paraphrase
```

The gate was fitted at the main template; rejection on the paraphrase is an observed limitation, not a reason to calibrate it on the paraphrase test split.

## Methods implemented

| Name | Implementation |
|---|---|
| `none` | Identity reference |
| `A` | Normalized training mean difference |
| `R0` | Original strength-independent normal retraction, zero directional step; intentionally ungated diagnostic |
| `B` | Original normalized ODESteer vector-field recipe, including its prior scaling and epsilon normalization |
| `C` | Projection into PCA followed by direction normalization |
| `C-end` | C plus endpoint-only correction with total budget `correction_ratio * T` |
| `E2` | Local tangent projection plus bounded normal corrections; support exit freezes the last accepted state |
| `F` | Periodic cubic spline through six model-labeled class centroids in the same PCA subspace; shortest sampled arc, with a measured initial bridge from the source |
| `S` | Numerical norm-preservation control: exact spherical steps with the same probe; activation arc-length T, not a full GeoSteer reproduction |
| `E3` | Optional kNN graph score with local ridge regression in tangent coordinates |
| `E2-ms`, `E2-h`, `E2-full` | Optional plain-mean-shift, step-halving and bounded residual-regression variants |
| `E1`, `D` | Optional normalized additive guidance, restricted/full space, using `additive_weight` |

All deployed methods return the input exactly at zero strength; only R0 intentionally violates this. For E2, each normal correction is clipped to `correction_ratio * (T/steps) / retractions`. Thus all normal corrections together are bounded by `correction_ratio * T`, independent of step/retraction count. E2/C-end/E3 total displacement is at most `(1 + correction_ratio) * T`, before any smaller support-exit truncation. E2-full has a separate residual budget and therefore a larger total bound. Returning early at exactly zero is not the only safeguard.

E2-h is an exploratory method-specific ablation; it does not establish a fair stopping-rule comparison with B. E1/D use one configurable additive weight per run, not an automatically selected lambda sweep. Enable E3 before `fit` so its graph artifact is created. Its target set can be labels, corrected density-ratio positives, or a top-scoring fraction. Unreachable graph neighborhoods yield a zero direction rather than a fabricated finite distance.

The default pilot uses one conditional slice, one neighborhood size, one tangent dimension and one correction ratio per run. Use distinct experiment directories for targeted sensitivity studies. The implementation is standalone rather than subclassing upstream `BaseODESteer`, avoiding an install-time dependency on the unlicensed upstream repository. The optional parity test is the explicit compatibility check.

The normalized-input probe's gradient is already tangent to the activation sphere, so B and S may be almost identical at small Euler step sizes. S tests finite-step norm drift, not a distinct concept-geometry mechanism or the full GeoSteer objective/schedule. A B≈S result is expected and does not settle novelty against GeoSteer.

## Evaluation and outputs

Each unique prompt/layer/state is decoded by a fresh model forward with the last-token activation replaced, then cached across methods, strengths and fitted seeds. No KV cache is shared and no generated tokens are steered. Checkpoints sample common normalized solver time, using recorded states. An early-frozen path is padded with its final accepted state; its short prefix is never stretched over all checkpoints. C-end replaces the terminal checkpoint with its corrected endpoint without adding a time interval. Full test and selected-strength validation paths can still be expensive; model-tail caching is not implemented.

Natural-output metrics use **nearest independent reference outputs**, not a fitted behavior spline. Endpoint deviation uses the target-reference subset; path deviation uses all primary-slice reference outputs. This is a finite-sample approximation to natural behavior support, not an exact reproduction of Wurgaft's spline-based E_BC. The code reports both mean and sum over checkpoints. Reports separate endpoint and path hypotheses, gray/brown/other mass, circular concentration, support distances, displacement, norms and runtime.

`path_length` is the polyline length through accepted solver-step endpoints; `correction_length` separately sums accepted normal corrections. Neither should be interpreted as equal to nominal T. Saved paths permit further parameterization checks. Runtime separates CPU geometry from full checkpoint decoding and is not a benchmark of optimized per-token deployment.

Main artifact directories under `output`:

```text
stimuli.jsonl                  deterministic data and split identities
coverage.json                  deduplicated primary counts by split and source band
readout/                      six/eight-option validation reports
collection/                   cached layers, probabilities, resolved model identity
noop-calibration.json          observed batch/precision errors and pass/fail
fits/                         PCA, probe, KDE, gates, optional graphs
diagnostics/                  geometry/score plots and small-strength audits
patches/                      causal donor-patching results
selection.json                frozen causal site/dimension selection
evaluations/validation/        strength sweeps, raw states and output probabilities
evaluations/validation-paths/  full paths at frozen validation strengths
decoded-cache/                reusable outputs for identical prompt/layer/states
operating-points.json          frozen strengths, criteria and provenance
evaluations/test/              test results at frozen strengths
reports/                      Markdown, JSON, success/quality plots
preflight/                    synthetic diagnostics (if explicitly run)
```

NPZ files contain arrays only and are loaded with `allow_pickle=False`. Artifact fingerprints enforce settings and explicit format versions. Scoped source hashes are retained as provenance, **not enforced cache keys**: editing reporting code or comments does not invalidate activation collection. Existing manifests preserve the source provenance of their original computation. For substantive algorithm changes, explicitly use a new output directory for recomputed stages; automatic source-change detection does not prevent mixing algorithms. Do not run concurrent writers into one output directory. Atomic saves protect interruption, and collection/evaluation resume completed shards. A completed fit can be reused, although its driver may recompute a small probe while checking configurations.

Paired comparisons match prompt IDs and fitted seeds, average seeds within each prompt for prompt-bootstrap intervals, and report seed variation separately. The nominal interval alpha is adjusted across both hypotheses and B/C/C-end comparisons; S is diagnostic and excluded from this family. Bootstrap coverage remains approximate. A criterion flag requires predeclared effect size, sufficient eligible band counts, comparable validation and test success, a favorable interval, consistent seed direction, and at least three fitted seeds. Smoke results cannot satisfy that confirmatory flag. A successful flag still requires another target/template and substantive review before a broader claim.

## Scope and first-run limitations

- This is unexecuted implementation, not a claim of passing tests or working on any particular accelerator.
- The dense grid may still yield too few valid target examples or a poor manifold in the chosen model. Stages fail explicitly instead of inventing references or bypassing the gate.
- Gray/brown HSL labels are coarse readout controls; inspect their validity on the selected model.
- Optional spherical/spline methods are controlled comparators, not claimed byte-for-byte reproductions of entire published pipelines.
- Unrelated controls report full-vocabulary endpoint KL, displacement and bin changes under ungated A/B/S; six held-out unrelated prompts are a diagnostic, not an unrelated-task accuracy benchmark.
- General text generation, fluency/perplexity, exact pullback-geodesic encoders and the deferred research phases remain outside the active pilot.
