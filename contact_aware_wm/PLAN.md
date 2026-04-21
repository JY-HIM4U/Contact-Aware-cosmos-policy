# Contact-Aware Cosmos Policy: Research Plan

_Branch: `finetune`. Base: `06b6207` (F/T latent injection, by Jaeyoun)._

## 1. Motivation

State-of-the-art video world models (incl. Cosmos-Policy) predict contact-rich
manipulation poorly. They hallucinate successful grasps when the gripper has
actually dropped the object, produce frictionless sliding, and miss the
discrete on/off transition that marks contact onset. We argue this gap is
**modality-bound**, not capacity-bound: vision alone under-determines the
contact state, and the subtle visual cues (shadow, deformation,
compression) are rare and ambiguous in demonstration data. Giving the model
an explicit contact signal — force/torque (F/T) or kinematics — should let it
learn contact dynamics from far fewer examples.

## 2. Problem Statement

> Does adding physical-contact modalities (6-DoF end-effector F/T and/or
> kinematic state) to a pretrained video world model improve prediction of
> contact-rich manipulation — measured by (a) future-frame fidelity, (b)
> false-positive success rate, and (c) downstream policy success — **without
> degrading generalization to OOD visual conditions**?

## 3. Method

### 3.1 Base model
Cosmos-Policy-LIBERO-Predict2-2B. We target both heads:
- `policy_text2world_model.py` — world-model training.
- `policy_video2world_model.py` — video-conditioned rollout.

### 3.2 Two conditioning axes
We compare vision-only vs. two added contact modalities:

| Modality     | Dim | Pros                                         | Cons                                     |
|--------------|-----|----------------------------------------------|------------------------------------------|
| Kinematics   | 13  | Always available; no sensor; always in-dist | Doesn't directly observe contact forces  |
| Raw F/T      | 6   | Directly observes contact                    | Sim-to-real gap; sensor calibration drift|

Plan: **start with kinematics** (lower risk, data already in LIBERO demos),
then add F/T, then combine.

### 3.3 Injection mechanism (already implemented)
`replace_latent_with_proprio` inserts each modality as **two latent tokens**
(current_t, future_t), growing `state_t` from 9 → 11. Loss heads extended:
`future_ft_mse`, `future_ft_l1`, masked by `world_model_sample_mask` vs.
`rollout_data_mask`. See `cosmos_policy/models/policy_text2world_model.py`
(L296–675) for the injection and loss path.

### 3.4 Fine-tuning strategy
- **LoRA** on attention + MLP projections of the diffusion transformer.
  Motivation: (a) the F/T-augmented dataset is the same size as LIBERO-90,
  (b) keep the pretrained visual prior intact, (c) fits on 2× A6000.
- Init from `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`. Freeze VAE + T5.
- Training: 4× H100-equivalent hours per row; identical budget across rows.

### 3.5 Handling OOD F/T
Real hardware F/T will differ in bias, gain, and noise from MuJoCo replay.
- **Train-time**: additive Gaussian noise, σ ∝ per-dim std; random channel
  dropout (zero-one dim 10% of steps).
- **Eval-time**: three regimes — clean, noisy, zero. Zero-F/T eval reveals
  whether the model **uses** F/T or just memorizes.

### 3.6 Causal direction
Cosmos's default world head attends bidirectionally. We additionally train a
**causal variant** (triangular mask on the temporal axis) — needed for
step-by-step policy rollout where the model cannot peek at future frames.
The bidirectional variant is the "strong teacher"; the causal variant is
what a policy would actually consume.

## 4. Experiment Plan

### 4.1 Data
- **Train**: LIBERO-90, F/T replayed via MuJoCo sensors
  (`gripper0_force_ee`, `gripper0_torque_ee`) — pipeline already exists:
  `contact_aware_wm/extract_ft_libero.py`.
- **Eval-ID**: LIBERO-Spatial / Object / Goal / 10 standard suites.
- **Eval-OOD**:
  - _Semantic_: LIBERO scene variants with randomized color, lighting, texture.
  - _Distractor_: added non-interacting objects in the scene.
  - _Perturb_: small gripper pose offsets at grasp time — stresses contact
    timing and is where vision-only baselines fail most.

### 4.2 Rows (model variants)

| ID     | Video | Kinematics | F/T | Status                          |
|--------|-------|------------|-----|---------------------------------|
| V      | Y     | –          | –   | baseline (pretrained Cosmos)    |
| V+K    | Y     | Y          | –   | **todo** — add K injection      |
| V+F    | Y     | –          | Y   | current `main` / our starting pt |
| V+K+F  | Y     | Y          | Y   | full contact-aware              |

All rows: same LoRA rank, optimizer, schedule, seed set {0,1,2}.

### 4.3 Metrics (columns)

**World-model fidelity**
- Future-frame PSNR, LPIPS over the 8-frame horizon.
- Future-proprio / future-F/T MSE, L1 (already logged).
- **Contact-event recall/precision**: in simulation, extract ground-truth
  contact on/off transitions (from MuJoCo contact tensors), measure whether
  the predicted rollout contains the same transition within ±k steps
  (k ∈ {2, 4, 8}).

**Success prediction** (the motivation's key failure mode)
- **False-positive rate**: % of rollouts where the ground-truth trial
  failed but the world model predicts success.
- Success-prediction calibration (ECE).

**Downstream policy**
- World model as data augmentation / teacher → LIBERO-10 success @ 50
  episodes × 3 seeds.

**OOD robustness**
- Same metrics on each OOD split; report ID-OOD gap per row.

### 4.4 Ablations
- LoRA rank sweep ∈ {8, 16, 32}.
- Train-time F/T noise σ ∈ {0, 0.1, 0.3}; eval at each σ.
- Zero-F/T eval on V+F / V+K+F — how much does the model use F/T?
- Bidirectional vs. causal head × rollout length ∈ {8, 16, 32}.

### 4.5 Expected results (hypotheses to falsify)
1. V+F, V+K+F reduce false-positive rate vs. V by **>30% relative** on
   LIBERO-10 pick-place.
2. V+K matches V on OOD-semantic (no extra visual entanglement) and beats
   V on OOD-perturb.
3. Zero-F/T eval of V+F degrades gracefully (< 10 pp drop) — not
   catastrophic — because LoRA fine-tune preserves the visual prior.
4. Causal head is within 1-2 PSNR of bidirectional at rollout length 16,
   widening to 3-5 at length 32.

Failure modes to watch:
- F/T dominates and visual generalization collapses (V+F underperforms V on
  OOD-semantic). Mitigation: larger F/T dropout at train time.
- LoRA capacity insufficient to learn the new modality. Mitigation: rank
  sweep; unfreeze input projection; full fine-tune as fallback.

## 5. Milestones

1. **M1 — Smoke test.** `train_ft_demo.py --dry_run` passes on matrix. 100-step
   LoRA run completes end-to-end on LIBERO-90 subset.
2. **M2 — V and V+F.** Baselines fully trained. First fidelity + false-positive
   numbers on LIBERO-10.
3. **M3 — V+K, V+K+F.** Kinematic row implemented (mirror F/T injection).
   OOD splits instrumented.
4. **M4 — Ablations + write-up.** LoRA rank, F/T noise, causal head.

## 6. Open questions
- F/T normalization: min-max vs. per-dim std. Current: min-max to [-1, +1]
  (see `libero_dataset.py` L283). Worth revisiting once we see training
  loss curves.
- Do we insert current_ft at the same latent index as current_proprio or
  a dedicated slot? Current: dedicated. Check whether joint indexing helps
  cross-modality attention.
- How to handle missing F/T in eval (hardware doesn't have wrist F/T):
  zero-impute vs. learned null token.
