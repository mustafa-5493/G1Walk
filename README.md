# G1Walk

A from-scratch implementation of PPO with a Transformer policy, trained on a custom MuJoCo environment for Unitree G1 humanoid locomotion using a 3-phase curriculum.

## Versions

| Version | Torso Wobble (angular vel²) | Elbow Angle | Notes |
|---------|----------------------------|-------------|-------|
| v1 | 2.674 | 0.156 | Baseline — 39M steps |
| v2 | 1.122 (-58%) | 0.163 | Foot impact penalty, elbow penalty, increased wobble penalty — 30M steps from v1 checkpoint |

## Demo

**v1 — Baseline**

https://github.com/user-attachments/assets/5d4e8333-5807-4569-8153-2baec54bad33

**v2 — Reduced torso wobble, emergent arm-leg coordination**



https://github.com/user-attachments/assets/0e0f3b59-43dd-4990-88b8-b0b7445695ad



*Known limitation in v2: diagonal drift artifact from wide lateral velocity training range. To be fixed in v3.*

## Results

| Metric | Value |
|--------|-------|
| Mean evaluation reward | 4827 ± 266 |
| Best episode reward | 5428 |
| Episode length | 1000 steps (never falls) |
| Training steps | 39M (v1) + 30M (v2) |
| Training time | ~18 hours on T4 GPU |

## Training Curve
<img width="1800" height="750" alt="training_curve" src="https://github.com/user-attachments/assets/f131f4fe-ebe0-40c2-95a7-5f56e1803228" />

## Architecture

**Policy:** Transformer Encoder
- 256 embed dim, 3 layers, 4 attention heads
- 8-frame observation history
- Separate actor and critic heads
- Orthogonal weight initialization

**Algorithm:** PPO (implemented from scratch in PyTorch)
- Generalized Advantage Estimation (GAE)
- Separate actor/critic optimizers
- Running mean/std observation and reward normalization
- Linear learning rate decay

**Environment:** Custom MuJoCo environment for Unitree G1 (23 DOF)
- Position actuators with per-group PD gains (legs, ankles, waist, arms)
- 102-dimensional observation space: torso state + joint positions/velocities + last action + velocity command + foot contacts
- Hard termination conditions

## Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Velocity tracking | 1.5 / 0.5 / 2.0 | Track vx, vy, yaw commands (exponential kernel) |
| Upright bonus | 3.0 | Quaternion w² × height |
| Alternating foot contact | 2.0 | Encourage natural gait timing |
| Foot air time | 0.5 | Reward proper swing phase |
| Survival | 0.5 | Per-step survival bonus |
| Energy penalty | -0.0005 | Penalize torque² |
| Jerkiness penalty | -0.05 | Penalize action delta² |
| Torso wobble penalty | -0.1 → -0.3 (v2) | Penalize angular velocity² |
| Arm flailing penalty | -0.0001 | Penalize arm joint velocity² |
| Foot slip penalty | -0.3 | Penalize horizontal velocity during contact |
| Foot separation reward | 1.0 | Prevent narrow/crossing stance |
| Foot impact penalty | -0.1 (v2) | Penalize foot velocity at contact + force spikes |
| Elbow resting pose penalty | -0.25 (v2) | Penalize elbow deviation from neutral |

## Curriculum (3 phases)

| Phase | Command | Advancement threshold |
|-------|---------|----------------------|
| 0 — Stand | vx=0 | Mean reward > 400 |
| 1 — Slow walk | vx=0.5 m/s | Mean reward > 600 |
| 2 — Variable velocity | vx∈[0.3, 1.5], vy∈[-0.3, 0.3], yaw∈[-0.5, 0.5] | Final phase |

Phase advancement occurred at:
- Phase 0 → 1: 5.9M steps
- Phase 1 → 2: 6.4M steps

## Observation Space (102 dims)

| Component | Dims |
|-----------|------|
| Torso quaternion (w,x,y,z) | 4 |
| Torso angular velocity | 3 |
| Torso linear velocity | 3 |
| Joint positions (29 joints) | 29 |
| Joint velocities (29 joints) | 29 |
| Last action | 29 |
| Velocity command (vx, vy, yaw) | 3 |
| Foot contacts (left, right) | 2 |

## Setup
```bash
# Requirements
pip install mujoco gymnasium torch numpy imageio

# Clone Unitree MuJoCo models
git clone https://github.com/unitreerobotics/unitree_mujoco

# Train
python scripts/train.py

# Evaluate
MUJOCO_GL=egl python scripts/evaluate.py
```

## Project Structure
```
G1Walk/
├── env/
│   └── g1_env.py              # Custom MuJoCo environment
├── policy/
│   └── transformer_policy.py  # Transformer Encoder policy
├── scripts/
│   ├── train.py               # PPO training loop
│   └── evaluate.py            # Evaluation + video recording
└── logs/
    └── train.csv              # Training curves
```

## What's From Scratch

Every core component was implemented without RL libraries:
- PPO algorithm (clipped surrogate, GAE, separate optimizers)
- Transformer policy (input projection, positional embeddings, encoder)
- Running mean/std normalizer
- Vectorized environment wrapper
- Custom G1 MuJoCo environment (observation space, reward, curriculum)

MuJoCo, PyTorch, and Gymnasium are used as infrastructure tools only.

## Limitations & Future Work

- v2 diagonal drift: lateral velocity training range will be constrained in v3
- No push recovery yet
- No rough terrain
- Personal motion imitation (v3): fine-tune on developer's own movement data via MediaPipe pose estimation
- Sim-to-real transfer not yet attempted

## References

- Schulman et al. — [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (2017)
- Schulman et al. — [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (2015)
- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- [Unitree G1](https://www.unitree.com/g1/) robot platform
