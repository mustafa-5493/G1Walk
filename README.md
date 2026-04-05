# G1Walk

A from-scratch implementation of PPO with a Transformer policy, trained on a custom MuJoCo environment for Unitree G1 humanoid locomotion using a 3-phase curriculum.

## Demo — v4 (Latest)


https://github.com/user-attachments/assets/714bff39-4341-4c7a-b372-447189df9109

Straight forward walking with forward-facing feet. No diagonal drift.

## Version History

| Version | Mean Eval Reward | Steps | Key Changes |
|---------|-----------------|-------|-------------|
| v1 | 4827 ± 266 | 39M | Baseline — PPO + Transformer from scratch |
| v2 | 4274 ± 478 | 30M from v1 | Foot impact penalty, torso wobble -58% |
| v3 | 5463 ± 13 | 11M from v2 | Arm swing reward, tighter velocity range, correct elbow penalty |
| v4 | 4167 ± 487 | 13M from scratch | Hard hip yaw constraint (±10°), vy=0 in phases 0-1, heading + drift penalties |

### v1 — Baseline

https://github.com/user-attachments/assets/5d4e8333-5807-4569-8153-2baec54bad33

### v2 — Reduced torso wobble

https://github.com/user-attachments/assets/0e0f3b59-43dd-4990-88b8-b0b7445695ad

*Torso wobble reduced 58% (2.674 → 1.122 angular vel²). Known limitation: diagonal drift from wide lateral velocity training range.*

## Training Curve (v4)
<img width="1800" height="750" alt="training_curve_v4" src="https://github.com/user-attachments/assets/2db8c3b5-6155-40a4-b75c-912c408ba7d6" />




## What Changed Each Version

**v1 → v2:** Increased wobble penalty (-0.1 → -0.3), added foot impact penalty, added elbow resting pose penalty. Result: measurably smoother torso.

**v2 → v3:** Fixed elbow joint indices, added cross-body arm-swing reward (anti-phase coordination), tightened velocity range (vy ±0.3 → ±0.1). Result: highest raw score across all runs.

**v3 → v4:** Trained from scratch with hard XML joint limit on hip yaw (±158° → ±10°), locked vy=0 during phases 0 and 1, added lateral drift and heading penalties in phase 2. Result: forward-facing feet (guaranteed by physics, not reward), no diagonal drift.

## Architecture

**Policy:** Transformer Encoder
- 256 embed dim, 3 layers, 4 attention heads
- 8-frame observation history
- Separate actor and critic heads
- Orthogonal weight initialization
- 2.3M parameters

**Algorithm:** PPO (implemented from scratch in PyTorch)
- Generalized Advantage Estimation (GAE)
- Separate actor/critic learning rates (1e-4 actor, 1e-3 critic)
- Running mean/std observation normalization
- Linear learning rate decay over 100M steps

**Environment:** Custom MuJoCo environment for Unitree G1 (23 DOF)
- Position actuators with per-group PD gains
- 102-dimensional observation space
- Hard termination conditions (height, tilt, forbidden contact, velocity)

## Reward Function (v4)

| Component | Weight | Phase | Description |
|-----------|--------|-------|-------------|
| Velocity tracking | 1.5 / 0.5 / 2.0 | All | Track vx, vy, yaw commands |
| Upright bonus | 3.0 | All | Quaternion w² × height |
| Alternating foot contact | 2.5 | All | Encourage natural gait timing |
| Foot air time | 0.5 | All | Reward proper swing phase |
| Survival | 0.5 | All | Per-step survival bonus |
| Torso wobble penalty | -0.3 | All | Penalize angular velocity² |
| Jerkiness penalty | -0.05 | All | Penalize action delta² |
| Foot slip penalty | -0.3 | All | Penalize sliding during contact |
| Foot separation reward | 1.0 | All | Prevent narrow/crossing stance |
| Foot impact penalty | -0.1 | All | Penalize foot velocity at contact |
| Arm swing reward | 0.5 | All | Cross-body anti-phase coordination |
| Energy penalty | -0.000005 | Phase 2 | Penalize actuator force² |
| Force spike penalty | -0.01 | Phase 2 | Penalize sudden large forces |
| Lateral drift penalty | -0.5 | Phase 2 | Penalize y-position deviation |
| Heading penalty | -1.0 | Phase 2 | Penalize movement direction error |

## Curriculum (3 phases)

| Phase | Command | Advancement |
|-------|---------|-------------|
| 0 — Stand | vx=0, vy=0, yaw=0 | Mean reward > 400 |
| 1 — Slow walk | vx∈[0.3, 0.8], vy=0, yaw=0 | Mean reward > 600 |
| 2 — Variable velocity | vx∈[0.5, 1.2], vy∈[-0.15, 0.15], yaw∈[-0.3, 0.3] | Final phase |

Phases 0 and 1 lock vy=0 to establish a strong forward-walking prior before introducing lateral commands.

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
pip install mujoco gymnasium torch numpy imageio

git clone https://github.com/unitreerobotics/unitree_mujoco

python scripts/train.py

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
- PPO algorithm (clipped surrogate objective, GAE, entropy regularization)
- Transformer policy (input projection, positional embeddings, encoder, actor/critic heads)
- Running mean/std normalizer for observations and returns
- Vectorized environment wrapper (32 parallel envs)
- Custom G1 MuJoCo environment (observation space, reward function, curriculum)

MuJoCo, PyTorch, and Gymnasium are used as infrastructure only.

## Limitations & Future Work

- Arm swing reward not yet producing visible anti-phase coordination
- No push recovery
- No rough terrain
- Sim-to-real transfer not yet attempted

## References

- Schulman et al. — [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (2017)
- Schulman et al. — [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (2015)
- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- [Unitree G1](https://www.unitree.com/g1/) robot platform
