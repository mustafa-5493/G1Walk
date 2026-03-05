import sys
sys.path.insert(0, '/workspace/G1Walk')

import torch
import numpy as np
import mujoco
import imageio
from collections import deque

from env.g1_env import G1Env, PHASE_FULL
from policy.transformer_policy import TransformerPolicy

# ── Config ────────────────────────────────────────────────────
XML_PATH   = '/workspace/unitree_mujoco/unitree_robots/g1/scene_23dof_pos.xml'
CKPT_PATH  = '/workspace/G1Walk/checkpoints/best.pt'
VIDEO_PATH = '/workspace/G1Walk/g1_demo.mp4'
OBS_DIM    = 102
ACT_DIM    = 29
HISTORY_LEN= 8
DEVICE     = 'cuda'
N_EPISODES = 5
FPS        = 50

def normalize(obs, mean, var):
    return np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)

# load checkpoint
ckpt   = torch.load(CKPT_PATH, weights_only=False)
policy = TransformerPolicy(OBS_DIM, ACT_DIM, HISTORY_LEN).to(DEVICE)
policy.load_state_dict(ckpt['policy'])
policy.eval()
obs_mean = ckpt['obs_rms_mean']
obs_var  = ckpt['obs_rms_var']
print('Checkpoint loaded. Phase:', ckpt['phase'])

# run episodes
env = G1Env(XML_PATH, phase=PHASE_FULL, render_mode='rgb_array')
episode_rewards = []
frames = []
record_episode = 0  # record first episode

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    obs_history = deque(
        [np.zeros(OBS_DIM, dtype=np.float32)] * HISTORY_LEN,
        maxlen=HISTORY_LEN)

    total_reward = 0
    ep_frames = []

    for step in range(1000):
        norm_obs = normalize(obs, obs_mean, obs_var)
        obs_history.append(norm_obs)

        hist_t = torch.FloatTensor(
            np.stack(list(obs_history))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mean, std, _ = policy(hist_t)
            action = mean.cpu().numpy()[0]  # deterministic

        action = np.clip(action, -1.0, 1.0)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if ep == record_episode:
            frame = env.render()
            if frame is not None:
                ep_frames.append(frame)

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)
    print(f'Episode {ep+1}: reward={total_reward:.1f} steps={step+1}')

    if ep == record_episode:
        frames = ep_frames

env.close()

print(f'\\nMean reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}')
print(f'Best episode: {max(episode_rewards):.1f}')

# save video
print(f'\\nSaving video: {VIDEO_PATH}')
imageio.mimsave(VIDEO_PATH, frames, fps=FPS)
print(f'Done. {len(frames)} frames saved.')
