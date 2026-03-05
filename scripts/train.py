import sys
sys.path.insert(0, '/workspace/G1Walk')

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from collections import deque
import os

from env.g1_env import G1Env, PHASE_STAND, PHASE_SLOW, PHASE_FULL
from policy.transformer_policy import TransformerPolicy

# ── Config ────────────────────────────────────────────────────
XML_PATH    = '/workspace/unitree_mujoco/unitree_robots/g1/scene_23dof_pos.xml'
OBS_DIM     = 102
ACT_DIM     = 29
HISTORY_LEN = 8       # Walter's design
N_ENVS      = 32      # Walter's design
N_STEPS     = 2048
BATCH_SIZE  = 512
N_EPOCHS    = 2
CLIP_RANGE  = 0.1
ENT_COEF    = 0.001
ACTOR_LR    = 1e-4
CRITIC_LR   = 1e-3
MAX_GRAD    = 0.5
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
TOTAL_STEPS = 100_000_000   # Walter's design
DEVICE      = 'cuda'

# curriculum thresholds — master each phase (Walter's design)
PHASE_THRESHOLDS = {
    PHASE_STAND: 400,
    PHASE_SLOW:  600,
}

# ── Running Mean Std ──────────────────────────────────────────
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.atleast_2d(x)
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        delta     = b_mean - self.mean
        tot       = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        M2        = (self.var * self.count + b_var * b_count +
                     delta**2 * self.count * b_count / tot)
        self.var   = M2 / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                       -clip, clip).astype(np.float32)

# ── Vectorized Env ────────────────────────────────────────────
class VecG1Env:
    def __init__(self, xml_path, n_envs, phase):
        self.envs   = [G1Env(xml_path, phase=phase) for _ in range(n_envs)]
        self.n_envs = n_envs

    def reset(self):
        return np.stack([e.reset()[0] for e in self.envs])

    def step(self, actions):
        results  = [e.step(a) for e, a in zip(self.envs, actions)]
        obs, rew, term, trunc, _ = zip(*results)
        done = np.array(term) | np.array(trunc)
        obs  = list(obs)
        for i, d in enumerate(done):
            if d:
                obs[i], _ = self.envs[i].reset()
        return np.stack(obs), np.array(rew, dtype=np.float32), done

    def set_phase(self, phase):
        for e in self.envs:
            e.set_phase(phase)

    def close(self):
        for e in self.envs: e.close()

# ── Main ──────────────────────────────────────────────────────
print(f'G1Walk | PPO + Transformer | {N_ENVS} envs | '
      f'{TOTAL_STEPS//1_000_000}M steps')

env    = VecG1Env(XML_PATH, N_ENVS, PHASE_STAND)
policy = TransformerPolicy(OBS_DIM, ACT_DIM, HISTORY_LEN).to(DEVICE)
obs_rms= RunningMeanStd(shape=(OBS_DIM,))
ret_rms= RunningMeanStd(shape=(1,))

total_params = sum(p.numel() for p in policy.parameters())
print(f'Policy params: {total_params:,}\n')

actor_params  = (list(policy.transformer.parameters()) +
                 list(policy.input_proj.parameters()) +
                 list(policy.actor_head.parameters()) +
                [policy.pos_emb, policy.log_std])
critic_params = list(policy.critic.parameters())

actor_opt  = Adam(actor_params,  lr=ACTOR_LR,  eps=1e-5)
critic_opt = Adam(critic_params, lr=CRITIC_LR, eps=1e-5)

# history buffers per env
obs_histories = [deque(
    [np.zeros(OBS_DIM, dtype=np.float32)] * HISTORY_LEN,
    maxlen=HISTORY_LEN) for _ in range(N_ENVS)]

# rollout buffers — store histories, not raw obs
hist_buf = torch.zeros(N_STEPS, N_ENVS, HISTORY_LEN, OBS_DIM).to(DEVICE)
act_buf  = torch.zeros(N_STEPS, N_ENVS, ACT_DIM).to(DEVICE)
lp_buf   = torch.zeros(N_STEPS, N_ENVS).to(DEVICE)
rew_buf  = torch.zeros(N_STEPS, N_ENVS).to(DEVICE)
val_buf  = torch.zeros(N_STEPS, N_ENVS).to(DEVICE)
done_buf = torch.zeros(N_STEPS, N_ENVS).to(DEVICE)

os.makedirs('/workspace/G1Walk/checkpoints', exist_ok=True)
os.makedirs('/workspace/G1Walk/logs', exist_ok=True)
log_f = open('/workspace/G1Walk/logs/train.csv', 'w')
log_f.write('steps,iteration,phase,mean_reward,vf_loss,entropy,kl\n')

# init
raw_obs = env.reset()
obs_rms.update(raw_obs)
for i in range(N_ENVS):
    norm = obs_rms.normalize(raw_obs[i])
    obs_histories[i].append(norm)

total_steps   = 0
iteration     = 0
ep_rewards    = []
curr_rewards  = np.zeros(N_ENVS)
best_reward   = -np.inf
current_phase = PHASE_STAND

def get_history_tensor():
    arr = np.stack([np.stack(list(h)) for h in obs_histories])
    return torch.FloatTensor(arr).to(DEVICE)

print(f"{'Iter':>5} | {'Steps':>12} | {'Phase':>5} | "
      f"{'Reward':>9} | {'VF Loss':>8} | {'Entropy':>8} | {'KL':>7}")
print('-' * 72)

while total_steps < TOTAL_STEPS:

    # linear lr decay (Walter's design)
    frac = max(1.0 - total_steps / TOTAL_STEPS, 1e-2)
    for pg in actor_opt.param_groups:  pg['lr'] = ACTOR_LR  * frac
    for pg in critic_opt.param_groups: pg['lr'] = CRITIC_LR * frac

    # ── collect rollout ──
    for step in range(N_STEPS):
        hist_t = get_history_tensor()
        with torch.no_grad():
            action, log_prob, value = policy.get_action(hist_t)

        action_np = action.cpu().numpy().clip(-1.0, 1.0)
        next_raw, reward, done = env.step(action_np)

        obs_rms.update(next_raw)
        for i in range(N_ENVS):
            norm = obs_rms.normalize(next_raw[i])
            obs_histories[i].append(norm)
            if done[i]:
                ep_rewards.append(curr_rewards[i])
                curr_rewards[i] = 0
                # reset history on episode end
                obs_histories[i] = deque(
                    [np.zeros(OBS_DIM, dtype=np.float32)] * HISTORY_LEN,
                    maxlen=HISTORY_LEN)
                norm2 = obs_rms.normalize(next_raw[i])
                obs_histories[i].append(norm2)

        curr_rewards += reward

        hist_buf[step]  = hist_t
        act_buf[step]   = action
        lp_buf[step]    = log_prob
        # normalize reward by running std
        ret_rms.update(reward.reshape(-1, 1))
        norm_reward = reward / (np.sqrt(ret_rms.var[0]) + 1e-8)
        rew_buf[step]   = torch.FloatTensor(norm_reward).to(DEVICE)
        val_buf[step]   = value
        done_buf[step]  = torch.FloatTensor(done).to(DEVICE)
        total_steps    += N_ENVS

    # ── GAE ──
    with torch.no_grad():
        _, _, last_val = policy.get_action(get_history_tensor())

    adv = torch.zeros_like(rew_buf)
    gae = 0.0
    for t in reversed(range(N_STEPS)):
        nv = last_val if t == N_STEPS-1 else val_buf[t+1]
        nt = 1.0 - (done_buf[t] if t == N_STEPS-1 else done_buf[t+1])
        delta = rew_buf[t] + GAMMA * nv * nt - val_buf[t]
        gae   = delta + GAMMA * GAE_LAMBDA * nt * gae
        adv[t]= gae

    ret = (adv + val_buf).detach()
    adv = adv.detach()
    T, N = N_STEPS, N_ENVS
    hb  = hist_buf.view(T*N, HISTORY_LEN, OBS_DIM)
    ab  = act_buf.view(T*N, ACT_DIM)
    lpb = lp_buf.view(T*N)
    rb  = ret.view(T*N)
    adb = adv.view(T*N)
    adb = (adb - adb.mean()) / (adb.std() + 1e-8)

    # ── PPO update ──
    vf_losses, entropies, kls = [], [], []
    for _ in range(N_EPOCHS):
        idx = torch.randperm(T * N)
        for start in range(0, T*N, BATCH_SIZE):
            b = idx[start:start+BATCH_SIZE]
            new_lp, ent, val = policy.evaluate(hb[b], ab[b])
            entropy = ent.mean()
            ratio   = (new_lp - lpb[b]).exp()
            pl = -torch.min(
                ratio * adb[b],
                torch.clamp(ratio, 1-CLIP_RANGE, 1+CLIP_RANGE) * adb[b]
            ).mean() - ENT_COEF * entropy
            actor_opt.zero_grad()
            pl.backward()
            nn.utils.clip_grad_norm_(actor_params, MAX_GRAD)
            actor_opt.step()

            vl = nn.functional.mse_loss(val, rb[b])
            critic_opt.zero_grad()
            vl.backward()
            nn.utils.clip_grad_norm_(critic_params, MAX_GRAD)
            critic_opt.step()

            with torch.no_grad():
                kl = (lpb[b] - new_lp).mean()
            vf_losses.append(vl.item())
            entropies.append(entropy.item())
            kls.append(kl.item())

    iteration   += 1
    mean_reward  = np.mean(ep_rewards[-100:]) if ep_rewards else 0.0
    mean_vl      = np.mean(vf_losses)
    mean_ent     = np.mean(entropies)
    mean_kl      = np.mean(kls)

    print(f"{iteration:>5} | {total_steps:>12,} | {current_phase:>5} | "
          f"{mean_reward:>9.1f} | {mean_vl:>8.3f} | "
          f"{mean_ent:>8.3f} | {mean_kl:>7.4f}")

    log_f.write(f"{total_steps},{iteration},{current_phase},"
                f"{mean_reward:.2f},{mean_vl:.4f},"
                f"{mean_ent:.4f},{mean_kl:.4f}\n")
    log_f.flush()

    # ── curriculum advancement ──
    if current_phase in PHASE_THRESHOLDS:
        if (mean_reward > PHASE_THRESHOLDS[current_phase]
                and len(ep_rewards) >= 100):
            current_phase += 1
            env.set_phase(current_phase)
            print(f'  → CURRICULUM: phase {current_phase}')

    # ── checkpointing every 1M steps ──
    if total_steps % 1_000_000 < N_ENVS * N_STEPS:
        ckpt = {
            'policy':        policy.state_dict(),
            'obs_rms_mean':  obs_rms.mean,
            'obs_rms_var':   obs_rms.var,
            'phase':         current_phase,
            'timesteps':     total_steps,
            'iteration':     iteration,
        }
        path = f'/workspace/G1Walk/checkpoints/ckpt_{total_steps//1_000_000}M.pt'
        torch.save(ckpt, path)

    if mean_reward > best_reward and ep_rewards:
        best_reward = mean_reward
        torch.save({
            'policy':       policy.state_dict(),
            'obs_rms_mean': obs_rms.mean,
            'obs_rms_var':  obs_rms.var,
            'phase':        current_phase,
        }, '/workspace/G1Walk/checkpoints/best.pt')

env.close()
log_f.close()
print('\nDone.')
