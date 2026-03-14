import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# ── Observation layout (102 dims) ─────────────────────────────
# torso_quat     (4)  – orientation (w,x,y,z)
# torso_angvel   (3)  – angular velocity
# torso_linvel   (3)  – linear velocity
# joint_pos      (29) – all joint positions
# joint_vel      (29) – all joint velocities
# last_action    (29) – previous policy output
# velocity_cmd   (3)  – (vx, vy, yaw)
# foot_contacts  (2)  – left/right foot ground contact
# ─────────────────────────────────────────────────────────────
OBS_DIM = 4 + 3 + 3 + 29 + 29 + 29 + 3 + 2  # = 102
ACT_DIM = 29

# Curriculum phases 
PHASE_STAND = 0
PHASE_SLOW  = 1
PHASE_FULL  = 2

# Natural standing pose — arms hang naturally
DEFAULT_POS = np.zeros(ACT_DIM, dtype=np.float32)
# slight knee bend for stability
DEFAULT_POS[3]  = 0.1    # left knee
DEFAULT_POS[9]  = 0.1    # right knee


class G1Env(gym.Env):
    """
    Custom Gymnasium environment for Unitree G1 velocity tracking.

    Design decisions by Walter:
    - Position actuators with per-group PD gains (defined in XML)
    - Full observation: body state + joints + last action + cmd + foot contacts
    - Reward: alternating gait + upright + velocity tracking + penalties
    - Termination: fall + tilt + arms up + forbidden contact + high vel + narrow stance
    - Curriculum: 3 phases with reward threshold advancement
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, xml_path, phase=PHASE_STAND,
                 render_mode=None, ctrl_dt=0.02, sim_steps=5):
        super().__init__()
        self.xml_path    = xml_path
        self.phase       = phase
        self.render_mode = render_mode
        self.ctrl_dt     = ctrl_dt
        self.sim_steps   = sim_steps

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = ctrl_dt / sim_steps

        # spaces
        obs_high = np.inf * np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(
            -obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            -np.ones(ACT_DIM, dtype=np.float32),
             np.ones(ACT_DIM, dtype=np.float32))

        # find foot and forbidden body ids
        self._find_body_ids()

        # episode state
        self.step_count         = 0
        self.last_action        = np.zeros(ACT_DIM, dtype=np.float32)
        self.cmd                = np.zeros(3, dtype=np.float32)
        self.prev_foot_contacts = np.zeros(2, dtype=np.float32)
        self.foot_air_time      = np.zeros(2, dtype=np.float32)
        self._update_max_steps()

        # renderer
        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(
                self.model, height=480, width=640)

    def _find_body_ids(self):
        self.left_foot_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        self.right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')
        forbidden_names = [
            'left_knee_link', 'right_knee_link',
            'left_elbow_link', 'right_elbow_link',
            'left_wrist_roll_link', 'right_wrist_roll_link',
        ]
        self.forbidden_body_ids = []
        for name in forbidden_names:
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.forbidden_body_ids.append(bid)

    def _update_max_steps(self):
        self.max_steps = {
            PHASE_STAND: 500,
            PHASE_SLOW:  750,
            PHASE_FULL:  1000,
        }[self.phase]

    def _sample_command(self):
        if self.phase == PHASE_STAND:
            return np.zeros(3, dtype=np.float32)
        elif self.phase == PHASE_SLOW:
            return np.array([0.5, 0.0, 0.0], dtype=np.float32)
        else:
            vx  = np.random.uniform(0.3, 1.5)
            vy  = np.random.uniform(-0.3, 0.3)
            yaw = np.random.uniform(-0.5, 0.5)
            return np.array([vx, vy, yaw], dtype=np.float32)

    def _get_foot_contacts(self):
        left, right = 0.0, 0.0
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if self.left_foot_id  in (b1, b2): left  = 1.0
            if self.right_foot_id in (b1, b2): right = 1.0
        return np.array([left, right], dtype=np.float32)

    def _check_forbidden_contact(self):
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            for fid in self.forbidden_body_ids:
                if fid in (b1, b2):
                    return True
        return False

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        foot_contacts = self._get_foot_contacts()
        return np.concatenate([
            qpos[3:7],         # torso quaternion (w,x,y,z)
            qvel[3:6],         # torso angular velocity
            qvel[0:3],         # torso linear velocity
            qpos[7:],          # joint positions (29)
            qvel[6:],          # joint velocities (29)
            self.last_action,  # last action (29)
            self.cmd,          # velocity command (3)
            foot_contacts,     # foot contacts (2)
        ]).astype(np.float32)

    def _get_reward(self, action, foot_contacts):
        qpos = self.data.qpos
        qvel = self.data.qvel
        vx_cmd, vy_cmd, yaw_cmd = self.cmd

        # 1. velocity tracking
        vx_rew  = 1.5 * np.exp(-3.0 * (qvel[0] - vx_cmd)**2)
        vy_rew  = 0.5 * np.exp(-4.0 * (qvel[1] - vy_cmd)**2)
        yaw_rew = 2.0 * np.exp(-3.0 * (qvel[5] - yaw_cmd)**2)

        # 2. upright — high weight (your design)
        w       = qpos[3]
        upright = 3.0 * (w**2) * np.clip(qpos[2] - 0.5, 0.0, 1.0)

        # 3. alternating foot contact — high weight (your design)
        left, right = foot_contacts
        alternating = 2.0 * float(left != right)
        for i, (prev, curr) in enumerate(
                zip(self.prev_foot_contacts, foot_contacts)):
            if prev == 1.0 and curr == 0.0:
                self.foot_air_time[i] = 0.0
            elif curr == 0.0:
                self.foot_air_time[i] += self.ctrl_dt
        air_time_rew = 0.5 * np.sum(np.clip(self.foot_air_time, 0.0, 0.5))

        # 4. energy penalty — high weight (your design)
        energy_pen = -0.0005 * np.sum(self.data.actuator_force**2)

        # 5. jerkiness penalty — high weight (your design)
        jerk_pen = -0.05 * np.sum((action - self.last_action)**2)

        # 6. torso wobble penalty — increased to reduce lateral swing (your design)
        wobble_pen = -0.3 * (qvel[3]**2 + qvel[4]**2)

        # 7. arm flailing penalty — medium weight (your design)
        arm_pen = -0.0001 * np.sum(qvel[6+15:]**2)

        # 8. foot slip penalty — high weight (your design)
        slip_pen = 0.0
        if left > 0.5 or right > 0.5:
            slip_pen = -0.3 * np.linalg.norm(qvel[0:2])

        # 9. foot separation reward (your design — prevent narrow stance)
        lf = self.data.xpos[self.left_foot_id]
        rf = self.data.xpos[self.right_foot_id]
        sep     = np.abs(lf[1] - rf[1])
        sep_rew = 1.0 * np.clip(sep - 0.15, -0.15, 0.2)

        # 10. foot impact penalty — force spike + foot velocity at contact (your design)
        impact_pen = 0.0
        for i, (prev, curr) in enumerate(zip(self.prev_foot_contacts, foot_contacts)):
            if prev == 0.0 and curr == 1.0:  # foot just made contact
                foot_id = self.left_foot_id if i == 0 else self.right_foot_id
                foot_vel = self.data.cvel[foot_id][3:6]  # translational velocity
                impact_pen += -0.1 * abs(foot_vel[2])   # vertical velocity at impact
        # force spike: penalize sudden large actuator forces
        force_spike = -0.1 * np.mean(np.clip(np.abs(self.data.actuator_force) - 50, 0, None))

        # 11. elbow resting pose penalty (your design: -0.25 x elbow_angle^2)
        left_elbow_angle  = qpos[7 + 18]   # left elbow joint
        right_elbow_angle = qpos[7 + 25]   # right elbow joint
        elbow_pen = -0.25 * (left_elbow_angle**2 + right_elbow_angle**2)

        # 12. survival
        survival = 0.5

        return float(
            vx_rew + vy_rew + yaw_rew +
            upright + alternating + air_time_rew +
            energy_pen + jerk_pen + wobble_pen +
            arm_pen + slip_pen + sep_rew +
            impact_pen + force_spike + elbow_pen + survival
        )

    def _is_terminated(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        # fallen
        if qpos[2] < 0.3:
            return True
        # too tilted (your design: 45 degrees, cos45=0.707)
        if abs(qpos[3]) < 0.7:
            return True
        # arms pointing up (your design)
        jp = qpos[7:]
        if jp[15] > 1.5 or jp[22] > 1.5:
            return True
        # forbidden body contact
        if self._check_forbidden_contact():
            return True
        # velocity too high (your design)
        if np.linalg.norm(qvel[0:3]) > 5.0:
            return True
        # feet too close (your design)
        lf = self.data.xpos[self.left_foot_id]
        rf = self.data.xpos[self.right_foot_id]
        if np.abs(lf[1] - rf[1]) < 0.05:
            return True

        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # set default standing pose
        self.data.qpos[7:] = DEFAULT_POS.copy()
        # small perturbation for robustness
        self.data.qpos[7:] += np.random.uniform(-0.02, 0.02, ACT_DIM)
        mujoco.mj_forward(self.model, self.data)

        self.step_count         = 0
        self.last_action        = DEFAULT_POS.copy()
        self.prev_foot_contacts = np.zeros(2, dtype=np.float32)
        self.foot_air_time      = np.zeros(2, dtype=np.float32)
        self.cmd                = self._sample_command()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # action = target joint position offset from default
        # scale to ±0.5 rad range
        target = DEFAULT_POS + 0.5 * action

        for _ in range(self.sim_steps):
            self.data.ctrl[:] = target
            mujoco.mj_step(self.model, self.data)

        foot_contacts           = self._get_foot_contacts()
        obs                     = self._get_obs()
        reward                  = self._get_reward(action, foot_contacts)
        self.prev_foot_contacts = foot_contacts.copy()
        self.last_action        = action.copy()
        self.step_count        += 1

        terminated = self._is_terminated()
        truncated  = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.data, camera="track")
        return self.renderer.render()

    def set_phase(self, phase):
        self.phase = phase
        self._update_max_steps()

    def close(self):
        if self.renderer:
            self.renderer.close()
