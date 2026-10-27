import torch


class FakeDexEnv:
    """Minimal stand-in for dexmachina's BaseEnv (``use_rl_games=True`` API).

    Mirrors the parts of BaseEnv.step that the wrappers rely on: on a time-out
    both ``terminated`` and ``truncated`` are True, and ``reset_idx`` runs
    inside ``step`` before the (post-reset) observation is returned.

    The observation of env ``i`` after ``t`` steps is ``t + 100 * i`` in every
    dimension, so pre- and post-reset observations are easy to tell apart.
    """

    def __init__(self, num_envs=2, obs_dim=3, action_dim=2, episode_length=3):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_clip = 1.0
        self.max_episode_length = episode_length
        self.t = torch.zeros(num_envs)
        self.rew_dict = {}
        self.received_actions = None
        self.dist = {"pos_dist": 0.01, "rot_dist": 0.1, "arti_dist": 0.05}

    def get_observations(self):
        offsets = 100.0 * torch.arange(self.num_envs, dtype=torch.float32)
        return {"policy": (self.t + offsets)[:, None].repeat(1, self.obs_dim)}

    def reset(self):
        self.t.zero_()
        return self.get_observations(), {}

    def reset_idx(self, env_idxs):
        self.t[env_idxs] = 0

    def step(self, actions):
        self.received_actions = actions.clone()
        self.t += 1
        timeout = self.t >= self.max_episode_length
        self.rew_dict = {k: torch.full((self.num_envs,), v) for k, v in self.dist.items()}
        reset_ids = timeout.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.reset_idx(reset_ids)
        return self.get_observations(), torch.ones(self.num_envs), timeout.clone(), timeout.clone(), {}
