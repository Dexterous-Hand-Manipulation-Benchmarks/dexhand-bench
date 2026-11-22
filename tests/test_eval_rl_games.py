"""dexmachina's rl_games evaluation pairs frames like the shared evaluation (team decision 005), without a simulator."""

import pytest
import torch

eval_rl_games = pytest.importorskip(
    "dexmachina.rl.eval_rl_games", reason="needs the ppo extra (rl_games)", exc_type=ModuleNotFoundError
)

EPISODE_LENGTH = 4
RESET_POSE = -1.0


class FakeObject:
    actuated = False

    def __init__(self):
        self.root_pos = torch.zeros(1, 3)
        self.root_quat = torch.zeros(1, 4)
        self.dof_pos = torch.zeros(1, 1)

    def set_state(self, value):
        self.root_pos = torch.full((1, 3), value)
        self.root_quat = torch.full((1, 4), value)
        self.dof_pos = torch.full((1, 1), value)


class FakeRobot:
    residual_qpos = torch.zeros(EPISODE_LENGTH, 2)


class FakeBaseEnv:
    """Like BaseEnv.step: the counter advances, the state is computed inside the step, then the env resets itself."""

    def __init__(self):
        self.max_episode_length = EPISODE_LENGTH
        self.num_envs = 1
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.object = FakeObject()
        self.objects = {"box": self.object}
        self.object_names = ["box"]
        self.robots = {"left": FakeRobot(), "right": FakeRobot()}
        self.table_height = -10.0
        self.nan_envs = torch.zeros(1, dtype=torch.bool)
        self.rew_dict = {key: torch.zeros(1) for key in ("pos_dist", "rot_dist", "arti_dist")}

    def _compute_intermediate_values(self):
        pass

    def step(self):
        self.episode_length_buf += 1
        self.object.set_state(float(self.episode_length_buf[0]))  # state after step t is t + 1
        self._compute_intermediate_values()  # the scored, pre-reset state
        done = self.episode_length_buf >= self.max_episode_length
        if done.any():  # dexmachina resets inside step(): the returned state is the reset pose
            self.object.set_state(RESET_POSE)
            self.episode_length_buf[done] = 0
        return done

    def get_recorded_frames(self):
        return []


class FakeVecEnv:
    def __init__(self):
        self.unwrapped = FakeBaseEnv()

    def reset(self):
        return torch.zeros(1, 3)

    def step(self, actions):
        done = self.unwrapped.step()
        return torch.zeros(1, 3), torch.zeros(1), done.nonzero().flatten(), {}


class FakeAgent:
    is_rnn = False

    def get_batch_size(self, obs, batch_size):
        return 1

    def get_action(self, obs, is_deterministic=True):
        return torch.zeros(1, 2)


def test_states_pair_with_the_next_demo_frame_and_the_last_state_is_not_the_reset_pose():
    # demo frame k has value 10 * k in every component
    demo = torch.arange(EPISODE_LENGTH, dtype=torch.float32)[:, None].repeat(1, 8) * 10
    _, eval_data = eval_rl_games.eval_one_episode(FakeVecEnv(), FakeAgent(), demo)
    # state after step t (value t + 1) is paired with frame t + 1, the last step with the last frame again
    assert eval_data["demo_state"][:, 0].tolist() == [10.0, 20.0, 30.0, 30.0]
    assert eval_data["obj_state"][:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert RESET_POSE not in eval_data["obj_state"]
