"""dexmachina's assisted-physics curriculum: the minimum spacing between gain reductions is counted in epochs.

Wrappers map training progress onto curriculum epochs (``epoch = step * Emax // total_steps``) and call
``set_curriculum`` only when the epoch advances (REPPO, FastTD3) or once per iteration (rl_games). Depending on
the batch layout, consecutive calls can therefore skip epochs or repeat one. The spacing must not depend on that.
"""
import pytest

curriculum = pytest.importorskip("dexmachina.envs.curriculum", reason="needs dexmachina")

MIN_SPACING = 40  # hard-coded in Curriculum.determine_decay
WAIT_EPOCHS = 100
DEQUE_LEN = 3
EPISODE_LENGTH = 200
LAST_EPOCH = 300


class _StubObject:
    device = "cpu"

    def set_joint_gains(self, **kwargs):
        pass


def _make_curriculum():
    cfg = curriculum.get_curriculum_cfg(dict(
        schedule="uniform", uniform_mode="slow", fixed_mode="uniform", wait_epochs=WAIT_EPOCHS,
        deque_len=DEQUE_LEN, skip_grad=True, rew_thresholds=dict(task=0.6), zero_epoch=10**6,
        upper_ratios=dict(kp=0.9, kv=0.9, fr=1.0), lower_ratios=dict(kp=0.8, kv=0.8, fr=1.0),
    ))
    return curriculum.Curriculum(cfg, _StubObject(), ["task"], num_envs=2, achieved_length=EPISODE_LENGTH,
                                 max_episode_length=EPISODE_LENGTH)


def _decay_epochs(call_epochs):
    """Epochs at which the gains were reduced, with every performance gate passing on every call."""
    curr = _make_curriculum()
    decays = []
    for epoch in call_epochs:
        for _ in range(DEQUE_LEN):  # a decay clears the deques; keep them full and above threshold
            curr.update_progress({"task": 1.0}, EPISODE_LENGTH)
        _, decayed, _ = curr.set_curriculum(epoch)
        if decayed:
            decays.append(epoch)
    return decays


# First reduction once the warm-up is over, then one every MIN_SPACING epochs (all multiples of 2).
EXPECTED = list(range(WAIT_EPOCHS, LAST_EPOCH + 1, MIN_SPACING))


def test_once_per_epoch_keeps_rl_games_timing():
    assert _decay_epochs(range(LAST_EPOCH + 1)) == EXPECTED


def test_skipped_epochs_still_count():
    # e.g. REPPO: ~6,100 calls for 10,000 epochs, so the epoch advances by more than one per call
    assert _decay_epochs(range(0, LAST_EPOCH + 1, 2)) == EXPECTED


def test_repeated_epochs_do_not_count():
    # a wrapper calling several times per epoch must not shorten the spacing
    calls = [epoch for epoch in range(LAST_EPOCH + 1) for _ in range(5)]
    assert _decay_epochs(calls) == EXPECTED
