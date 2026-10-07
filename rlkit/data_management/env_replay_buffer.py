import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from gym.spaces import Box, Discrete, Tuple

try:
    from gymnasium.spaces import Box as GymnasiumBox
    from gymnasium.spaces import Discrete as GymnasiumDiscrete
    from gymnasium.spaces import Dict as GymnasiumDict
    from gymnasium.spaces import Tuple as GymnasiumTuple
except ImportError:
    GymnasiumBox = None
    GymnasiumDiscrete = None
    GymnasiumDict = None
    GymnasiumTuple = None


BOX_SPACE_TYPES = (Box,) + ((GymnasiumBox,) if GymnasiumBox is not None else ())
DISCRETE_SPACE_TYPES = (Discrete,) + ((GymnasiumDiscrete,) if GymnasiumDiscrete is not None else ())
TUPLE_SPACE_TYPES = (Tuple,) + ((GymnasiumTuple,) if GymnasiumTuple is not None else ())
DICT_SPACE_TYPES = ((GymnasiumDict,) if GymnasiumDict is not None else ())


class MultiTaskReplayBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            tasks,
            goal_radius,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = dict([(idx, SimpleReplayBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            goal_radius=goal_radius,
        )) for idx in tasks])


    def add_sample(self, task, observation, action, reward, terminal,
            next_observation, **kwargs):
        if isinstance(self._action_space, DISCRETE_SPACE_TYPES):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(
                observation, action, reward, terminal,
                next_observation, **kwargs)

    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task].random_batch(batch_size)
        return batch
    
    def random_seq_batch(self, task, batch_size, seq_length):
        return self.task_buffers[task].random_sequence_batch(batch_size, seq_length)

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path):
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()


def get_dim(space):
    if isinstance(space, BOX_SPACE_TYPES):
        return int(np.prod(space.shape))
    elif isinstance(space, DISCRETE_SPACE_TYPES):
        return int(space.n)
    elif isinstance(space, TUPLE_SPACE_TYPES):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif DICT_SPACE_TYPES and isinstance(space, DICT_SPACE_TYPES):
        return sum(get_dim(subspace) for subspace in space.spaces.values())
    elif hasattr(space, 'flat_dim'):
        return int(space.flat_dim)
    else:
        # import OldBox here so it is not necessary to have rand_param_envs 
        # installed if not running the rand_param envs
        try:
            from rand_param_envs.gym.spaces.box import Box as OldBox
        except ImportError:
            OldBox = ()

        if isinstance(space, OldBox):
            return int(np.prod(space.shape))

        raise TypeError("Unknown space: {}".format(space))
