import threading

import gym
import numpy as np


class DeepMindLabyrinth(object):

  ACTION_SET_DEFAULT = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (0, 0, 0, -1, 0, 0, 0),   # Backward
      (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
      (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
      (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
      (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
      (0, 0, 0, 0, 1, 0, 0),    # Fire
  )

  ACTION_SET_MEDIUM = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (0, 0, 0, -1, 0, 0, 0),   # Backward
      (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
      (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
      (0, 0, 0, 0, 0, 0, 0),    # Idle.
  )

  ACTION_SET_SMALL = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
  )

  def __init__(
      self, level, mode, action_repeat=4, render_size=(64, 64),
      action_set=ACTION_SET_DEFAULT, level_cache=None, seed=None,
      runfiles_path=None):
    assert mode in ('train', 'test')
    import deepmind_lab
    if runfiles_path:
      print('Setting DMLab runfiles path:', runfiles_path)
      deepmind_lab.set_runfiles_path(runfiles_path)
    self._config = {}
    self._config['width'] = render_size[0]
    self._config['height'] = render_size[1]
    self._config['logLevel'] = 'WARN'
    if mode == 'test':
      self._config['allowHoldOutLevels'] = 'true'
      self._config['mixerSeed'] = 0x600D5EED
    self._action_repeat = action_repeat
    self._random = np.random.RandomState(seed)
    self._env = deepmind_lab.Lab(
        level='contributed/dmlab30/'+level,
        observations=['RGB_INTERLEAVED'],
        config={k: str(v) for k, v in self._config.items()},
        level_cache=level_cache)
    self._action_set = action_set
    self._last_image = None
    self._done = True

  @property
  def observation_space(self):
    shape = (self._config['height'], self._config['width'], 3)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return gym.spaces.Discrete(len(self._action_set))

  def reset(self):
    self._done = False
    self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
    obs = self._get_obs()
    return obs

  def step(self, action):
    raw_action = np.array(self._action_set[action], np.intc)
    reward = self._env.step(raw_action, num_steps=self._action_repeat)
    self._done = not self._env.is_running()
    obs = self._get_obs()
    return obs, reward, self._done, {}

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._last_image

  def close(self):
    self._env.close()

  def _get_obs(self):
    if self._done:
      image = 0 * self._last_image
    else:
      image = self._env.observations()['RGB_INTERLEAVED']
    self._last_image = image
    return {'image': image}



class DeepMindControl:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky_actions else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
    return gym.spaces.Dict({
        'image': self._env.observation_space,
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    })

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs

  def step(self, action):
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)


class CollectDataset:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    if isinstance(action, dict):
      transition.update(action)
    else:
      transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      for key, value in self._episode[1].items():
        if key not in self._episode[0]:
          self._episode[0][key] = 0 * value
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      # print(info['success'])
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    # Missing keys will be filled with a zeroed out version of the first
    # transition, because we do not know what action information the agent will
    # pass yet.
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)

class MetaWorld:
    def __init__(self, name, seed=None, action_repeat=1, size=(64, 64), camera=None):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )
        import os
        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat

        self._camera = camera

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        return spaces

    @property
    def action_space(self):
        action = self._env.action_space
        return action

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        done = None
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action)
            success += float(info["success"])
            reward += rew or 0.0
            if done:
              break
        success = min(success, 1.0)
        assert success in [0.0, 1.0]

        obs = {
            "reward": reward,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=3
            ),
            "state": state,
        }
        info['success'] = success
        return obs, reward, done, info

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera, device_id=3
            ),
            "state": state,
        }
        return obs
        
class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)

class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.action_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.action_space, self._key: space}

  def step(self, action):
    print(action)
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})

class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.discrete = True
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class SelectAction:

  def __init__(self, env, key):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    return self._env.step(action[self._key])
