import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings

from generator import Generator

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import torch
from torch import distributions as torchd
from torch import nn

import exploration as expl
import models
import tools
import wrappers
import random

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):

  def __init__(self, config, logger, dataset, vae_dataset):
    super(Dreamer, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)

    self._should_train_vae = tools.Every(15)
    self._should_save_vae = tools.Every(10000)
    self._should_psuedo_train = tools.Every(config.psuedo_train_every)

    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = count_steps(config.traindir)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = dataset
    self._vae_dataset = vae_dataset
    self._wm = models.WorldModel(self._step, config)
    self._wm_old = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mean
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()
    self.generator = Generator()

  def reset(self, config, logger, dataset, vae_dataset):
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_train_vae = tools.Every(15)
    self._should_save_vae = tools.Every(10000)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = count_steps(config.traindir)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = dataset
    self._vae_dataset = vae_dataset

    self._should_psuedo_train = tools.Every(config.psuedo_train_every)


  def __call__(self, obs, reset, state=None, reward=None, training=True, label=0):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]

    if training:
      train_vae = self._should_train_vae(step)

      if self._should_save_vae(step):
        save_vae_img = step
      else:
        save_vae_img = 0


    if training and self._should_train(step):

      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      if steps == self._config.pretrain:
        train_vae = False
      for _ in range(steps):
        self._train(next(self._dataset), label, train_vae=train_vae, save_vae=save_vae_img, vae_data=next(self._vae_dataset))
      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        openl = self._wm.video_pred(next(self._dataset), label=label)
        self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

      if label > 0 and self._should_psuedo_train(step):
        self._train(None, None, psuedo_train=True, train_vae=train_vae, save_vae=save_vae_img)

    policy_output, state = self._policy(obs, state, training, label)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return policy_output, state

  def _policy(self, obs, state, training, label):
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
    else:
      latent, action = state
    embed = self._wm.encoder(self._wm.preprocess(obs), label)
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample, label=label)
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)
    if not training:
      actor = self._task_behavior.actor(feat, label)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat, label)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat, label)
      action = actor.sample()
    logprob = actor.log_prob(action)
    latent = {k: v.detach()  for k, v in latent.items()}
    action = action.detach()
    if self._config.actor_dist == 'onehot_gumble':
      action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  def _train(self, data, label, psuedo_train=False, train_vae=True, save_vae=0, vae_data=None):
    metrics = {}

    # action_rand = data['action']
    if psuedo_train:
      data, label, vae_data = self.generator.generate_traj()
      
      action_tmp = data['action']
      idx = torch.randperm(action_tmp.shape[1])
      action_rand = action_tmp[:, idx].view(action_tmp.size())

    # add extra reward constrain on real data with random previous label
    if psuedo_train:
      # label_pre = random.randint(0, label-1)
      reward_pre_target = self._wm_old._forward_tmp(data, label, action_rand)
    else:
      action_rand = None
      reward_pre_target = None

    post, context, mets = self._wm._train(data, label, psuedo_train=psuedo_train, train_vae=train_vae, save_vae=save_vae, vae_data=vae_data, action_rand=action_rand, reward_pre_target=reward_pre_target)
    #print(mets)
    metrics.update(mets)
    start = post
    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}
    reward = lambda f, s, a: self._wm.heads['reward'](
        self._wm.dynamics.get_feat(s), label).mode()
    dynamics = None
    metrics.update(self._task_behavior._train(start, reward, label=label, dynamics=dynamics, psuedo_train=psuedo_train)[-1])
    #print(metrics)
    if self._config.expl_behavior != 'greedy':
      raise NotImplementedError
      if self._config.pred_discount:
        data = {k: v[:, :-1] for k, v in data.items()}
      mets = self._expl_behavior.train(start, context, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)


def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config, vae=False):
  if not vae:
    generator = tools.sample_episodes(
        episodes, config.batch_length, config.oversample_ends)
    dataset = tools.from_generator(generator, config.batch_size)
  else:
    generator = tools.sample_episodes_first(
        episodes, config.batch_length, config.oversample_ends)
    dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def make_env(config, logger, mode, train_eps, eval_eps, i):
  suite, task = config.task[i].split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, config.size,
        grayscale=config.grayscale,
        life_done=False and ('train' in mode),
        sticky_actions=True,
        all_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'dmlab':
    env = wrappers.DeepMindLabyrinth(
        task,
        mode if 'train' in mode else 'test',
        config.action_repeat)
    env = wrappers.OneHotAction(env)
  elif suite == "metaworld":
    task = "-".join(task.split("_"))
    env = wrappers.MetaWorld(
        task,
        config.seed,
        config.action_repeat,
        config.size,
        config.camera,
    )
    env = wrappers.NormalizeActions(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  if (mode == 'train') or (mode == 'eval'):
    callbacks = [functools.partial(
        process_episode, config, logger, mode, train_eps, eval_eps)]
    env = wrappers.CollectDataset(env, callbacks)
  env = wrappers.RewardObs(env)
  return env


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


def main(config):
  import copy
  temp_config = copy.deepcopy(config)
  print(temp_config.act)


  for i in range(4):
    print(i)
    print(temp_config.act)
    config = copy.deepcopy(temp_config)
    print(config.act)
    logdir = pathlib.Path(config.logdir[i]).expanduser()
    config.traindir = config.traindir or logdir / 'train_eps'
    config.evaldir = config.evaldir or logdir / 'eval_eps'
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    print(config.act)
    config.act = getattr(torch.nn, config.act)

    print('Logdir', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print('Create envs.')
    if config.offline_traindir:
      directory = config.offline_traindir.format(**vars(config))
    else:
      directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
      directory = config.offline_evaldir.format(**vars(config))
    else:
      directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, i)
    train_envs = [make('train') for _ in range(config.envs)]
    eval_envs = [make('eval') for _ in range(config.envs)]
    acts = train_envs[0].action_space
    print(acts)
    config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]


    if not config.offline_traindir:
      prefill = max(0, config.prefill - count_steps(config.traindir))
      print(f'Prefill dataset ({prefill} steps).')
      if hasattr(acts, 'discrete'):
        random_actor = tools.OneHotDist(torch.zeros_like(torch.Tensor(acts.low))[None])
      else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(torch.Tensor(acts.low)[None],
                                  torch.Tensor(acts.high)[None]), 1)
      def random_agent(o, d, s, r):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {'action': action, 'logprob': logprob}, None
      tools.simulate(random_agent, train_envs, prefill)
      tools.simulate(random_agent, eval_envs, episodes=1)
      logger.step = config.action_repeat * count_steps(config.traindir)

    print('Simulate agent.')
    train_dataset = make_dataset(train_eps, config)
    train_dataset_vae = make_dataset(train_eps, config, vae=True)
    eval_dataset = make_dataset(eval_eps, config)
    if i == 0:
      agent = Dreamer(config, logger, train_dataset, train_dataset_vae).to(config.device)

    else:
      agent.reset(config, logger, train_dataset, train_dataset_vae)
    agent.requires_grad_(requires_grad=False)

    if (logdir / 'latest_model.pt').exists():
      agent.load_state_dict(torch.load(logdir / 'latest_model.pt'))
      agent._should_pretrain._once = False
    ##
    agent._wm.curr_label = i
    policy_old = copy.deepcopy(agent._task_behavior.actor)
    agent._wm_old.load_state_dict(agent._wm.state_dict())
    agent.generator.update(agent._wm_old, policy_old, label=i-1, random_actor=random_actor)
    agent._task_behavior.value_old.load_state_dict(agent._task_behavior.value.state_dict())
    

    if i > -1:
      state = None
      while agent._step < config.steps:
        logger.write()
        print('Start evaluation.')
        video_pred = agent._wm.video_pred(next(eval_dataset), label=i)
        logger.video('eval_openl', to_np(video_pred))
        eval_policy = functools.partial(agent, training=False)
        tools.simulate(eval_policy, eval_envs, episodes=1, label=i)
        print('Start training.')
        state = tools.simulate(agent, train_envs, config.eval_every, state=state, label=i)
        if np.mod(agent._step, 100000) == 0 and agent._step !=0:
          name = str(agent._step) + '_model.pt'
          torch.save(agent.state_dict(), logdir / name)
        torch.save(agent.state_dict(), logdir / 'latest_model.pt')
        torch.save(agent._wm.vae, logdir / 'vae.pt')
    for env in train_envs + eval_envs:
      try:
        env.close()
      except Exception:
        pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  args, remaining = parser.parse_known_args()
  print(remaining)
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(parser.parse_args(remaining))
