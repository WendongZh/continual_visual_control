import argparse
import pathlib
import random
import sys

import ruamel.yaml as yaml
import torch
from torch import distributions as torchd

import models
import tools
import random


class Generator():
    def __init__(self):
        self.curr_lable = 0

    def update(self, world_model, policy, label, random_actor):
        self.vae = world_model.vae
        self.wm = world_model
        self.policy = policy
        self.curr_lable = label
        self.random_actor = random_actor

    def generate_traj(self, horizon=50, bs=50, label=None):  # use label
        policy = self.policy
        dynamics = self.wm.dynamics
        random_actor = self.random_actor

        if label is None:
            label = random.randint(0, self.curr_lable)
        first_frame = self.vae.sample(label=label, batch_size=bs)
        first_frame = first_frame.permute(0, 2, 3, 1) - 0.5

        data_vae = {'image': first_frame.unsqueeze(1)}
        embed = self.wm.encoder(data_vae, label)
        init_action = torch.zeros(bs, 1, 6).to(embed.device)
        start, _ = self.wm.dynamics.observe(embed, init_action, label=label)

        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            if random.random() > 0.2:
                action = policy(inp, label).sample()
            else:
                action = random_actor.sample([bs])
                action = action.squeeze(1).to(feat.device)
            succ = dynamics.img_step(state, action, sample=True, label=label)  # hard code here
            return succ, feat, action
        feat = 0 * dynamics.get_feat(start)
        # action = policy(feat, label).mode()
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, feat, init_action))
        states = {k: torch.cat([
            start[k][None], v[:-1]], 0) for k, v in succ.items()}
        openl = self.wm.heads['image'](feats, label).mode()
        data = {}
        data['image'] = openl.detach().permute(1, 0, 2, 3, 4)
        reward = self.wm.heads['reward'](feats, label).mode()
        data['reward'] = reward.permute(1, 0, 2)
        # insert a_0 and discard a_t
        actions = actions.permute(1, 0, 2)
        action_out = torch.cat([init_action, actions.detach()[:, :-1, :]], dim=1)
        data['action'] = action_out
        return data, label, data_vae

if __name__ == "__main__":
    from dreamer_v2 import Dreamer, count_steps, make_env
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    print(remaining)
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / 'configs_v2.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining)

    logdir = pathlib.Path('debug').expanduser()
    config.traindir = config.traindir or logdir / 'train_eps'
    config.evaldir = config.evaldir or logdir / 'eval_eps'

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
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, i=2)
    train_envs = [make('train') for _ in range(config.envs)]
    eval_envs = [make('eval') for _ in range(config.envs)]
    acts = train_envs[0].action_space
    print(acts)
    config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

    vae = torch.load("vae.pt").cuda()
    world_model = models.WorldModel(5000, config).cuda()
    world_model.load_state_dict(torch.load("wm.pt"))

    policy = torch.load('policy.pt')

    data_generator = Generator()
    data_generator.update(world_model, policy, label=1)
    data_generator.generate_traj()

    print('---- Passed !!! ----')
