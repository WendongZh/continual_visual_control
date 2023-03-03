import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch import nn
from torch.optim import Adam

import networks
import tools
from vae import VAE
import cv2
import random

to_np = lambda x: x.detach().cpu().numpy()


class WorldModel(nn.Module):

  def __init__(self, step, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self.encoder = networks.ConvEncoder(config.grayscale,
        config.cnn_depth, config.act, config.encoder_kernels)
    if config.size[0] == 64 and config.size[1] == 64:
      embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      embed_size *= 2 * 2
    else:
      raise NotImplemented(f"{config.size} is not applicable now")
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, embed_size, config.device)
    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    shape = (channels,) + config.size
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    self.heads['image'] = networks.ConvDecoder(
        feat_size,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    self.heads['reward'] = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.discount_layers, config.units, config.act, dist='binary')
    for name in config.grad_heads:
      assert name in self.heads, name
    self._model_opt = tools.Optimizer(
        'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt,
        use_amp=self._use_amp)
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale)
    self.curr_label = 0
    self.vae = VAE()
    self.save_pre_img = [[], [], [], []] # save ground truth image for vae training

  def _train(self, data, label, psuedo_train=False, train_vae=True, save_vae=0, vae_data=None, action_rand=None, reward_pre_target=None): # use label
    if not psuedo_train:
      data = self.preprocess(data)
      vae_data = self.preprocess(vae_data)


    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        embed = self.encoder(data, label)
        post, prior = self.dynamics.observe(embed, data['action'], label=label)
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        losses = {}
        likes = {}
        # train the rest
        for name, head in self.heads.items():
          grad_head = (name in self._config.grad_heads)
          feat = self.dynamics.get_feat(post)
          feat = feat if grad_head else feat.detach()
          pred = head(feat, label)
          like = pred.log_prob(data[name])
          likes[name] = like
          losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

        # add extra constrain on reward head
        if psuedo_train:
          with torch.no_grad():
            embed_pre = self.encoder(data, label)
            post_pre, _ = self.dynamics.observe(embed_pre, action_rand, label=label)
            feat_pre = self.dynamics.get_feat(post_pre)
          
          reward_pre = self.heads['reward'](feat_pre, label)
          like_pre = reward_pre.log_prob(reward_pre_target)
          likes['reward_pre'] = like_pre
          losses['reward_pre'] = -torch.mean(like_pre) * self._scales.get('reward', 1.0) * 0.5


        model_loss = sum(losses.values()) + kl_loss

              # if not psuedo_train:
      if train_vae:
        # train vae
        start_index = random.randint(0, 2)
        if not psuedo_train:
          vae_img = vae_data['image'][:, start_index*5]
          vae_img = vae_img.reshape(-1, *data['image'].shape[-3:]) + 0.5

          #save ground truth image
          if len(self.save_pre_img[label]) < 4 and np.random.random() < 0.01:
            self.save_pre_img[label].append(vae_img)

        else:
          vae_img = vae_data['image'][:, 0]
          vae_img = vae_img.reshape(-1, *data['image'].shape[-3:]) + 0.5

          used_pre_img = random.choice(self.save_pre_img[label])
          selected_index = random.randint(0, 49)
          vae_img[selected_index] = used_pre_img[selected_index]

        bs = 50
        for i in range(vae_img.shape[0] // bs):
            # print(vae_img[i*bs: (i+1) * bs].shape, vae_img[i*bs: (i+1) * bs].min(), vae_img[i*bs: (i+1) * bs].max())
            loss_recon, loss_kl = self.vae.train(vae_img[i*bs: (i+1) * bs], label=label)
            if save_vae != 0:
              print('step: %d, loss_recon: %f, loss_kl: %f' % (save_vae, loss_recon, loss_kl))

        if save_vae != 0:
          if not psuedo_train:
            flag = 'real'
          else:
            flag = 'replay'
          self._save_vae_img(save_vae, label, vae_img, flag)
      metrics = self._model_opt(model_loss, self.parameters())

    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = to_np(torch.mean(kl_value))
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
      context = dict(
          embed=embed, feat=self.dynamics.get_feat(post),
          kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics

  def _forward_tmp(self, data, label, action_rand): # use label , psuedo_train=False

    with torch.no_grad():
      embed = self.encoder(data, label)
      post, prior = self.dynamics.observe(embed, action_rand, label=label)

      feat = self.dynamics.get_feat(post)
      # feat = feat if grad_head else feat.detach()
      pred = self.heads['reward'](feat, label)
      

    return pred.mode()

  def _train_vae(self, data, label, psuedo_train=False): # use label
    if not psuedo_train:
      data = self.preprocess(data)

    with tools.RequiresGrad(self):
      # with torch.cuda.amp.autocast(self._use_amp):
      if not psuedo_train:
        # train vae
        vae_img = data['image'].reshape(-1, *data['image'].shape[-3:]) + 0.5
        bs = 64
        for i in range(vae_img.shape[0] // bs):
          # print(vae_img[i*bs: (i+1) * bs].shape, vae_img[i*bs: (i+1) * bs].min(), vae_img[i*bs: (i+1) * bs].max())
          loss_recon, loss_kl = self.vae.train(vae_img[i*bs: (i+1) * bs], label=self.curr_label)

    return loss_recon, loss_kl

  def _save_vae_img(self, step, label, img_gt, flag='real'):
    for j in range(label+1):
      img = self.vae.sample(j, 3)
      tmp = img.detach().cpu()

      for i in range(3):
          # tmp = img[i]
          img = tmp[i].permute(1,2,0)
          img = img.numpy()*255
          img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          cv2.imwrite('./save_img/' + flag + '_test_label_' + str(j) + '_' + str(i) + '_step' + str(step) +'.png', img)

    img_gt = img_gt.detach().cpu().numpy()
    for i in range(5):
      name_gt = './save_img/' + flag + '_gt_' + str(label) + '_' + str(step) + '_' + str(i) + '.png'
      gt_tmp = np.uint8(img_gt[i]*255)
      gt_tmp = cv2.cvtColor(gt_tmp, cv2.COLOR_RGB2BGR)
      cv2.imwrite(name_gt, gt_tmp)

  def preprocess(self, obs):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
    if self._config.clip_rewards == 'tanh':
      obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
    elif self._config.clip_rewards == 'identity':
      obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs

  def video_pred(self, data, label): # use label
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data, label)

    states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5], label=label)
    recon = self.heads['image'](
        self.dynamics.get_feat(states), label).mode()[:6]
    reward_post = self.heads['reward'](
        self.dynamics.get_feat(states), label).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics.imagine(data['action'][:6, 5:], init, label=label)
    openl = self.heads['image'](self.dynamics.get_feat(prior), label).mode()
    reward_prior = self.heads['reward'](self.dynamics.get_feat(prior), label).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    self.actor = networks.ActionHead(
        feat_size,  # pytorch version
        config.num_actions, config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)

    # add old value function
    self.value_old = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)

    self.value = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    if config.slow_value_target or config.slow_actor_target:
      self._slow_value = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.value_layers, config.units, config.act)
      self._updates = 0
    kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
    self._actor_opt = tools.Optimizer(
        'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    self._value_opt = tools.Optimizer(
        'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)

  def _train(
      self, start, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None, label=None, dynamics=None, psuedo_train=False):  # use label
    objective = objective or self._reward
    self._update_slow_target()
    metrics = {}

    with tools.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(self._use_amp):
        imag_feat, imag_state, imag_action = self._imagine(
            start, self.actor, self._config.imag_horizon, repeats, label=label, dynamics=dynamics)
        reward = objective(imag_feat, imag_state, imag_action)
        actor_ent = self.actor(imag_feat, label).entropy()
        state_ent = self._world_model.dynamics.get_dist(
            imag_state).entropy()
        target, weights = self._compute_target(
            imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
            self._config.slow_actor_target, label=label)
        actor_loss, mets = self._compute_actor_loss(
            imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
            weights, label=label)
        metrics.update(mets)
        if self._config.slow_value_target != self._config.slow_actor_target:
          target, weights = self._compute_target(
              imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
              self._config.slow_value_target, label=label)
        value_input = imag_feat

    with tools.RequiresGrad(self.value):
      with torch.cuda.amp.autocast(self._use_amp):
        value = self.value(value_input[:-1].detach(), label)
        target = torch.stack(target, dim=1)
        value_loss = -value.log_prob(target.detach())

        # add value regularization
        if psuedo_train:
          value_old = self.value_old(value_input[:-1].detach(), label).mode()
          mask_tmp = ((value_old.detach() - target)>0).float()
          target = target * mask_tmp + (1 - mask_tmp) * value_old
          value_loss = -value.log_prob(target.detach())

        if self._config.value_decay:
          value_loss += self._config.value_decay * value.mode()
        value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])

    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))
    metrics['actor_ent'] = to_np(torch.mean(actor_ent))
    with tools.RequiresGrad(self):
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      metrics.update(self._value_opt(value_loss, self.value.parameters()))
    return imag_feat, imag_state, imag_action, weights, metrics

  def _imagine(self, start, policy, horizon, repeats=None, label=None, dynamics=None):  # use label
    if dynamics is None:
      dynamics = self._world_model.dynamics
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _ = prev
      feat = dynamics.get_feat(state)
      inp = feat.detach() if self._stop_grad_actor else feat
      action = policy(inp, label).sample()
      succ = dynamics.img_step(state, action, sample=self._config.imag_sample, label=label)
      return succ, feat, action
    feat = 0 * dynamics.get_feat(start)
    action = policy(feat, label).mode()
    succ, feats, actions = tools.static_scan(
        step, [torch.arange(horizon)], (start, feat, action))
    states = {k: torch.cat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")

    return feats, states, actions

  def _compute_target(
      self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
      slow, label): # use label
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(reward)
    if self._config.future_entropy and self._config.actor_entropy() > 0:
      reward += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and self._config.actor_state_entropy() > 0:
      reward += self._config.actor_state_entropy() * state_ent
    if slow:
      value = self._slow_value(imag_feat, label).mode()
    else:
      value = self.value(imag_feat, label).mode()
    target = tools.lambda_return(
        reward[:-1], value[:-1], discount[:-1],
        bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target, weights

  def _compute_actor_loss(
      self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
      weights, label): # use label
    metrics = {}
    inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
    policy = self.actor(inp, label)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)
    if self._config.imag_gradient == 'dynamics':
      actor_target = target
    elif self._config.imag_gradient == 'reinforce':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode(), label).detach()
    elif self._config.imag_gradient == 'both':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode(), label).detach()
      mix = self._config.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix
    else:
      raise NotImplementedError(self._config.imag_gradient)
    if not self._config.future_entropy and (self._config.actor_entropy() > 0):
      actor_target += self._config.actor_entropy() * actor_ent[:-1][:,:,None]
    if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
      actor_target += self._config.actor_state_entropy() * state_ent[:-1]
    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1
