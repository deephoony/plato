"""PLATO: CSRO base (FOCAL + CLUB + BRAC) plus the three PLATO modules
LORE, SIMULATE rollout, and pessimism-weighted policy improvement. With all
PLATO flags off (use_decoder / use_simulate / use_lore = False), the agent
reduces to CSRO."""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch import nn as nn
from collections import OrderedDict

import rlkit.torch.pytorch_util as ptu
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.torch.brac import divergences
from rlkit.torch.brac import utils


class PLATO(OfflineMetaRLAlgorithm):

    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            extreme_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs,
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            extreme_tasks=extreme_tasks,
            goal_radius=goal_radius,
            **kwargs,
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        self.recurrent                      = kwargs['recurrent']
        self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.use_brac                       = kwargs['use_brac']
        self.use_value_penalty              = kwargs['use_value_penalty']
        self.alpha_max                      = kwargs['alpha_max']
        self._c_iter                        = kwargs['c_iter']
        self.train_alpha                    = kwargs['train_alpha']
        self._target_divergence             = kwargs['target_divergence']
        self.alpha_init                     = kwargs['alpha_init']
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']
        self.z_loss_weight                  = kwargs['z_loss_weight']
        self.max_entropy                    = kwargs['max_entropy']
        self.allow_backward_z               = kwargs['allow_backward_z']

        self.use_FOCAL_cl                   = kwargs.get('use_FOCAL_cl', True)
        self.use_club                       = kwargs.get('use_club', True)
        self.club_loss_weight               = kwargs.get('club_loss_weight', 25.0)
        self.club_model_loss_weight         = kwargs.get('club_model_loss_weight', 10.0)

        self.use_decoder                    = kwargs.get('use_decoder',         False)
        self.use_simulate                   = kwargs.get('use_simulate',        False)
        self.use_lore                       = kwargs.get('use_lore',            False)
        self.use_adaptive_alpha             = kwargs.get('use_adaptive_alpha',  False)

        # Ablation flags. lore_random_dir replaces the centroid-outward LORE
        # direction with a random unit vector (no-direction ablation);
        # q_use_synthetic adds an extra critic step on the imagined Bellman
        # target. Both default to False so the main PLATO algorithm is
        # unaffected.
        self.lore_random_dir                = bool(kwargs.get('lore_random_dir', False))
        self.q_use_synthetic                = bool(kwargs.get('q_use_synthetic', False))

        if self.use_decoder:
            self.recon_loss_weight     = kwargs.get('recon_loss_weight', 1.0)
            self.reward_recon_scale    = kwargs.get('reward_recon_scale', 1.0)
            self.recon_detach_z        = kwargs.get('recon_detach_z', True)
            self.num_ensemble          = kwargs.get('num_ensemble', 10)
            self.use_bootstrap         = kwargs.get('use_bootstrap', True)
            self.bootstrap_ratio       = kwargs.get('bootstrap_ratio', 0.8)
            self.decoder_lr            = kwargs.get('decoder_lr', 3e-4)

        if self.use_simulate:
            assert self.use_decoder, "use_simulate requires use_decoder=True"
            assert self.use_lore, "use_simulate requires use_lore=True"
            self.rollout_horizon       = int(kwargs.get('rollout_horizon', 3))
            self.rollout_weight        = kwargs.get('rollout_weight', 0.3)
            self.lambda_pess           = kwargs.get('lambda_pess', 2.5)
            self.simulate_warmup_steps = int(kwargs.get('simulate_warmup_steps', 10000))
            self.rollout_state_clip_K  = float(kwargs.get('rollout_state_clip_K', 10.0))

        if self.use_lore:
            self.lore_alpha            = float(kwargs.get('lore_alpha', 0.5))
            self.lore_p_exponent       = float(kwargs.get('lore_p_exponent', 2.0))

        if self.use_adaptive_alpha:
            assert self.use_decoder, "use_adaptive_alpha requires use_decoder"
            self.alpha_base            = float(kwargs.get('alpha_base', 1.0))
            self.eta_u                 = float(kwargs.get('eta_u', 0.5))
            self.u_ema_tau             = float(kwargs.get('u_ema_tau', 0.99))
            self.u_warmup_steps        = int(kwargs.get('u_warmup_steps', 5000))
            self.u_clip                = float(kwargs.get('u_clip', 5.0))
            self._u_mean = 0.0
            self._u_var  = 1.0

        self.policy_grad_clip = float(kwargs.get('policy_grad_clip', 0.0))

        self.loss = {}
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_criterion  = nn.MSELoss()
        self.vf_criterion  = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.club_criterion   = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()

        # nets = [agent, qf1, qf2, vf, c, club_model, decoder_ensemble?]
        self.qf1, self.qf2, self.vf, self.c, self.club_model = nets[1:6]
        self.target_vf = self.vf.copy()
        if self.use_decoder:
            self.decoder_ensemble = nets[6]
        else:
            self.decoder_ensemble = None

        self.policy_optimizer     = optimizer_class(self.agent.policy.parameters(),         lr=self.policy_lr)
        self.qf1_optimizer        = optimizer_class(self.qf1.parameters(),                  lr=self.qf_lr)
        self.qf2_optimizer        = optimizer_class(self.qf2.parameters(),                  lr=self.qf_lr)
        self.vf_optimizer         = optimizer_class(self.vf.parameters(),                   lr=self.vf_lr)
        self.c_optimizer          = optimizer_class(self.c.parameters(),                    lr=self.c_lr)
        self.context_optimizer    = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.club_model_optimizer = optimizer_class(self.club_model.parameters(),           lr=self.context_lr)
        if self.use_decoder:
            self.decoder_optimizer = optimizer_class(self.decoder_ensemble.parameters(),    lr=self.decoder_lr)

        if self.use_decoder:
            normal_idx = self.n_tasks // 2 - 1
            size = self.replay_buffer.task_buffers[normal_idx].size()
            obss = self.replay_buffer.task_buffers[normal_idx]._observations[:size]
            rewards = self.replay_buffer.task_buffers[normal_idx]._rewards[:size]
            self.mu_state   = ptu.FloatTensor(np.mean(obss, axis=0))
            self.std_state  = ptu.FloatTensor(np.std(obss, axis=0) + 1e-6)
            self.mu_reward  = ptu.FloatTensor(np.mean(rewards, 0))
            self.std_reward = ptu.FloatTensor(np.std(rewards, 0) + 1e-6)

        self._divergence    = None
        self._num_steps     = 0
        self._visit_num_steps_train = 10
        self._alpha_var     = torch.tensor(1.)

    @property
    def networks(self):
        nets = self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf,
                                       self.c, self.club_model]
        if self.use_decoder:
            nets = nets + [self.decoder_ensemble]
        return nets

    @property
    def get_alpha(self):
        return utils.clip_v2(self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        if self.use_decoder:
            self.mu_state   = self.mu_state.to(device)
            self.std_state  = self.std_state.to(device)
            self.mu_reward  = self.mu_reward.to(device)
            self.std_reward = self.std_reward.to(device)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def unpack_batch(self, batch, sparse_reward=False):
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
                   for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent),
        ) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size
        context_batch = self.sample_context(indices)
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst, z_vars_lst = [], []
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        policy_outputs, task_z, _ = self.agent(obs, context, task_indices=indices)
        new_actions = policy_outputs[0]
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def FOCAL_z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b
            for j in range(i + 1, len(indices)):
                idx_j = j * b
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1 / (torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss / (pos_cnt + epsilon) + neg_z_loss / (neg_cnt + epsilon)

    def _build_policy_input(self, obs_flat, z):
        return torch.cat([obs_flat, z], dim=1)

    def _compute_z_aug_lore(self, t, b, z_per_sample):
        """LORE perturbation:
            c       = mean(z_tasks),  d_i = ||z_i - c||
            n_i     = (z_i - c) / d_i,   w_i = (d_i / mean_d) ** p
            z_aug_i = z_i + alpha * std_d * w_i * n_i

        With lore_random_dir=True (no-direction ablation), n_i is a random
        unit vector instead of the centroid-outward direction.
        """
        d = z_per_sample.size(-1)
        z_tb = z_per_sample.detach().view(t, b, d)
        z_tasks = z_tb[:, 0, :]
        c = z_tasks.mean(dim=0, keepdim=True)
        deltas = z_tasks - c
        distances = deltas.norm(dim=-1, keepdim=True) + 1e-8
        if self.lore_random_dir:
            rand = torch.randn_like(deltas)
            n_unit = rand / (rand.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            n_unit = deltas / distances
        mean_d = distances.mean()
        std_d = distances.std() + 1e-6
        weights = (distances / (mean_d + 1e-8)) ** self.lore_p_exponent
        z_aug_tasks = z_tasks + self.lore_alpha * std_d * weights * n_unit

        with torch.no_grad():
            perturb_mag = (z_aug_tasks - z_tasks).norm(dim=-1).mean()
            z_norm_mean = z_tasks.norm(dim=-1).mean()
            self.loss['lore_mean_d']        = float(mean_d.item())
            self.loss['lore_std_d']         = float(std_d.item())
            self.loss['lore_perturb_mag']   = float(perturb_mag.item())
            self.loss['lore_perturb_ratio'] = float((perturb_mag / (z_norm_mean + 1e-8)).item())
            self.loss['lore_weight_max']    = float(weights.max().item())

        z_aug = z_aug_tasks.unsqueeze(1).expand(-1, b, -1).reshape(t * b, d)
        return z_aug.detach()

    def _simulate_rollout(self, t, b, s0, z_adv, obs_dim):
        """Roll out the decoder ensemble for H steps starting from s0 under
        z_adv. Returns list of {s, a, r, s_next, u} per step (the per-step
        ensemble disagreement u is what enters the pessimism weight)."""
        rollout = []
        s = s0
        for h in range(self.rollout_horizon):
            with torch.no_grad():
                in_ = self._build_policy_input(s, z_adv)
                policy_out = self.agent.policy(t, b, in_, reparameterize=False, return_log_prob=True)
                a = policy_out[0]
                pred = self.decoder_ensemble(t, b, z_adv, s, a)
                pred_mean = pred.mean(dim=0)
                u = pred.var(dim=0).mean(dim=-1, keepdim=True)
                delta_s = pred_mean[:, :obs_dim] * self.std_state
                r_synth = pred_mean[:, obs_dim:] * self.std_reward + self.mu_reward
                s_next = s + delta_s
                if self.rollout_state_clip_K > 0:
                    K = self.rollout_state_clip_K
                    s_normed = (s_next - self.mu_state) / (self.std_state + 1e-6)
                    s_normed = s_normed.clamp(min=-K, max=K)
                    s_next = s_normed * self.std_state + self.mu_state
            rollout.append({
                's':      s.detach(),
                'a':      a.detach(),
                'r':      r_synth.detach(),
                's_next': s_next.detach(),
                'u':      u.detach(),
            })
            s = s_next.detach()
        return rollout

    def _ema_update_and_normalise(self, u_scalar):
        u_mean = float(u_scalar.mean().item())
        u_var  = float(u_scalar.var().item() + 1e-8)
        self._u_mean = self.u_ema_tau * self._u_mean + (1 - self.u_ema_tau) * u_mean
        self._u_var  = self.u_ema_tau * self._u_var  + (1 - self.u_ema_tau) * u_var
        normalised = (u_scalar - self._u_mean) / (np.sqrt(self._u_var) + 1e-6)
        return F.relu(normalised).clamp(max=self.u_clip)

    def _take_step(self, indices, context):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        t, b, _ = obs.size()
        obs_flat      = obs.view(t * b, -1)
        actions_flat  = actions.view(t * b, -1)
        next_obs_flat = next_obs.view(t * b, -1)
        rewards_flat  = rewards.view(self.batch_size * num_tasks, -1)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs_flat, actions_flat, task_z)
            q2_pred = self.qf2(t, b, obs_flat, actions_flat, task_z)
            v_pred  = self.vf(t, b, obs_flat, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs_flat, actions_flat, task_z.detach())
            q2_pred = self.qf2(t, b, obs_flat, actions_flat, task_z.detach())
            v_pred  = self.vf(t, b, obs_flat, task_z.detach())

        c_loss = self._divergence.dual_critic_loss(obs_flat, new_actions.detach(), actions_flat, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()

        div_estimate = self._divergence.dual_estimate(obs_flat, new_actions, actions_flat, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs_flat, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs_flat, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        if self.use_club:
            self.club_model_optimizer.zero_grad()
            z_target = self.agent.encode_no_mean(context).detach()
            z_param  = self.club_model(context[..., :self.club_model.input_size])
            z_mean   = z_param[..., :self.latent_dim]
            z_var    = F.softplus(z_param[..., self.latent_dim:])
            club_model_loss = self.club_model_loss_weight * (
                ((z_target - z_mean) ** 2 / (2 * z_var)) + torch.log(torch.sqrt(z_var))
            ).mean()
            club_model_loss.backward()
            self.loss["club_model_loss"] = club_model_loss.item()
            self.club_model_optimizer.step()

        self.context_optimizer.zero_grad()
        if self.use_decoder:
            self.decoder_optimizer.zero_grad()

        if self.use_club:
            z_target = self.agent.encode_no_mean(context)
            z_param  = self.club_model(context[..., :self.club_model.input_size]).detach()
            z_mean   = z_param[..., :self.latent_dim]
            z_var    = F.softplus(z_param[..., self.latent_dim:])
            z_t, z_b, _ = z_mean.size()
            position = -((z_target - z_mean) ** 2 / z_var).mean()
            z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b ** 2, -1)
            z_var_expand  = z_var[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b ** 2, -1)
            z_target_repeat = z_target.repeat(1, z_b, 1)
            negative = -((z_target_repeat - z_mean_expand) ** 2 / z_var_expand).mean()
            club_loss = self.club_loss_weight * (position - negative)
            club_loss.backward(retain_graph=True)
            self.loss["club_loss"] = club_loss.item()

        if self.use_FOCAL_cl:
            z_loss = self.z_loss_weight * self.FOCAL_z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()

        # Decoder ensemble training
        if self.use_decoder:
            z_for_recon = task_z.detach() if self.recon_detach_z else task_z
            decoder_pred = self.decoder_ensemble(t, b, z_for_recon, obs_flat, actions_flat)
            target = torch.cat([
                (next_obs_flat - obs_flat) / self.std_state,
                (rewards_flat - self.mu_reward) / self.std_reward,
            ], dim=-1)
            target_expanded = target.unsqueeze(0).expand(self.num_ensemble, -1, -1)
            if self.use_bootstrap:
                per_element_mse = (decoder_pred - target_expanded) ** 2
                if self.reward_recon_scale != 1.0:
                    dim_weight = torch.ones(1, 1, per_element_mse.size(-1), device=ptu.device)
                    dim_weight[:, :, obs_dim:] = self.reward_recon_scale
                    per_element_mse = per_element_mse * dim_weight
                mask = (torch.rand(self.num_ensemble, t * b, 1, device=ptu.device) < self.bootstrap_ratio).float()
                recon_loss = self.recon_loss_weight * (per_element_mse * mask).sum() \
                             / (mask.sum() * per_element_mse.size(-1) + 1e-8)
            else:
                recon_loss = self.recon_loss_weight * self.mse_criterion(decoder_pred, target_expanded)
            recon_loss.backward(retain_graph=True)
            self.loss["recon_loss"] = recon_loss.item()
            self.decoder_optimizer.step()

        self.context_optimizer.step()

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_scaled = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_scaled + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # SIMULATE rollout under LORE-perturbed z
        z_det = task_z.detach()
        simulate_active = (
            self.use_simulate
            and self._num_steps >= self.simulate_warmup_steps
        )
        if simulate_active:
            z_adv = self._compute_z_aug_lore(t, b, z_det)
            rollout = self._simulate_rollout(t, b, obs_flat, z_adv, obs_dim)
            u_first = rollout[0]['u']
            self.loss['u_raw'] = float(u_first.mean().item())
        else:
            z_adv = z_det
            rollout = []
            u_first = None

        if self.use_adaptive_alpha:
            if u_first is None:
                with torch.no_grad():
                    real_pred = self.decoder_ensemble(t, b, z_det, obs_flat, actions_flat)
                    u_first = real_pred.var(dim=0).mean(dim=-1, keepdim=True)
            u_tilde = self._ema_update_and_normalise(u_first)
            if self._num_steps < self.u_warmup_steps:
                gate_u = torch.zeros_like(u_tilde)
            else:
                gate_u = u_tilde
            alpha_effective = (self.alpha_base * (1.0 + self.eta_u * gate_u)).detach()
            self.loss['u_raw']    = float(u_first.mean().item())
            self.loss['u_tilde']  = float(u_tilde.mean().item())
            self.loss['alpha_eff']= float(alpha_effective.mean().item())
        else:
            alpha_effective = None

        min_q_new_actions = torch.min(
            self.qf1(t, b, obs_flat, new_actions, task_z.detach()),
            self.qf2(t, b, obs_flat, new_actions, task_z.detach()),
        )
        if self.max_entropy:
            if alpha_effective is not None:
                v_target = min_q_new_actions - alpha_effective * log_pi.view(-1, 1)
            else:
                v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()

        log_policy_target = min_q_new_actions
        if self.use_brac:
            if self.max_entropy:
                if alpha_effective is not None:
                    policy_loss = (alpha_effective * log_pi.view(-1, 1)
                                   - log_policy_target
                                   + self.get_alpha.detach() * div_estimate.view(-1, 1)).mean()
                else:
                    policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                if alpha_effective is not None:
                    policy_loss = (alpha_effective * log_pi.view(-1, 1) - log_policy_target).mean()
                else:
                    policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = -log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        # Pessimism-weighted policy improvement on the imagined rollout
        if simulate_active and len(rollout) > 0:
            synth_policy_loss = 0.0
            for step_data in rollout:
                s_h = step_data['s']
                u_h = step_data['u']
                in_h = self._build_policy_input(s_h, z_adv)
                policy_out_h = self.agent.policy(t, b, in_h, reparameterize=True, return_log_prob=True)
                new_a_h    = policy_out_h[0]
                log_pi_h   = policy_out_h[3]
                q1_h = self.qf1(t, b, s_h, new_a_h, z_adv)
                q2_h = self.qf2(t, b, s_h, new_a_h, z_adv)
                min_q_h = torch.min(q1_h, q2_h)
                weight_h = 1.0 / (1.0 + self.lambda_pess * u_h.detach())
                if alpha_effective is not None:
                    synth_term = weight_h * (alpha_effective * log_pi_h.view(-1, 1) - min_q_h)
                else:
                    synth_term = weight_h * (log_pi_h.view(-1, 1) - min_q_h)
                synth_policy_loss = synth_policy_loss + synth_term.mean()
            synth_policy_loss = synth_policy_loss * (self.rollout_weight / max(1, self.rollout_horizon))
            if torch.isfinite(synth_policy_loss).all():
                policy_loss = policy_loss + synth_policy_loss
                self.loss['synth_policy_loss'] = float(synth_policy_loss.item())
                self.loss['synth_nan_skip'] = 0.0
            else:
                self.loss['synth_nan_skip'] = 1.0
        else:
            self.loss['synth_policy_loss'] = 0.0

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        if self.policy_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), max_norm=self.policy_grad_clip)
        self.policy_optimizer.step()
        self.loss["policy_loss"] = policy_loss.item()

        # Q-synth ablation: extra critic step on the imagined Bellman target.
        if self.q_use_synthetic and simulate_active and len(rollout) > 0:
            qf_synth_loss = 0.0
            for step_data in rollout:
                s_h, a_h, r_h, s_next_h = (
                    step_data['s'], step_data['a'], step_data['r'], step_data['s_next']
                )
                with torch.no_grad():
                    q_target_h = r_h + self.discount * self.target_vf(t, b, s_next_h, z_adv)
                q1_h = self.qf1(t, b, s_h, a_h, z_adv)
                q2_h = self.qf2(t, b, s_h, a_h, z_adv)
                qf_synth_loss = qf_synth_loss + ((q1_h - q_target_h) ** 2).mean() \
                                              + ((q2_h - q_target_h) ** 2).mean()
            qf_synth_loss = qf_synth_loss * (self.rollout_weight / max(1, self.rollout_horizon))
            if torch.isfinite(qf_synth_loss).all():
                self.qf1_optimizer.zero_grad()
                self.qf2_optimizer.zero_grad()
                qf_synth_loss.backward()
                self.qf1_optimizer.step()
                self.qf2_optimizer.step()
                self.loss['qf_synth_loss']     = float(qf_synth_loss.item())
                self.loss['qf_synth_nan_skip'] = 0.0
            else:
                self.loss['qf_synth_nan_skip'] = 1.0

        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            self._alpha_var.clamp_(0.0, self.alpha_max)
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()

        for i in range(len(self.agent.z_means[0])):
            z_mean_i = ptu.get_numpy(self.agent.z_means[0][i])
            self.eval_statistics[f'train/z_mean_{i}'] = z_mean_i
        z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
        self.eval_statistics['train/Z_variance'] = z_sig

        if self.use_club:
            self.eval_statistics['train/loss_club_model'] = ptu.get_numpy(club_model_loss)
            self.eval_statistics['train/loss_club']        = ptu.get_numpy(club_loss)
        if self.use_FOCAL_cl:
            self.eval_statistics['train/loss_focal']       = ptu.get_numpy(z_loss)
        self.eval_statistics['train/loss_qf']     = np.mean(ptu.get_numpy(qf_loss))
        self.eval_statistics['train/loss_vf']     = np.mean(ptu.get_numpy(vf_loss))
        self.eval_statistics['train/loss_policy'] = np.mean(ptu.get_numpy(policy_loss))
        if self.use_brac:
            self.eval_statistics['train/loss_dual_brack'] = np.mean(ptu.get_numpy(c_loss))
        self.eval_statistics['train/avg_q_values'] = ptu.get_numpy(q1_pred).mean()
        self.eval_statistics['train/avg_v_values'] = ptu.get_numpy(v_pred).mean()
        self.eval_statistics['train/log_policy']   = ptu.get_numpy(log_pi).mean()
        self.eval_statistics['train/alpha']        = ptu.get_numpy(self._alpha_var.reshape(-1)).mean()
        self.eval_statistics['train/div_estimate'] = ptu.get_numpy(div_estimate).mean()

        self.eval_statistics['train/loss_recon']        = float(self.loss.get('recon_loss',        0.0))
        self.eval_statistics['train/synth_policy_loss'] = float(self.loss.get('synth_policy_loss', 0.0))
        self.eval_statistics['train/synth_nan_skip']    = float(self.loss.get('synth_nan_skip',    0.0))
        self.eval_statistics['train/u_raw']             = float(self.loss.get('u_raw',             0.0))
        self.eval_statistics['train/u_tilde']           = float(self.loss.get('u_tilde',           0.0))
        self.eval_statistics['train/alpha_eff']         = float(self.loss.get('alpha_eff',         0.0))
        self.eval_statistics['train/lore_mean_d']        = float(self.loss.get('lore_mean_d',        0.0))
        self.eval_statistics['train/lore_std_d']         = float(self.loss.get('lore_std_d',         0.0))
        self.eval_statistics['train/lore_perturb_mag']   = float(self.loss.get('lore_perturb_mag',   0.0))
        self.eval_statistics['train/lore_perturb_ratio'] = float(self.loss.get('lore_perturb_ratio', 0.0))
        self.eval_statistics['train/lore_weight_max']    = float(self.loss.get('lore_weight_max',    0.0))

        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            club_model=self.club_model.state_dict(),
            c=self.c.state_dict(),
        )
        if self.use_decoder:
            snapshot['decoder_ensemble'] = self.decoder_ensemble.state_dict()
        return snapshot
