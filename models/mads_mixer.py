# DCES Utility Decoupling: monotonic value decomposition.
#
#   Q_tot(s,a,c,pi_theta) = w1(s) * Q_pi_theta(s,a) + w2(s) * Q_pi_delta(alpha, c)
#   w1(s) > 0, w2(s) > 0   (monotonic; QMIX-style)
#   trained by minimizing the TD error of Q_tot on the shared reward.
#
# Q_pi_theta (policy-specific, step-level) is supplied externally (the curriculum
# learner's critic value V_pi_theta(s)). This module owns Q_pi_delta (curriculum-
# specific, episodic) and the mixer weights.
#
# Ablation switches (constructor args), each maps to one paper ablation:
#   use_policy_conditioning=False  ->  Q_pi_delta(c)          (drop alpha)   [#4]
#   use_monotonic=False            ->  unconstrained mixer                   [#6]
# The behavior-feature mode (rbf vs raw) is handled upstream where alpha is built [#5].
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim, n_hidden=2):
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class CurriculumCritic(nn.Module):
    """Q_pi_delta(alpha, c): episodic curriculum-specific value.
    Input LayerNorm tames the (possibly large / ill-conditioned) RBF feature alpha."""
    def __init__(self, alpha_dim, c_dim, hidden=64, use_policy_conditioning=True):
        super().__init__()
        self.use_policy_conditioning = use_policy_conditioning
        # Normalize alpha and c SEPARATELY: alpha is high-magnitude (ill-conditioned
        # K_inv) and c is small; a joint LayerNorm drowns out c and collapses Q_delta.
        self.ln_c = nn.LayerNorm(c_dim)
        in_dim = c_dim
        if use_policy_conditioning:
            self.ln_alpha = nn.LayerNorm(alpha_dim)
            in_dim += alpha_dim
        self.net = _mlp(in_dim, hidden, 1)

    def forward(self, alpha, c):
        # alpha: [B, alpha_dim], c: [B, c_dim]
        if self.use_policy_conditioning:
            x = torch.cat([self.ln_alpha(alpha), self.ln_c(c)], dim=-1)
        else:
            x = self.ln_c(c)
        return self.net(x).squeeze(-1)  # [B]


class MonotonicMixer(nn.Module):
    """Q_tot = w1(s)*Q_pi + w2(s)*Q_delta (+ b(s)), with w1,w2 > 0 when monotonic."""
    def __init__(self, state_dim, hidden=64, use_monotonic=True):
        super().__init__()
        self.use_monotonic = use_monotonic
        self.ln = nn.LayerNorm(state_dim)
        self.w_head = _mlp(state_dim, hidden, 2)      # -> [B, 2] weights
        self.b_head = _mlp(state_dim, hidden, 1)      # -> [B, 1] bias

    def forward(self, q_pi, q_delta, state):
        state = self.ln(state)
        w = self.w_head(state)                        # [B, 2]
        if self.use_monotonic:
            w = F.softplus(w)                         # strictly > 0  -> monotonic
        w1, w2 = w[..., 0], w[..., 1]
        b = self.b_head(state).squeeze(-1)
        return w1 * q_pi + w2 * q_delta + b, w1, w2


class PolicyCritic(nn.Module):
    """Q_pi_theta(s): policy-specific value component owned by the decomposition
    (separate from the PPO critic, to avoid scale conflict / bootstrap divergence)."""
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.ln = nn.LayerNorm(state_dim)
        self.net = _mlp(state_dim, hidden, 1)

    def forward(self, state):
        return self.net(self.ln(state)).squeeze(-1)


class UtilityDecomposition(nn.Module):
    """Monotonic value decomposition trained as a centralized critic.

    Q_tot(s,c,alpha) = w1(s)*Q_pi_theta(s) + w2(s)*Q_pi_delta(alpha,c)
    Trained by regressing Q_tot onto the (normalized) GAE return target of the
    curriculum learner's trajectory -- a stable TD(lambda)/MC target instead of a
    1-step bootstrap (which diverged). Q_pi_delta drives the curriculum designer.
    """
    def __init__(self, state_dim, alpha_dim, c_dim, hidden=64,
                 use_monotonic=True, use_policy_conditioning=True):
        super().__init__()
        self.policy_critic = PolicyCritic(state_dim, hidden)
        self.curriculum_critic = CurriculumCritic(
            alpha_dim, c_dim, hidden, use_policy_conditioning)
        self.mixer = MonotonicMixer(state_dim, hidden, use_monotonic)

    def q_delta(self, alpha, c):
        return self.curriculum_critic(alpha, c)

    def iql_loss(self, state, step_target, alpha_step, c_step,
                 alpha_env, c_env, ep_target):
        """Independent Q-Learning value decomposition (per user's spec): every Q
        regresses directly toward the return of the shared system reward.
          Q_pi_theta(s)   -> step return         (step-level policy value)
          Q_pi_delta(a,c) -> episode return       (episodic curriculum value)
          Q_tot = mix(Q_pi_theta, Q_pi_delta, s) -> step return   (trains omega1,2)
        Q_pi_delta (episodic) is the curriculum-specific value driving the designer.
        """
        q_pi = self.policy_critic(state)                          # [B_step]
        q_delta_step = self.curriculum_critic(alpha_step, c_step)  # [B_step] (broadcast)
        q_tot, w1, w2 = self.mixer(q_pi, q_delta_step, state)      # [B_step]
        q_delta_env = self.curriculum_critic(alpha_env, c_env)     # [N] episodic

        l_pi = F.mse_loss(q_pi, step_target)
        l_delta = F.mse_loss(q_delta_env, ep_target)
        l_tot = F.mse_loss(q_tot, step_target)
        loss = l_pi + l_delta + l_tot
        info = {
            "l_pi": l_pi.item(), "l_delta": l_delta.item(), "l_tot": l_tot.item(),
            "q_pi": q_pi.mean().item(), "q_delta": q_delta_env.mean().item(),
            "q_delta_std": q_delta_env.std().item(),
            "w1": w1.mean().item(), "w2": w2.mean().item(),
        }
        return loss, info, q_delta_env
