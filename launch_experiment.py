import pathlib
import numpy as np
import torch
import random
import hydra
import omegaconf
import os 
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.agents.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, EnsembleFlattenMlp
from rlkit.torch.agents.plato import PLATO
from rlkit.torch.agents.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def experiment(algorithm, cfg):
    DEBUG = cfg.util_params.debug
    os.environ['DEBUG'] = str(int(DEBUG))
    
    exp_id = 'debug' if DEBUG else cfg.util_params.exp_name

    experiment_log_dir, wandb_logger = setup_logger(
        cfg.env_name,
        variant= omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        exp_id=exp_id,
        base_log_dir=cfg.util_params.base_log_dir, 
        seed=cfg.seed,
        snapshot_mode="gap_and_last",
        snapshot_gap=5, 
        use_wandb=cfg.use_wandb,
    )

    # optionally save eval trajectories as pkl files
    if cfg.algo_params.dump_eval_paths:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train(wandb_logger)
    if wandb_logger:
        wandb_logger.finish()

def initialize(cfg, ):
    if cfg.get("env_name") is None or cfg.get("algo_type") is None:
        raise ValueError(
            "Both Hydra groups must be selected before launch: use "
            "`python launch_experiment.py env=<env> algo=<algo> ...`."
        )

    # Backward-compatible alias: several env configs still use meta_batch_size.
    if "meta_batch_size" in cfg.algo_params:
        with omegaconf.open_dict(cfg):
            cfg.algo_params.meta_batch = cfg.algo_params.meta_batch_size

    # create multi-task environment and sample tasks, normalize obs if provided with 'normalizer.npz'
    env = NormalizedBoxEnv(ENVS[cfg.env_name](**dict(cfg.env_params)))
    global_seed(cfg.seed)
    try:
        env.reset(seed=cfg.seed)
    except TypeError:
        env.seed(cfg.seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = cfg.latent_size

    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if cfg.algo_params.use_next_obs_in_context else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if cfg.algo_params.use_information_bottleneck else latent_dim
    net_size = cfg.net_size
    recurrent = cfg.algo_params.recurrent
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    if cfg.algo_type != 'PLATO':
        raise ValueError("This release contains PLATO only: use `algo=plato`.")

    context_encoder = encoder_model(
        hidden_sizes=cfg.encoder_dims,
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
    )
    qf_hidden_sizes = [net_size, net_size, net_size]
    policy_hidden_sizes = [net_size, net_size, net_size]

    qf1 = FlattenMlp(
        hidden_sizes=qf_hidden_sizes,
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=qf_hidden_sizes,
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=qf_hidden_sizes,
        input_size=obs_dim + latent_dim,
        output_size=1,
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=policy_hidden_sizes,
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    # critic network for divergence in dual form (see BRAC paper https://arxiv.org/abs/1911.11361)
    c = FlattenMlp(
        hidden_sizes=policy_hidden_sizes,
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1
    )

    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **dict(cfg.algo_params)
    )
    task_modes = env.task_modes()
    
    # CLUB estimator on the context, as in CSRO: input [s, a] for reward-shift
    # families and [s, a, r] when s' is part of the context (dynamics shift).
    # The decoder ensemble supplies both the imagined rollouts and u_h.
    if cfg.algo_params.use_club_sa:
        club_input_dim = obs_dim + action_dim
    else:
        club_input_dim = obs_dim + action_dim + reward_dim if cfg.algo_params.use_next_obs_in_context else obs_dim + action_dim

    club_model = encoder_model(
        hidden_sizes=cfg.encoder_dims,
        input_size=club_input_dim,
        output_size=latent_dim * 2,
        output_activation=torch.tanh,
    )

    nets = [agent, qf1, qf2, vf, c, club_model]
    if cfg.algo_params.get('use_decoder', False):
        decoder_ensemble = EnsembleFlattenMlp(
            num_ensemble=cfg.algo_params.get('num_ensemble', 10),
            hidden_sizes=[net_size, net_size, net_size],
            input_size=latent_dim + obs_dim + action_dim,
            output_size=obs_dim + reward_dim,
        )
        nets.append(decoder_ensemble)

    algorithm = PLATO(
        env=env,
        train_tasks=task_modes['train'],
        eval_tasks=task_modes['moderate'],
        extreme_tasks=task_modes['extreme'],
        nets=nets,
        latent_dim=latent_dim,
        **dict(cfg.algo_params)
    )

    # optional GPU mode
    ptu.set_gpu_mode(cfg.use_gpu, cfg.gpu_id)
    algorithm.to()
    return algorithm


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="experiment")
def main(cfg):
    algorithm = initialize(cfg)
    experiment(algorithm, cfg)

if __name__ == "__main__":
    main()
