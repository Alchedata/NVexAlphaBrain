"""RL Token (RLT) — Physical Intelligence's bottleneck-encoder TD3 method.

Public API re-exported here so callers can write:

    from AlphaBrain.training.reinforcement_learning.algos.RLT import (
        RLTActor, RLTEncoderDecoder, rlt_td_critic_update, ...
    )

instead of the longer per-module path
``...algos.RLT.rlt_trainer`` / ``...algos.RLT.rlt_actor_critic`` / etc.
"""
from AlphaBrain.training.reinforcement_learning.algos.RLT.rlt_actor_critic import (
    RLTActor,
    RLTCritic,
    RLTQCritic,
    soft_update_target,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT.rlt_encoder_decoder import (
    RLTDecoder,
    RLTEncoder,
    RLTEncoderDecoder,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT.rlt_rollout_fast import (
    rlt_collect_group_steplock,
    rlt_collect_multitask_steplock,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT.rlt_trainer import (
    BatchInferenceServer,
    RLTEpisode,
    RLTStepRecord,
    collect_observations_fast,
    compute_rlt_gae,
    extract_action_queries_dataset,
    extract_action_queries_from_obs,
    pretrain_encoder_step,
    push_episodes_to_buffer,
    rlt_collect_group,
    rlt_ppo_loss,
    rlt_td_actor_update,
    rlt_td_critic_update,
    rlt_td_update,
    vla_finetune_step,
)
