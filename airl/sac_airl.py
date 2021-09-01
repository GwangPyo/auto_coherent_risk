from sac.policy import MlpIQNSACPolicy


class AIRLPolicy(MlpIQNSACPolicy):
    def train_step(self, batch_data, critic_optim, actor_optim):
        # obs, actions, rewards, next_obs, dones, info = batch_data
        obs, action_expert, _, next_obs, dones, _ = batch_data
        expert_batch = obs.shape[0]//2


