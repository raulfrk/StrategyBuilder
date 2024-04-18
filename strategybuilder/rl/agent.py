import numpy as np
from gymnasium import make
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, \
    BaseCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

BEST_HYPERPARAMS = {
    "learning_rate": 0.0001,
    "n_steps": 1600,
    "batch_size": 1024,  # 512 might be good too
    "ent_coef": 0.0005,
    "policy_kwargs": {"net_arch": [64, 64, 64, 64]}
}


class ReportingCallback(BaseCallback):
    def __init__(self, verbose: int = 0, header="Training - "):
        super().__init__(verbose)
        self.cash = []
        self.eod_cash = []
        self.date = None
        self.header = header

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        # print(self.locals["infos"])
        # print(self.locals["new_obs"])
        if self.locals["infos"][0]["date"].date() != self.date and self.cash:
            print(f"{self.header}End of period cash", self.cash[-1])
            print(f"{self.header}Day", self.locals["infos"][0]["date"].date())
            self.eod_cash.append(self.cash[-1])
            self.cash = []
        self.date = self.locals["infos"][0]["date"].date()
        self.cash.append(self.locals["infos"][0]["cash"])
        # print(self.locals)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if len(self.eod_cash) > 0:
            print(f"{self.header}Ep avg cash:", sum(self.eod_cash) / len(self.eod_cash))
        self.eod_cash = []

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def train_agent(env, eval_env, save_path: str, eval_freq=10000, n_eval_episodes=20, max_timesteps=1000000,
                checkpoint_save_freq=100_000, with_reporting: bool = False, reward_threshold=0.2,
                max_no_improvement_evals=3, use_stop_on_no_improvement=False, **kwargs):
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=max_no_improvement_evals, verbose=1)
    if use_stop_on_no_improvement:
        eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=save_path,
                                             log_path=save_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                                             deterministic=True, render=False, callback_after_eval=stop_train_callback)
    else:
        eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=save_path,
                                             log_path=save_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                                             deterministic=True, render=False)
    agent = MaskablePPO("MultiInputPolicy", env, verbose=1, **kwargs)
    cb_list = [eval_callback, CheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_path=save_path,
    )]
    if with_reporting:
        cb_list.append(ReportingCallback())
    callbacks = CallbackList(cb_list)
    agent.learn(total_timesteps=max_timesteps, callback=callbacks)
    return agent


def get_input_weights(agent_path: str, env):
    agent = MaskablePPO.load(agent_path)

    params = agent.policy.state_dict()
    labels = [x for x in env.observation_space.keys()]
    # Access the weights of the first layer (assuming it's a fully connected layer)
    first_layer_weights = params['mlp_extractor.policy_net.0.weight'].cpu().numpy()

    # Calculate feature importance based on weights
    feature_importance = np.abs(first_layer_weights).sum(axis=0)  # Sum of absolute weights along each feature dimension
    importance_dict = {}
    for i, importance in enumerate(feature_importance):
        importance_dict[labels[i]] = importance
    return importance_dict


def create_taining_envs(train, test, env_name, obs_columns, n_subproc, commission, use_subproc=False, **kwargs):
    train_data = np.array_split(train, n_subproc)
    train_envs = []

    for env_data in train_data:
        def f():
            env = make(env_name, data=env_data, obs_columns=obs_columns, reward_multiplier=1, commission=commission,
                       **kwargs)
            env = Monitor(env)
            return env

        train_envs.append(f)
    if use_subproc:
        train_env = SubprocVecEnv(train_envs, start_method="fork")
    else:
        train_env = DummyVecEnv(train_envs)
    train_env = VecNormalize(train_env, clip_obs=100, clip_reward=10)

    eval_env = make(env_name, data=test, obs_columns=obs_columns, reward_multiplier=1, commission=commission, **kwargs)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, clip_obs=100, clip_reward=10)

    return train_env, eval_env


def create_taining_envs_multi_symbol(train_datasets, test, env_name, obs_columns, n_subproc, commission,
                                     use_subproc=False, **kwargs):
    train_data = []
    train_envs = []
    for train in train_datasets:
        train_data.extend(np.array_split(train, n_subproc // len(train_datasets)))
    for env_data in train_data:
        def f():
            env = make(env_name, data=env_data, obs_columns=obs_columns, reward_multiplier=1, commission=commission,
                       **kwargs)
            env = Monitor(env)
            return env

        train_envs.append(f)
    if use_subproc:
        train_env = SubprocVecEnv(train_envs, start_method="fork")
    else:
        train_env = DummyVecEnv(train_envs)
    train_env = VecNormalize(train_env, clip_obs=100, clip_reward=10)

    eval_env = make(env_name, data=test[0], obs_columns=obs_columns, reward_multiplier=1, commission=commission,
                    **kwargs)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, clip_obs=100, clip_reward=10)

    return train_env, eval_env
