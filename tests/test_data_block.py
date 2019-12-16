import pytest
from fastai.basic_train import ItemLists

def validate_item_list(item_list: ItemLists):
    # Check items
    for i, item in enumerate(item_list.items):
        if item.done: assert not item_list.items[
            i - 1].done, f'The dataset has duplicate "done\'s" that are consecutive.'
        assert item.state.s is not None, f'The item: {item}\'s state is None'
        assert item.state.s_prime is not None, f'The item: {item}\'s state prime is None'



# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_mdp_from_pickle(env):
#     data = MDPDataBunch.from_env(env, render='rgb_array')
#     model = DQN(data)
#     learner = AgentLearner(data, model)
#     learner.fit(2)
#     data.to_pickle(path='data/CartPole-v0_testing')
#     data = MDPDataBunch.from_pickle(path='data/CartPole-v0_testing')
#     del data
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_mdp_to_csv(env):
#     data = MDPDataBunch.from_env(env, render='rgb_array')
#     model = DQN(data)
#     learner = AgentLearner(data, model)
#     learner.fit(2)
#     data.to_csv()
#     data.train_ds.env.close()
#     data.valid_ds.env.close()
#     del learner
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_mdp_to_pickle(env):
#     data = MDPDataBunch.from_env(env, render='rgb_array')
#     model = DQN(data)
#     learner = AgentLearner(data, model)
#     learner.fit(2)
#     data.to_pickle()
#     data.train_ds.env.close()
#     data.valid_ds.env.close()
#     del learner
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_mdp_clean_callback(env):
#     data = MDPDataBunch.from_env(env, render='rgb_array')
#     model = DQN(data)
#     learner = AgentLearner(data, model)
#     learner.fit(15)
#     data.train_ds.env.close()
#     data.valid_ds.env.close()
#     del learner
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_mdp_databunch(env):
#     data = MDPDataBunch.from_env(env, add_valid=False, render='rgb_array')
#     for i in range(5):
#         for _ in data.train_ds:
#             data.train_ds.action = Action(taken_action=data.train_ds.action.action_space.sample(),
#                                           action_space=data.train_ds.action.action_space)
#
#     validate_item_list(data.train_ds.x)
#     del data
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_mdp_dataset_iter(env):
#     dataset = MDPDataset(gym.make(env), memory_manager = partial(MDPMemoryManager, strategy='k_partition_top'), bs=8,
#                          render='rgb_array')
#
#     for epoch in range(5):
#         for el in dataset:
#             dataset.action.set_single_action(dataset.env.action_space.sample())
#
#     # Check items
#     for i, item in enumerate(dataset.x.items):
#         if item.done: assert not dataset.x.items[
#             i - 1].done, f'The dataset has duplicate "done\'s" that are consecutive.'
#         assert item.state.s is not None, f'The item: {item}\'s state is None'
#         assert item.state.s_prime is not None, f'The item: {item}\'s state prime is None'
#     del dataset
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_mdpdataset_init(env):
#     try:
#         init_env = gym.make(env)
#     except error.DependencyNotInstalled as e:
#         print(e)
#         return
#
#     data = MDPDataset(init_env, None, 64, 'rgb_array')
#
#     try:
#         max_steps = data.max_steps
#         assert max_steps is not None, f'Max steps is None for env {env}'
#     except MaxEpisodeStepsMissingError as e:
#         return
#
#     envs_to_test = {
#         'CartPole-v0': 200,
#         'MountainCar-v0': 200,
#         'maze-v0': 2000
#     }
#
#     if env in envs_to_test:
#         assert envs_to_test[env] == max_steps, f'Env {env} is the wrong step amount'
#     del data
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_bound_init(env):
#     try:
#         init_env = gym.make(env)
#     except error.DependencyNotInstalled as e:
#         print(e)
#         return
#
#     for bound in (Bounds(init_env.action_space), Bounds(init_env.observation_space)):
#         if env.lower().__contains__('continuous'):
#             assert bound.n_possible_values == np.inf, f'Env {env} is continuous, should have inf v.'
#         if env.lower().__contains__('deterministic'):
#             assert bound.n_possible_values != np.inf, f'Env {env} is deterministic, should have discrete v.'
#     init_env.close()
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_action_init(env):
#     try:
#         init_env = gym.make(env)
#     except error.DependencyNotInstalled as e:
#         print(e)
#         return
#
#     taken_action = init_env.action_space.sample()
#     raw_action = np.random.rand(len(Bounds(init_env.action_space).max))
#     init_env.reset()
#     _ = init_env.step(taken_action)
#
#     action = Action(taken_action=taken_action, raw_action=raw_action, action_space=init_env.action_space)
#
#     if list_in_str(env, ['mountaincar-', 'cartpole', 'pong']):
#         assert any([action.taken_action.dtype in (int, torch.int, torch.int64)]), f'Action is wrong dtype {action}'
#         assert any([action.raw_action.dtype in (float, torch.float32, torch.float64)]), f'Action is wrong dtype {action}'
#     if list_in_str(env, ['carracing', 'pendulum']):
#         assert any([action.taken_action.dtype in (float, torch.float32, torch.float64)]), f'Action is wrong dtype {action}'
#         assert any([action.raw_action.dtype in (float, torch.float32, torch.float64)]), f'Action is wrong dtype {action}'
#     init_env.close()
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_state_init(env):
#     try:
#         init_env = gym.make(env)
#     except error.DependencyNotInstalled as e:
#         print(e)
#         return
#
#     taken_action = init_env.action_space.sample()
#     state = init_env.reset()
#     state_prime, reward, done, info = init_env.step(taken_action)
#     State(state, state_prime, init_env.observation_space)
#     init_env.close()
#
#
# @pytest.mark.parametrize("env", sorted(['CartPole-v0']))
# def test_state_str(env):
#     try:
#         init_env = gym.make(env)
#     except error.DependencyNotInstalled as e:
#         print(e)
#         return
#
#     render = 'rgb_array'
#     if isinstance(init_env, TimeLimit) and isinstance(init_env.unwrapped, (AlgorithmicEnv, discrete.DiscreteEnv)):
#         render = 'ansi' if render == 'rgb_array' else render
#
#     taken_action = init_env.action_space.sample()
#     state = init_env.reset()
#
#     try:
#         alt_s = init_env.render(render)
#     except NotImplementedError:
#         return
#
#     state_prime, reward, done, info = init_env.step(taken_action)
#     alt_s_prime = init_env.render(render)
#     State(state, state_prime, alt_s, alt_s_prime, init_env.observation_space).__str__()
#     init_env.close()
#
#
# # @pytest.mark.parametrize("env", sorted(['CartPole-v0', 'maze-random-5x5-v0']))
# # def test_state_full_episode(env):
# #     try:
# #         init_env = gym.make(env)
# #     except error.DependencyNotInstalled as e:
# #         print(e)
# #         return
# #
# #     done = False
# #     state = init_env.reset()
# #     while not done:
# #         taken_action = init_env.action_space.sample()
# #         alt_state = init_env.render('rgb_array')
# #         state_prime, reward, done, info = init_env.step(taken_action)
# #         alt_s_prime = init_env.render('rgb_array')
# #         State(state, state_prime, alt_state, alt_s_prime, init_env.observation_space)
# #         state = state_prime
# #         if done:
# #             assert state_prime is not None, 'State prime is None, this should not have happened.'
# #     init_env.close()
