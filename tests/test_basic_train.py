
# def pipeline_fn(_):
#     group_interp = GroupAgentInterpretation()
#     data = MDPDataBunch.from_env('CartPole-v1', max_steps=40, render='rgb_array', bs=5, device='cpu')
#     model = DQN(data, tree=ExperienceReplay(memory_size=100, reduce_ram=True))
#     learn = AgentLearner(data, model)
#     learn.fit(2)
#     interp = AgentInterpretation(learn)
#     interp.plot_rewards(cumulative=True, per_episode=True, group_name='run', no_show=True)
#     group_interp.add_interpretation(interp)
#     data.close()
#     return group_interp.analysis
#
#
# def test_pipeline_init():
#     pl = PipeLine(2, pipeline_fn)
#     print(pl.start(5))
