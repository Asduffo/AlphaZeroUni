from gym.envs.registration import register

""""""
# import gym
# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if 'Connect4Env-v0' in env:
#         print("Remove {} from registry".format(env))
#         del gym.envs.registration.registry.env_specs[env]


register(
    id='Connect4Env-v0',
    entry_point='gym_connect4.envs:Connect4Env',
)

