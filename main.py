import gym
import ai

env = gym.make('CartPole-v0')

hyper_params = {
    'lr': 1e-3,
    'gamma': 0.95,
    'epsilon': 1,
    'epsilon_final': 0.01,
    'epsilon_decay': 0.95
}

# env.observation_space = 4
# env.action_space = 2
brain = ai.DeepQNetwork(4, 2, hyper_params)
brain.load()

batch_size = 1024
epochs = 5

i = 0
try:
    while 1:
        state = env.reset()
        while 1:
            env.render()
            action = brain.predict(state)
            next_state, reward, done, info = env.step(action)
            reward *= 100

            brain.memory.append([state, action, reward, next_state])
            state = next_state

            i += 1
            if done:
                brain.memory[-1][2] = -reward
                break

            if i == batch_size:
                brain.batch_training(batch_size, epochs)
                i = 0

except KeyboardInterrupt:
    brain.save()

print('CTRL+C')
