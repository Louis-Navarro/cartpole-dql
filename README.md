AI using DQL to play CartPole-v0
================================
Here is a pretty simple AI that learns to play OpenAI's CartPole map using Deep-Q-Learning.

## How to install
1. Download this repository using `git clone https://github.com/docaro/cartpole-dql.git` (or by downloading the `.zip` file) and open a terminal in the folder (using `cd` for instance).

2. Install the required packages using `pip install -r requirements`. This may take a couple of minutes.

3. Congratulation, you can now launch `main.py` and see the AI learning (and dying a lot) !

## How does the program work
1. The environment is created using the gym package, which was made by the OpenAI team.

2. The AI was created using Tensorflow 2.0, the AI package developped by Google Deep Mind in C++, Python and CUDA for many different languages.

3. For the AI to predict which action to take, it needs to know the number of inputs and of outputs when created. Thanksfully, all gym's environments have a feature called `observation_space` which tells us the number of inputs and a feature called `action_space` which tells us the number of possible actions.

4. When asking the AI which action it should do, it returns the index of the action to take (0 or 1 since there is only 2 possible action : left and right). Then, we can use `env.step` to get the `next_state`, `reward`, and if he died (`done`).

5. Those variables are used in the training. Using Bellman's Equation (see image below) we can get the desired prections and then train the AI (using Adam optimizer and MSE loss)
![bellman equation]

6. We also need the AI to explore and not just take the best decision (to arrive at the global minimum and not the local minimum) so we define a variable named `epsilon` which defines the chance of taken a random decision instead of using the AI


[bellman equation]: https://www.oreilly.com/library/view/reinforcement-learning-with/9781788835725/assets/5051739f-0788-416d-9182-38ae2169ffca.png
