# Introduction

This project uses reinforcement learning for finding the best actions a digital company can take during user marketing funnel in order to get the maximum amount of users subscribed.

The process is all contained in [Exploratory Data Analysis](notebooks/Exploratory%20Data%20Analysis.ipynb).

Before opening the notebook, execute
```bash
source setup.sh
```
to create a Conda Python virtual environment and install the required Python packages.

If you are not on a Linux based Operating System (like Windows), you will need to setup the environment manually. Make use of the `pyproject.toml` file and Poetry tool for a convenient and easy way of handling the process.

# Data description

The [data](data/SalesCRM%20-%20CRM.csv) comes from a real digital company and it is property anonymized.

There is information about 11032 users in a tabular format. For each of them, there is information about the steps he took in the funnel and at what date each change happened. There are also contextual variables: user's location, education level, whether he did or didn't get the call, etc.

As the steps in the funnel do not always follow the same sequence, that presents the possibility of finding the optimal sequence.

# Approach

The goal of the project was to find the best strategy for the company to maximize the probability that a user subscribes to the product. By best strategy, I mean the best actions the company can take at each point of the user flow.

The first step towards a solution is the right interpretation of the data. Since we have the information about the sequence of steps in the funnel, we can consider that each change from one step to the next is a transition that is caused by the user taking an action, or by the company taking an action. The company actions are the ones that will be optimized. In particular, we decided that three types of transitions will be company actions: messaging the user, proposing a demo of the product and assigning an account manager. The rest of transitions (filling the customer survey, trying the demo without being proposed, taking the call, etc.) will be performed by the user.

Once the interpretation has been established, we must build an environment for the reinforcement learning agent to learn from.
The environment is just an abstraction of how the user behaves. At each point in time, the environment is in a given state, which is composed by the current step, the previous steps, the country of the user, his education level, how many messages he received, etc. When the agent/company takes an action, the state changes, i.e. the environment changes according to a model.

Therefore, it is that model we must implement. The chosen model was a Markov process in which the probability of moving from one step to another depends on the current step, but also on the demographic variables and on whether or not some key previous steps were visisted. For example, the probability of subscribing once the user has an account manager assigned depends on that step (having been assined an account manager), but also on whether or not he filled the customer survey previously, or whether he got the first call after one message or three. The information from the different variables was put together using a Naive-Bayes approach, which is pretty reasonable for the little amount of data available for so many variables.

With the interpretation in hand and the environment implemented, the only thing left is the choice of a model and the training. The model was Proximal Policy Optimization algorithm implemented by `stable-baselines3` Python package.

For the evaluation, we used numerical analysis but also graph visualizations for a more explainable behavior.

# Conclusions

A best strategy was found, as well as the demographics onto which the company should focus its attention.
The best strategy turned out to be to message the user until he gets the call, then offering a demo to get them to sign up to the platform and then assign an account manager.
Some of the countries and education levels we should focus our attention were also listed. The UK, Germany, Saudi Arabia and Canada seem to have high success rate, as well as the education group B27.

The success rate was aroudn 13% in the simulated scenarios, much bigger than the original success rate of 0.45%.

The most important conclusions don't come from the specific best strategy, but from the lessons learned during the process.

Regarding the hardships of this project, the complexity of the environment implementation was quite high. The `gymnasium` package provides a nice and flexible interface, but its very simple and all logic must be handled by the data scientist. This not only required complex design but also meant that the trainings became slower and slower as more features were added.
Nonetheless, the training times are still very good (a few minutes for 100000 episodes).

The biggest help to deal with the complexity of the project was unit testing.
For future developments, I would strongly advise to continue using unit testing and TDD to include new features smoothly.
Also, trying other learning algorithms could yield better performance.

On the other hand, I must mention the importance of using the information of whether or not a previous state was visited. In a purely Markov process that only takes in account the current step in the process, we lose the information about the path we took. And that had a huge impact in how the trained agent behave. Adding whether or not key previous states were visited in a Naive Bayes approach led to a more reasonable behavior. Of course this must be done with care since assuming statistical independence between steps can be non-realistic. For example, signing up to the platform is very correlated with filling the customer survey. This can of course add distorsion to the model.

But in general, the addition of those variables was better than droping them. Indeed the data shows that the user is much more likely to subscribe if those previous steps have happened. It is better that the environment takes that into consideration.
