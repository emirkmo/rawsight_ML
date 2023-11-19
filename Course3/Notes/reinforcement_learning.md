# Reinforcement learning

## Formalism

- **Agent** is the learner and decision maker.
- **States** The observations made by the agent can be in at time $t$ (possible states of existence).
- **Actions** $A_t$ are the set of choices that can be made by the agent at time $t$ (possible moves).
- **Rewards** $R_t$ are the feedback to the agent at time $t$. Rewards can be set for reaching a certain state or set of outcomes which arise from state (winning in chess while states are chess boards positions and pieces). Or for achieving an outcome based on an outcome (winning in chess)
- **Discount factor** $\gamma$ is a constant in $[0, 1]$ that determines the present value of future rewards.
- **Return** $G_t$ is the total discounted reward from time step $t$. $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$. Return is discounted because the agent prefers rewards now rather than later.
- **Policy** $\pi$ is a distribution over actions given states. A way of picking actions that the agent will follow given the current state, rewards, and possible states (and possibly actions). Based on the return.

Job of a reinforcement algorithm could be to find the best policy. The best policy is the one that maximizes the expected return. It basically means choosing a policy that defines the best action for each state.

### Markov Decision Process (MDP)

Reinforcement learning is basically a Markov Decision Process (MDP). MDP means action and reward are dependent on current state, not on the history of states. The agent is in a state, takes an action, and receives a reward. The next state and reward depends only on the current state and action.

Mathematically the component of a Markov Decision Process is a tuple $(S, A, P, R, \gamma)$ where:

- $S$ is a finite set of states.
- $A$ is a finite set of actions.
- $P$ is a state transition probability matrix. $P_{ss'}^a = P[S_{t+1} = s' | S_t = s, A_t = a]$. It is the probability of transitioning to state $s'$ at time $t+1$ given that the agent was in state $s$ at time $t$ and took action $a$.
- $R$ is a reward function. $R_s^a = E[R_{t+1} | S_t = s, A_t = a]$. It is the expected immediate reward received after transitioning to state $s'$ at time $t+1$ given that the agent was in state $s$ at time $t$ and took action $a$.
- $\gamma$ is a discount factor, $\gamma \in [0, 1]$.
- $\pi$ is a policy, $\pi(a|s) = P[A_t = a | S_t = s]$. It is the probability of taking action $a$ at time $t$ given that the agent was in state $s$ at time $t$.
- $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ is the return. It is the total discounted reward from time step $t$.
- $v(s) = E[G_t | S_t = s]$ is the state-value function. It is the expected return starting from state $s$.
- $q(s, a) = E[G_t | S_t = s, A_t = a]$ is the action-value function. It is the expected return starting from state $s$, taking action $a$.

### Reward strategies

Rewards can be linear -1, 0, 1. Or non-linear, e.g. 1, 0, -1000.

An example of linear rewards is the game of chess, where the reward is 1 for a win, 0 for a draw, and -1 for a loss.

An example of a non-linear reward is helicopter piloting, where the reward is -1000 for for crashing, 1 for flying well.

In the case of the mars rower, it could be 40 for getting around an obstacle on the left side, 0 for all other states, and 100 for reaching the goal via the right side. So rewards don't have to be negative, or equal/even. It is context dependent.


### State action value function

Start in state s. Take action a (once), then behave optimally afterwards.

If you can compute Q(s, a) for all s and a, then you can compute the optimal policy by choosing the action that maximizes Q(s, a)
for each state.

#### Bellman equation

Used to Compute Q(s, a). R(s) reward. $\gamma$ discount factor. $\pi$ policy.
s' is the next state. a' is the next possible action.

Formula:

$Q(s, a) = R(s) + \gamma \max_{a'} Q(s', a')$


The reward function is recursive for Bellman equation for each next step. Immediate reward + discounted future reward
Where discount future reward is current reward + i (no discount), then discounted future reward...

#### Stochastic


Just a probably of taking an action in a state (so not deterministic). In the stochastic reinforcment learning case,
we are maximing the expected return (average value) is the goal of the reward sequences.

Bellman equation for stochastic:

$Q(s, a) = R(s) + \gamma E[\max_{a'} Q(s', a')]$

Where $E[\max_{a'} Q(s', a')]$ is the expected value of the maximum action value function for the future states.
What, on average, you expect to receive as reward from that state on.

(This usually lowers the reward for each state) due to missteps.
