# Markov Decision Processes (MDP)

## Markov Decision Processes
!!! df "**Definition** (Markov Decision Processes)"
    A _Markov decision process_ is a tuple $(S, A, \{P_{sa}\}, \gamma, R)$, where: 

    - $S$ is a set of states. (e.g. position on a chess board)<br>
    
    - $A$ is a set of actions. (e.g. next possible move)<br>
    
    - $P_{sa}$ are the state transition probabilities. For each state $s \in S$ and action $a \in A$, $P_{sa}$ is a distribution over the state space. Briefly, $P_{sa}$ gives the distribution over what states we will transition to if we take action $a$ in state $s$.<br>
    
    - $\gamma \in [0, 1]$ is called the discount factor.<br>
    
    - $R : S \times A \rightarrow \mathbb{R}$ is the reward function. (Also written as a function of a state $S$ only, i.e. $R : S \rightarrow \mathbb{R}$).<br>
    
    The logistic: <br>
    
    We start in some state $s_0$, and choose some action $a_0 \in A$. The state of the MDP randomly transitions to some successor state $s_1$, drawn according to $s_1 \sim P_{s_0 a_0}$. Then, we choose another action $a_1$. As a result of this action, the state transitions again, now to some $s_2 \sim P_{s_1 a_1}$. We then pick $a_2$, and so on.... Pictorially, we can represent this process as follows:<br>
    
    $$
    s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} s_3 \xrightarrow{a_3} \cdots
    $$

    Upon visiting the sequence of states $s_0, s_1, \ldots$ with actions $a_0, a_1, \ldots$, our total payoff is given by

    $$
    R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \cdots.
    $$

    Or, when we are writing rewards as a function of the states only, this becomes

    $$
    R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots.
    $$

    For most of our development, we will use the simpler state-rewards $R(s)$, though the generalization to state-action rewards $R(s, a)$ offers no special difficulties. <br>

    Our goal in reinforcement learning is to choose actions over time so as to maximize the expected value of the total payoff:

    $$
    E \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \right]
    $$

    Note that the reward at timestep $t$ is **discounted** by a factor of $\gamma^t$. Thus, to make this expectation large, we would like to accrue positive rewards as soon as possible (and postpone negative rewards as long as possible).

!!! df "**Definition** (Policy)"
    A _policy_ is any function $\pi : S \rightarrow A$ mapping from the states to the actions. We say that we are **executing** some policy $\pi$ if, whenever we are in state $s$, we take action $a = \pi(s)$. We also define the **value function** for a policy $\pi$ according to

    $$
    V^\pi(s) = E \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \mid s_0 = s, \pi \right]
    $$

    $V^\pi(s)$ is simply the expected sum of discounted rewards upon starting in state $s$, and taking actions according to $\pi$. Given a fixed policy $\pi$, its value function $V^\pi$ satisfies the **Bellman equations**:

    $$
    \begin{align*}
        V^\pi(s) &= E \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \mid s_0 = s, \pi \right] \\
                &= R(s) + \gamma E \left[ R(s_1) + \gamma R(s_2) + \gamma^2 R(s_3) + \cdots \right] \tag{s_0 = s} \\
                &= R(s) + \gamma E \left[ V^\pi(s_1) \right] \\
                &= R(s) + \gamma \sum_{s' \in S} P_{s \pi(s)}(s') V^\pi(s')
    \end{align*}
    $$

    This says that the expected sum of discounted rewards $V^\pi(s)$ for starting in $s$ consists of two terms: First, the **immediate reward** $R(s)$ that we get right away simply for starting in state $s$, and second, the expected sum of future discounted rewards. Examining the second term in more detail, we see that the summation term above can be rewritten $E_{s' \sim P_{s \pi(s)}} [V^\pi(s')]$. This is the expected sum of discounted rewards for starting in state $s'$, where $s'$ is distributed according $P_{s \pi(s)}$, which is the distribution over where we will end up after taking the first action $\pi(s)$ in the MDP from state $s$. Thus, the second term above gives the expected sum of discounted rewards obtained after the first step in the MDP. <br> <br>

    Bellman’s equations can be used to efficiently solve for $V^\pi$. Specifically, in a finite-state MDP ($|S| < \infty$), we can write down one such equation for $V^\pi(s)$ for every state $s$. This gives us a set of $|S|$ linear equations in $|S|$ variables (the unknown $V^\pi(s)$’s, one for each state), which can be efficiently solved for the $V^\pi(s)$’s.


!!! im "**Important Note** (Computing $V^{\pi}$)"
    Observe that

    $$
    \begin{aligned}
        V^\pi(s_1) &= R(s_1) + \gamma \sum_{s' \in S} P_{s_1 \pi(s_1)}(s') V^\pi(s') \\
        V^\pi(s_2) &= R(s_2) + \gamma \sum_{s' \in S} P_{s_2 \pi(s_2)}(s') V^\pi(s') \\
        &\ \vdots \\ \\
        V^\pi(s_{|S|}) &= \cdots
    \end{aligned}
    $$

    Thus, we define 
    
    $$
    P^\pi = 
    \begin{bmatrix}
        P_{s_1, \pi(s_1)} & \cdots & P_{s_1, \pi(s_{|S|})} \\
        \vdots & \ddots & \vdots \\
        P_{s_{|S|}, \pi(s_1)} & \cdots & P_{s_{|S|}, \pi(s_{|S|})}
    \end{bmatrix}
    $$

    where $P^\pi$ is a $|S| \times |S|$ matrix, with rows and columns indexed by states $s_i$. We then get the abbreviated form of computing $V^{\pi}$:

    $$
    \begin{aligned}
        \mathbf{V}^\pi &= \underbrace{\mathbf{R}}_{|S| \times 1} + \gamma \underbrace{\mathbf{P}^\pi}_{|S| \times |S|} \underbrace{\mathbf{V}^\pi}_{|S| \times 1} \\
        \mathbf{V}^\pi &= (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}
    \end{aligned}
    $$

!!! df "**Definition** (Optimal Value Function)"
    We define the _optimal value function_ according to

    $$
    V^*(s) = \max_\pi V^\pi(s)
    $$

    In other words, this is the best possible expected sum of discounted rewards that can be attained using any policy. There is also a version of Bellman’s equations for the optimal value function:

    $$
    V^*(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s'). \tag{2}
    $$

    The first term above is the immediate reward as before. The second term is the maximum over all actions $a$ of the expected future sum of discounted rewards we’ll get upon after action $a$.

    We also define a policy $\pi^* : S \rightarrow A$ as follows:

    $$
    \pi^*(s) = \arg \max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s'). \tag{3}
    $$

    Note that $\pi^*(s)$ gives the action $a$ that attains the maximum in the “max” in Equation (2).

    It is a fact that for every state $s$ and every policy $\pi$, we have

    $$
    V^*(s) = V^{\pi^*}(s) \geq V^\pi(s).
    $$

    The first equality says that the $V^{\pi^*}$, the value function for $\pi^*$, is equal to the optimal value function $V^*$ for every state $s$. Further, the inequality above says that $\pi^*$’s value is at least as large as the value of any other policy. In other words, $\pi^*$ as defined in Equation (3) is the optimal policy. <br> <br>

!!! nt "**Note** (Interesting Property of $\pi^*$)"
    Note that $\pi^*$ has the interesting property that it is the optimal policy for *all* states $s$. Specifically, it is not the case that if we were starting in some state $s$ then there’d be some optimal policy for that state, and if we were starting in some other state $s'$ then there’d be some other policy that’s optimal policy for $s'$. The same policy $\pi^*$ attains the maximum in Equation (1) for all states $s$. This means that we can use the same policy $\pi^*$ no matter what the initial state of our MDP is.

## Solving Finite-State MDPs

We now describe two efficient algorithms for solving finite-state MDPs. For
now, we will consider only MDPs with finite state and action spaces. we will also assume that we know the state
transition probabilities $\{P_{sa}\}$ and the reward function $R$.

!!! df "**Algorithm** (Value Iteration)"
    The _value iteration_ algorithm follows: <br>

    1. For each state $s$, initialize $V(s) := 0$.

    2. Repeat until convergence `{` <br>
        &nbsp;&nbsp;&nbsp;&nbsp; For every state, update $V(s) := R(s) + \max_{a \in A} \gamma \sum_{s'} P_{sa}(s') V(s')$ <br>
    `}` <br>

    This algorithm is repeatedly trying to update the estimated value function using Bellman Equations $(2)$.

!!! im "**Important Note** (Two Ways of Updating the Inner Loop)"
    There are two possible ways of performing the updates in the inner loop of the algorithm. In the first, we can first compute the new values for $V(s)$ for every state $s$, and then overwrite all the old values with the new values. This is called a **synchronous** update. In this case, the algorithm can be viewed as implementing a “Bellman backup operator” that takes a current estimate of the value function, and maps it to a new estimate. <br>
    
    Alternatively, we can also perform **asynchronous** updates. Here, we would loop over the states (in some order), updating the values one at a time.

!!! df "**Definition** (Policy Iteration)"
    The _policy iteration_ algorithm proceeds as follows: <br>

    1. Initialize $\pi$ randomly. 

    2. Repeat until convergence `{` <br>
        &nbsp;&nbsp;&nbsp;&nbsp;(a) Let $V := V^\pi$. <br>
        &nbsp;&nbsp;&nbsp;&nbsp;(b) For each state $s$, let $\pi(s) := \arg \max_{a \in A} \sum_{s'} P_{sa}(s') V(s')$. <br>
    `}` <br>

    Thus, the inner-loop repeatedly computes the value function for the current policy, and then updates the policy using the current value function. (The policy $\pi$ found in step (b) is also called the policy that is **greedy with respect to $V$**.) Note that step (a) can be done via solving Bellman’s equations as described earlier, which in the case of a fixed policy, is just a set of $|S|$ linear equations in $|S|$ variables.

    After at most a finite number of iterations of this algorithm, $V$ will converge to $V^*$, and $\pi$ will converge to $\pi^*$.

Both value iteration and policy iteration are standard algorithms for solving MDPs, and there isn’t currently universal agreement over which algorithm is better. For small MDPs, policy iteration is often very fast and converges with very few iterations. However, for MDPs with large state spaces, solving for $V^\pi$ explicitly would involve solving a large system of linear equations, and could be difficult. In these problems, value iteration may be preferred. For this reason, in practice value iteration seems to be used more often than policy iteration.



    




    