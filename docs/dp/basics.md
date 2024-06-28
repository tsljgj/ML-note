# Basics of Dynamic Programming

## DP Overview
!!! df "**Definition** (Intuitive Definition of Dynamic Programming)"
    _Dynamic programming_ is a combination of recursion and memorization (and _GUESSING_). It is an exhaustive search. One can think of it as a "careful brute-force." There are two uses for dynamic programming:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;1.  Finding an optimal solution: We want a solution to be as large as possible or as small as possible <br>
    &nbsp;&nbsp;&nbsp;&nbsp;2.  Counting the number of solutions: We want to calculate number of all possible solutions

!!! df "**Definition** (Bellman Equation)"
    Bellman Equation is the formula of how subproblems can be used to calculate the main problem. e.g. $f(x) = f(x-1) + f(x-2)$ is a Bellman Equation.

!!! df "**Definition** (Subproblem Dependency DAG)"
    The dependency relation between subproblems form a directed acyclic graph - _Subproblem Dependency DAG_.  

!!! wr "**Warning** (Limitation of DP)"
    To use memorization, the subproblem dependency graph must be acyclic (it is called DAG).      

!!! df "**Definition** (Bottom-Up Implementation)"
    Bottom-up implementing a DP algorithm means following topological order of the subproblem dependency DAG to do the exact same computation as up-bottom (recursion). For example, when calculating the ith term of Fibonacci sequence, recursively call $f(x) = f(x-1) + f(x-2)$ is up-bottom; calculating the first term to the ith term is bottom-up. 
    
!!! mt "**Methodology** (Solving DP Problems)"
    &nbsp;&nbsp;&nbsp;&nbsp;1.  Define subproblems $\rightarrow$ #subproblems<br>
    &nbsp;&nbsp;&nbsp;&nbsp;2.  Guess (try all possibility after settling something) $\rightarrow$ #choices for guess<br>
    &nbsp;&nbsp;&nbsp;&nbsp;3.  Relate subproblem solutions $\rightarrow$ time/subproblem (often similar to guess)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;4.  Recurs & memorize (or build DP table bottom-up) $\rightarrow$ check if subproblem recurrence is acyclic<br>
    &nbsp;&nbsp;&nbsp;&nbsp;5.  Solve original problem.
    
!!! nt "**Note** (DP Time Complexity)"
    The time Complexity of DP can be calculated by 
    
    $$
    \text{time} = \Theta(\text{#subproblems} \times \text{time per subproblem})
    $$
    
    Note that we treat recursive calls in subproblems as $\Theta(1)$ as they have been memorized!

!!! ex "**Exercise** (The Triangle - IOI 94')"
    
    ``` cpp linenums="1"
            7
          3   8
        8   1   0
      2   7   4   4
    4   5   2   6   5
    ```

    Figure 1 shows a number triangle.

    Write a program that calculates the highest sum of numbers passed on a route that starts at the top and ends somewhere on the base.

    Each step can go either diagonally down to the left or diagonally down to the right.

    ??? sl "**Solution**"
        Define $F[i][j]$ be the maximum sum from top to $(i,j)$. The Bellman equation is:
        
        $$\begin{align*}
        F[i][j] = A[i][j] + \max
            \begin{cases} 
                F[i-1][j] \\
                F[i-1][j-1] \; \text{if j>1}\\
            \end{cases}
        \end{align*}$$
        
!!! ex "**Exercise** (Coin Problem)"
    Consider a set of coins of value $\{1, 3, 4\}$. Find the minimum number of coins s.t. the total value is equal to $S$.<br>
    
    ??? sl "**Solution**"
        Note that this problem ask for the "minimum." We think of using DP. Define $f(x)$ be the minimum #coins where the total value equals to $x$. Observe that to find $f(S)$, it suffices to find $\min\{f(S-1), f(S-3), f(S-4)\}$ (We are guessing now). The Bellman Equation is:
    
        $$
        f(x) = \min\{f(x-1), f(x-3), f(x-4)\} + 1
        $$

        The Bottom-up implementation would be:

        ``` cpp linenums="1"
        f[1]=1, f[2]=2, f[3]=1, f[4]=1; // base case
        for(int i=5;i<=S;i++)
            f[i]=min(min(f[i-1],f[i-3]),f[i-4])+1;
        ```

!!! ex "**Exercise** (Shortest Path: Bellman-Ford)"
    Consider a non-negative weighted directed graph $G = (V, E)$. Find the shortest path between vertex $s$ and vertex $t$. Note that the graph may contain cycles.

    ??? sl "**Solution**"
        Note that this problem ask for the "minimum." We think of using DP. Define $\mathcal{S}(u,v)$ be the shortest path from $u$ to $v$. A naive way of writing the Bellman Equation is 
        
        $$\begin{align*}
        \mathcal{S}(s,t) = \min_{(u,t)\in E}(\mathcal{S}(s,u) + w_{u,t})
        \end{align*}$$

        So far, by Bottom-up implementation, we'll go over all the vertices and edges once. The time complexity is $\Theta(n + m)$. However, this cannot deal with cyclic graph. To find the shortest path in a cyclic graph, we consider convert cyclic $G = (V, E)$ to acyclic $G' = (V', E')$ following "steps": 
        
        <figure markdown="span">
        ![Image title](../../graphs/dp/basics/sp.svg){ width="400" }
        </figure> 

        Define $\mathcal{S}_k(u,v)$ be the shortest path from $u$ to $v$ in $k$ steps. We now come up with a new Bellman Equation:
        
        $$\begin{align*}
        \mathcal{S}_k(s,t) = \min_{(u,t)\in E}(\mathcal{S}_{k-1}(s,u) + w_{u,t})
        \end{align*}$$
        
        Since all edges in $G$ are non-negatively weighted, the maximum steps can take is $n-1$. The #subproblems is now $n^2$ ($k$ can be 0; $k$ has $n$ choices, vertex $t$ has $n$ choices). The time complexity is $\Theta(nm)$ because for each $k$, we go over all edges exactly once.

        The Bottom-up implementation would be:

        ``` cpp linenums="1"
        set all values in s equal to 1e9
        S[0][s] = 0;
        for(int k=1;k<n;k++)
            for(int t=1;t<=n;t++)
                for(all (u,t) in E)
                    S[k][u]=min(S[k][u], S[k-1][u]+w[u][t]);
        ```
!!! im "**Important Note** (Alternative View of DP)"
    For optimization problem, we can view DP as searching for the shortest (longest) path on a DAG.

!!! df "**Definition** (Parent Pointer)"
    The parent pointer is used to record "plan" rather than the "value." It keeps track of how the optimal solution arrives.

!!! ex "**Exercise** (Text Justification)"
    In a text justification problem, define the "badness" of a line starting from ith word and ending in jth word to be: 
    
    $$\begin{align*}
    \text{badness}(i,j) = 
        \begin{cases} 
        {(\text{page width} - \text{words width sum})}^2 \\
        \infty \&nbsp; \text{ if unfit} 
        \end{cases}
    \end{align*}$$

    Find the smallest badness sum of the text. In addition, find all the indexes of words that start a line when the smallest badness sum is achieved.

    ??? sl "**Solution**"
        Define the subproblem `dp[i]` to be the smallest badness sum of the text starting from ith word. This means there are total $n$ subproblems. After setting `i`, we need to **guess** which word to end the line ($\mathcal{O}(n)$). Thus, we have the Bellman equation to be:
        
        $$\begin{align*}
        \text{dp[i]} = \min(\text{dp[j]} + \text{badness}(i,j) \text{ for j in range(i+1, n+1)}) 
        \end{align*}$$

        Check the topological order: the order is $i = n, n-1, n-2, ..., 0$. <br>
        The time complexity is $\mathcal{O}(n^2)$. <br>
        Base Case: `dp[0] = 0` <br>
        The parent point is $\text{parent[i]} = \text{argmin}(...) = j \text{ value}$. <br>
        To print the plan, we access parent pointers this order: 0, parent[0], parent[parent[0]], ...   

## DP for String Problem
!!! mt "**Methodology** (Picking Subproblem for String/Sequence Input)"
    When the input is a string or a sequence, consider choosing these as the subproblem: <br>
    &nbsp;&nbsp;&nbsp;&nbsp;1.  suffixes `x[i:]`   $\mathcal{O}(n)$ &nbsp; topo: right to left <br>
    &nbsp;&nbsp;&nbsp;&nbsp;2.  prefixes `x[:i]`   $\mathcal{O}(n)$   &nbsp; topo: left to right <br>
    &nbsp;&nbsp;&nbsp;&nbsp;3.  substring `x[i:j]` $\mathcal{O}(n^2)$ &nbsp; topo: short to long (length)

!!! ex "**Exercise** (Parenthesization)"
    Consider the problem of matrix multiplication. Given $n$ matrices $A_1, A_2, ..., A_n$. Find the minimum cost of multiplication $A_1A_2...A_n$. Suppose the cost of $(a,b) \times (b,c)$ is $a\times b \times c$. You can change the order of multiplication.

    ??? sl "**Solution**"
        We choose substring as our subproblem. Define `dp[i][j]` to be the minimum cost of multiplying $A_iA_{i+1}...A_j$. #subproblem $= \mathcal{O}(n^2)$. The Bellman equation is:
        
        $$\begin{align*}
        \text{dp}[i][j] = \min_{i\le k < j}(dp[i][k] + dp[k+1][j] + \text{ cost of } A_{i:k} \times A_{k+1:j})
        \end{align*}$$

        The time complexity: $\Theta(n^3)$.<br>
        Topological order: increasing substring size.

        Bottom-up implementation:

        ``` cpp linenums="1"
        for(int len=2;len<=n;len++){
            for(int i=1;i+len-1<=n;i++){
                int j=i+len-1;
                for(int k=i;k<j;k++){
                    dp[i][j] = min(dp[i][j], dp[i][k]+dp[k+1][j]+cost);
                }
            }
        }
        ```

!!! ex "**Exercise** (Edit Distance)"
    Given two strings $\mathbf{x}$ and $\mathbf{y}$, what's the cheapest possible sequence of character edits to turn $\mathbf{x}$ into $\mathbf{y}$? Here are three choices of editing (you can do these operations anywhere in the string): <br>
    &nbsp;&nbsp;&nbsp;&nbsp; 1.  insert C, cost $c_i$ <br>
    &nbsp;&nbsp;&nbsp;&nbsp; 2.  delete C, cost $c_d$ <br>
    &nbsp;&nbsp;&nbsp;&nbsp; 3.  replace C with C', cost $c_{C',r}$ <br>
    
    ??? sl "**Solution**"
        The subproblem is suffixes: edit distance on `x[i:]` & `y[j:]` for all i, j. #subproblems = $nm$. Now guess. To convert $\mathbf{x}$ to $\mathbf{y}$, we must somehow make the first character $C_x$ in $\mathbf{x}$ equal to the first character $C_y$ in $\mathbf{y}$. We now have three choices: <br>
        &nbsp;&nbsp;&nbsp;&nbsp; 1. insert $C_y$ at the head of $\mathbf{x}$ <br>
        &nbsp;&nbsp;&nbsp;&nbsp; 2. delete $C_x$ &nbsp; <br>
        &nbsp;&nbsp;&nbsp;&nbsp; 3. replace $C_x$ with $C_y$. <br>
        
        These choices cover all the possibility. Thus, we have the Bellman equation:
        
        $$\begin{align*}
        \text{dp[i][j]} = \min(\text{dp[i][j+1]}+c_i, \text{dp[i+1][j+1]}+c_d, \text{dp[i+1][j+1]}+c_{C_y,r})
        \end{align*}$$

        Now, draw the dependency table. Note that dp[i][j] only depends on dp[i][j+1] and dp[i+1][j+1].

        <figure markdown="span">
        ![Image title](../../graphs/dp/basics/edit.svg){ width="300" }
        </figure>

        The Bottom-up implementation would be from buttom to top of the table (right to left is also fine):
        
        ``` cpp linenums="1"
        for(int i=n;i>=0;i--)
            for(int j=m;j>=0;j--)
                dp[i][j] = min(dp[i][j+1]+ci, min(dp[i+1][j+1]+cd, dp[i+1][j+1]+c[j]));
        ```

        The final answer would be dp[0][0]. <br>
        Each subproblem takes constant time. The total time complexity is $\Theta(nm)$.<br>

!!! ex "**Exercise** (LCS: Longest Common Subsequence)"
    Given two strings $\mathbf{x}$ and $\mathbf{y}$. Drop some characters in each string to make the remaining strings equal. What is length of the longest remaining string (Longest Common Subsequence)?

    ??? sl "**Solution**"
        Note that if we do not allow replacement in the "Edit Distance" problem, we are doing something very similar to LCS which only allows dropping. The precise way of converting "Edit Distance" problem to LCS is by defining: <br>
        &nbsp;&nbsp;&nbsp;&nbsp;1.  insert C, cost $1$ <br>
        &nbsp;&nbsp;&nbsp;&nbsp;2.  delete C, cost $1$ <br>
        &nbsp;&nbsp;&nbsp;&nbsp;3.  replace C with C', cost $0$ if C = C', else cost $\infty$ <br>

        Detailed solution:<br>
        Define $F[i][j]$ be the LCS of $\mathbf{x}[1:i]$ and $\mathbf{y}[1:j]$. The Bellman equation is:
        
        $$\begin{align*}
        F[i][j] = \max
            \begin{cases} 
                F[i-1][j] \\
                F[i][j-1] \\
                F[i-1][j-1] + 1 \qquad \text{if } A[i] = B[j]
            \end{cases}
        \end{align*}$$
        
!!! ex "**Exercise** (LIS: Longest Increasing Subsequence)"
    Given a sequence $\mathbf{A}$ with length $N$, find the length of the longest increasing subsequence.

    ??? sl "**Solution**"
        This time we use prefixes. However, to guess all the possibility, we now define our subproblem in a way that enforces an element to be in the LIS: Define $F[i]$ to be the length of LIS that has $\mathbf{A}[i]$ as its last element. The Bellman equation is 
        
        $$\begin{align*}
        F[i] = \max_{0\le j < i, \mathbf{A}[j]<\mathbf{A}[i]}{F[j] + 1}
        \end{align*}$$

        The key point here is: we cannot define $F[i]$ to be the LIS in $A[1:i]$ as we do not know whether $F[1:N]$ also uses the same subsequence in $F[i]$. In other word, we are not counting all the possibility. 

!!! mt "**Methodology** (Two Kinds of Guessing)"
    &nbsp;&nbsp;&nbsp;&nbsp;1.  Guess which subproblem to use to solve bigger subproblem. <br>
    &nbsp;&nbsp;&nbsp;&nbsp;2.  Add more subproblems to guess/remember more (e.g. knapsack)

!!! ex "**Exercise** (Knapsack)"
    Given a knapsack (a bag) with capacity $C$ and a list of items of weight $w_i$. Find the maximum weight one can take.
    ??? sl "**Solution**"
        Subproblem = suffix of items. Define dp[i][c] to be the maximum weight when the knapsack has c capacity and item[i:] left to put in. #subproblems = $\Theta(nC)$. Guessing: is item i in the knapsack or not? Two choices. Bellman equation:
        
        $$\begin{align*}
        \text{dp[i][c]} = \max(\text{dp[i+1][c]}, \text{dp}[i+1][c-w_i] + w_i)
        \end{align*}$$

        The time complexity is $\Theta(nC)$.

We now have gone through many basic problems of dynamic programming. In other chapters of DP, we will see more variation of these problems (LCS, LIS, Knapsack...).