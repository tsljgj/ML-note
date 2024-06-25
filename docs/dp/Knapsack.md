# Knapsack

Knapsack problem is an important topic in DP. We have already discussed one classic knapsack problem in the previous chapter. In this chapter, we will dive deeper into variations of Knapsack.

## 0/1 Knapsack

!!! df "**Model** (0/1 Knapsack)"
    Given $N$ items. $W_i$ denotes the weight/volume/size of the ith item. $V_i$ denotes the value of the ith item. Now, we have a knapsack with capacity $M$. Want to know the maximum value when putting items into the knapsack without exceeding the capacity. We can put each item into the knapsack only **once**.

    We call this type of problems as "0/1 Knapsack" because we are guessing whether an item should be in the knapsack or not. The common strategy to solve 0/1 Knapsack is to use prefixes (items that have been already decided) + updated capacity as subproblems (Later we will find a more elegant way to define subproblems for 0/1 Knapsack). Thus, the solution to above problem is:
    
    Define $F[i][j]$ to be the maximum value of first i items + j capacity. The Bellman equation is:
    
    $$\begin{align*}
    F[i][j] = \max \begin{cases}
    F[i-1][j] \\
    F[i-1][j-W_i]+V_i\qquad \text{if } j \ge W_i \\
    \end{cases}
    \end{align*}$$

    The bottom-up implementation is:

    ``` cpp linenums="1"
    memset(f, 0xcf, sizeof(f)); // setting inital value to be negative infinite (-808464433)
    f[0][0] = 0; // Base Case
    for(int i=1;i<=n;i++)
        for(int j=0;j<=m;j++)
            if(j>=w[i])
                f[i][j] = max(f[i-1][j], f[i-1][j-w[i]]+v[i]);
            else
                f[i][j] = f[i-1][j];
    ```

!!! rm "**Remark** (Rolling Array)"
    Note that $F[i][j]$ only relies on $F[i-1][0:j]$. We can use _Rolling Array_ to optimize space cost:

    ``` cpp linenums="1"
    int f[2][MAX_M+1];
    memset(f, 0xcf, sizeof(f));
    f[0][0];
    for(int i=1;i<=n;i++){
        for(int j=0;j<=m;j++)
            if(j>=w[i])
                f[i&1][j] = max(f[(i-1)&1][j], f[(i-1)&1][j-w[i]]+v[i]);
            else
                f[i&1][j] = f[(i-1)&1][j];
    }    

    int ans = 0;
    for(int j=0;j<=m;j++){
        ans = max(ans, f[n&1][j]);
    }
    ```
    
    Note that `i&1` is $1$ when `i` is odd, $0$ when `i` is even. The space complexity is thus $\mathcal{O}(M)$ instead of $\mathcal{O}(NM)$.
    
!!! im "**Important Note** (Simplifying Template)"
    Note that each $F[i-1][j]$ is only responsible for $F[i][j]$, i.e. one copy. Thus, we can make the code even simpler by using only one dimension: $F[j]$. The final version of 0/1 Knapsack code is:
    
    ``` cpp linenums="1"
    int f[MAX_M+1];
    memset(f, 0xcf, sizeof(f));
    f[0]=0;
    for(int i=1;i<=n;i++)
        for(int j=m;j>=v[i];j--)
            f[j] = max(f[j], f[j-v[i]] + w[i]);
    ```

    We need to update `f[j]` in reverse order (from `m` to `v[i]`) because `f[j]` is updated by `f[j-v[i]]`, which is less than `j` so it would be updated before we update `f[j]` in forward order (from `v[i]` to `m`). 

    **Moreover**, let us think about the meaning of updating `f[j]` in forward order. If we update `f[j-v[i]]` before updating `f[j]`, what might happen is: `f[j-v[i]]` takes ith item, i.e. 
    
    $$f[j-v[i]] = max(f[j], f[j-v[i]] + w[i]) = f[j-v[i]] + w[i]$$
    
    Then, if `f[j]` again takes in ith item, we took ith item for **two** times. This indicates that by looping in forward order, we allow items to be taken multiple times instead of just one time.

## Unbounded Knapsack
!!! df "**Model** (Unbounded Knapsack)"
    Given $N$ items. $W_i$ denotes the weight/volume/size of the ith item. $V_i$ denotes the value of the ith item. Now, we have a knapsack with capacity $M$. Want to know the maximum value when putting items into the knapsack without exceeding the capacity. Each item can be put into the knapsack for **infinite** times. <br> <br>

    The naive way of thinking this problem is to view each item to be multiple items and then use the exact same method as 0/1 Knapsack. Define $s$ to be the maximum number of ith item we can take under capacity $j$. Then, the Bellman equation is:
    
    $$\begin{align*}
    F[i][j] = \max \begin{cases}
    F[i-1][j] \\
    F[i-1][j-W_i]+V_i\; \\
    F[i-1][j-2W_i]+2V_i\; \\
    \cdots                  \\
    F[i-1][j-sW_i]+sV_i\;     \\
    \end{cases}
    \end{align*}$$

    However, this is too slow. We need a better way to do this:

    Observe the difference between the above equation and the second equation which replaces $j$ with $j-W_i$: 

    $$\begin{align*}
    F[i][j] = \max \begin{cases}
    F[i-1][j] \\
    F[i-1][j-W_i]+V_i\; \\
    F[i-1][j-2W_i]+2V_i\; \\
    \cdots                  \\
    F[i-1][j-sW_i]+sV_i\;     \\
    \end{cases}
    \end{align*}$$

    $$\begin{align*}
    F[i][j-W_i] = \max \begin{cases}
    F[i-1][j-W_i] \\
    F[i-1][j-2W_i]+V_i\; \\
    F[i-1][j-3W_i]+2V_i\; \\
    \cdots                  \\
    F[i-1][j-sW_i]+(s-1)V_i\;     \\
    \end{cases}
    \end{align*}$$
    
    Note that 
    
    $$\begin{align*}
    F[i][j] = \max(f[i-1][j], f[i][j-W_i] + V_i)
    \end{align*}$$
    
    This indicates that we can update $f[i][j-W_i]$ first. Intuitively, we can interpret this equation as the maximum value of choosing first $i$ items under $j$ capacity is the maximum between choosing $0$ ith item and choosing another ith item in addition to $F[i][j-W_i]$, which is the maximum value of choosing first $i$ items under $j-W_i$ capacity. The final Bellman equation is:
    
    $$\begin{align*}
    F[i][j] = \max \begin{cases}
    F[i-1][j] \\
    F[\mathbf{i}][j-W_i]+V_i \qquad \text{if } j \ge W_i \\
    \end{cases}
    \end{align*}$$

    Let us compare this Bellman equation with 0/1 Knapsack's:

    $$\begin{align*}
    F[i][j] = \max \begin{cases}
    F[i-1][j] \\
    F[\mathbf{i-1}][j-W_i]+V_i \qquad \text{if } j \ge W_i \\
    \end{cases}
    \end{align*}$$

    The change is from $i-1$ to $i$. In brief, in Unbounded Knapsack, we allow ith item to be picked multiple time.

    Similar to 0/1 Knapsack, we can also omit one dimension of subproblems. In fact, we have already discussed the implementation of Unbounded Knapsack in 0/1 Knapsack. The bottom-up implementation is:

    ``` cpp linenums="1"
    int f[MAX_M+1];
    memset(f, 0xcf, sizeof(f));
    f[0]=0;
    for(int i=1;i<=n;i++)
        for(int j=v[i];j<=m;j++)
            f[j] = max(f[j], f[j-v[i]] + w[i]);
    ```

!!! im "**Important Note** (Uniqueness of Unbounded Knapsack)"
    In all Knapsack problems that have subproblems reduced to one dimension, only Unbounded Knapsack is implemented in forward looping order. If not Unbounded Knapsack, the looping is always from `m` to `v[i]` (from larger to smaller) instead of `v[i]` to `m` (from smaller to larger). 
    

    
    
    
    

    
    
    
  
    

    

    
    
     
    
