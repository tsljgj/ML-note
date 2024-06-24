# Knapsack

Knapsack problem is an important topic in DP. We have already discussed one classic knapsack problem in the previous chapter. In this chapter, we will dive deeper into variations of Knapsack.

## 0/1 Knapsack

!!! df "**Model** (0/1 Knapsack)"
    Given $N$ items. $W_i$ denotes the weight/volume/size of the ith item. $V_i$ denotes the value of the ith item. Now, we have a knapsack with capacity $M$. Want to know the maximum value when putting items into the knapsack without exceeding the capacity.

    We call this type of problems as "0/1 Knapsack" because we are guessing whether an item should be in the knapsack or not. The common strategy to solve 0/1 Knapsack is to use prefixes (items that have been already decided) + updated capacity as subproblems (Later we will find a more elegant way to define subproblems for 0/1 Knapsack). Thus, the solution to above problem is:
    
    Define $F[i][j]$ to be the maximum value of first i items + j capacity. The Bellman equation is:
    
    $$\begin{align*}
    F[i][j] = \max \begin{cases}
    F[i-1][j] \\
    F[i-1][j-W_i]+V_i\; \text{if } j \ge W_i \\
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

!!! im "**Important Note** (Rolling Array)"
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
    
    
    
  
    

    

    
    
     
    
