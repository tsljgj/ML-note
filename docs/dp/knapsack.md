# Knapsack

Knapsack problem is a special topic in linear DP. It can be viewed as "DP for String Problems" which use prefixes as subproblems. We have already discussed one classic knapsack problem in the previous chapter. In this chapter, we will dive deeper into variations of Knapsack.

## 0-1 Knapsack

!!! df "**Model** (0-1 Knapsack)"
    Given $N$ items. $W_i$ denotes the weight/volume/size of the ith item. $V_i$ denotes the value of the ith item. Now, we have a knapsack with capacity $M$. Want to know the maximum value when putting items into the knapsack without exceeding the capacity. Each item can only be chosen once.

    We call this type of problems as "0-1 Knapsack" because we are guessing whether an item should be in the knapsack or not. The common strategy to solve 0-1 Knapsack is to use prefixes (items that have been already decided) + updated capacity as subproblems (Later we will find a more elegant way to define subproblems for 0-1 Knapsack). Thus, the solution to above problem is:
    
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

!!! st "**Strategy** (Rolling Array)"
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
    Note that each $F[i-1][j]$ is only responsible for $F[i][j]$, i.e. one copy. Thus, we can make the code even simpler by using only one dimension: $F[j]$. The final version of 0-1 Knapsack code is:
    
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

!!! ex "**Exercise** (Counting in 0-1 Knapsack)"
    Given a set of $n$ integers $A = \{a_1, a_2, \cdots, a_n\}$. Count the number of subsets of $A$ such that the sum of the subset equals to $M$.

    ??? sl "**Solution**"
        This is exactly a 0-1 Knapsack problem. The only difference is we add up $dp[j]$ instead of using $\max$. Note that the base case is `dp[0]=1`. 

        ``` cpp linenums="1"
        #include<bits/stdc++.h>
        using namespace std;
        int dp[10200];
        int main(){
            int n,m;
            cin>>n>>m;
            dp[0] = 1;
            for(int i=1;i<=n;i++){
                int w;
                cin>>w;
                for(int j=m;j>=w;j--){
                    dp[j] += dp[j-w];
                }
            }
            
            cout<<dp[m];
        }
        ```

## Unbounded Knapsack (UKP)
!!! df "**Model** (Unbounded Knapsack)"
    Given $N$ items. $W_i$ denotes the weight/volume/size of the ith item. $V_i$ denotes the value of the ith item. Now, we have a knapsack with capacity $M$. Want to know the maximum value when putting items into the knapsack without exceeding the capacity. Each item has infinite copies. <br> <br>

    The naive way of thinking this problem is to view each item to be multiple items and then use the exact same method as 0-1 Knapsack. Define $s$ to be the maximum number of ith item we can take under capacity $j$. Then, the Bellman equation is:
    
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
    
    From above, notice that 
    
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

    Let us compare this Bellman equation with 0-1 Knapsack's:

    $$\begin{align*}
    F[i][j] = \max \begin{cases}
    F[i-1][j] \\
    F[\mathbf{i-1}][j-W_i]+V_i \qquad \text{if } j \ge W_i \\
    \end{cases}
    \end{align*}$$

    The change is from $i-1$ to $i$. In brief, in Unbounded Knapsack, we allow ith item to be picked multiple time.

    Similar to 0-1 Knapsack, we can also omit one dimension of subproblems. In fact, we have already discussed the implementation of Unbounded Knapsack in 0-1 Knapsack. The bottom-up implementation is:

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

!!! ex "**Exercise** (Counting in UKP)"
    Given a currency system with $n$ denominations, count the number of ways to pay for a $m$ value bill.

    ??? sl "**Solution**"
        This is exactly a Unbounded Knapsack problem. The only difference is we add up $dp[j]$ instead of using $\max$. Note that the base case is `dp[0]=1`. 

        ``` cpp linenums="1"
        #include<bits/stdc++.h>
        using namespace std;
        long long dp[3020];
        int main(){
            int n,m;
            cin>>n>>m;
            dp[0]=1;
            for(int i=1;i<=n;i++){
                long long c;
                cin>>c;
                for(int j=1;j<=m;j++){
                    if(j>=c) dp[j]+=dp[j-c];
                }
            }
            
            cout<<dp[m];
        }
        ```

!!! ex "**Exercise** (Simplifying Monetary System)"
    Define two monetary systems \( A \) and \( B \) as equivalent if every value that can be formed using the currency denominations in \( A \) can also be formed using the denominations in \( B \), and every value that cannot be formed using the denominations in \( A \) also cannot be formed using the denominations in \( B \). Given a monetary system \( A \) with \( n \) different denominations, determine the minimum number of denominations \( m \) for a system \( B \) such that \( A \) is equivalent to \( B \).

    ??? sl "**Solution**"
        Consider the sorted denominations $a_1, a_2, \cdots, a_n$. Note that we must not pick denominations less than $a_1$ because they are not representable in system $A$. We must pick $a_1$. For $a_2$, if it can be formed by $a_1$, then we should not pick it; otherwise, we must pick it or we cannot form $a_2$. Simiarly, when we want to determine whether to pick $a_i$, we want to know if $a_i$ can be formed by $a_1, a_2, \cdots, a_{i-1}$. This can be solved using the method similar to "Counting in UKP." The only difference is, instead of adding up $dp[j]$, we now seek maximum.

        ``` cpp linenums="1"
        #include<bits/stdc++.h>
        using namespace std;
        int a[25020], dp[25020];
        int main(){
            int n,ans;
            cin>>n;
            for(int i=1;i<=n;i++)
                cin>>a[i];
            sort(a+1,a+n+1);
            memset(dp, 0, sizeof(dp));
            dp[0] = 1;
            ans = 0;
            for(int i=1;i<=n;i++){
                if(dp[a[i]]==1) continue;
                ans++;
                for(int j=a[i];j<=25000;j++){
                    dp[j] = max(dp[j], dp[j-a[i]]);
                }
            }
            cout<<ans<<endl;
        }
        ```
        
## Bounded Knapsack (BKP)
!!! df "**Model** (Bounded Knapsack)"
    Given $N$ items. $W_i$ denotes the weight/volume/size of the ith item. $V_i$ denotes the value of the ith item. Now, we have a knapsack with capacity $M$. Want to know the maximum value when putting items into the knapsack without exceeding the capacity. The ith item has $C_i$ copies. <br> <br>

    We can solve this by using 0-1 Knapsack, which would take $\mathcal{O}(M * \sum^{N}_{i=1}C_i)$.

!!! st "**Strategy** (Binary Splitting)"
    Let $m \in \mathbb{N}$. Define $p$ to be the maximum integer s.t&nbsp; $2^0 + 2^1 + 2^2 + \cdots + 2^p \le m$. Define $R$ to be the difference between $2^0 + 2^1 + 2^2 + \cdots + 2^p$ and $m$, i.e. 
    
    $$R = m - (2^0 + 2^1 + \cdots + 2^p) = m - 2^{p+1} + 1$$
    
    Define set $B = \{2^x \mid x \in \mathbb{N} \land x \le p\} \cup \{R\}$. Then we have a surjection:
    
    $$\begin{align*}
    \{s \in \mathbb{N} \mid s \; \text{is the sum of some subset of B} \} \twoheadrightarrow
    \{x \in \mathbb{N} \mid x \le m\}
    \end{align*}$$

    Note that $|B| = p + 2$. For Bounded Knapsack problems, we can use _Binary Splitting_ to divide $C_i$ ith items into at most $p+2$ items. Their weights are &nbsp; 
    
    $$2^0 * V_i, 2^1 * V_i, \cdots, 2^p * V_i, R * V_i$$

    The time complexity is reduced from $\mathcal{O}(M * \sum^{N}_{i=1}C_i)$ to $\mathcal{O}(M * \sum^{N}_{i=1}\log{C_i})$.

## Multiple-Choice Knapsack Problem (MCKP)

!!! df "**Model** (Multiple-Choice Knapsack)"
    Given $N$ categories of items. The ith category has $C_i$ different items. $W_{ij}$ denotes the weight/volume/size of the jth item in the ith category. $V_{ij}$ denotes the value of the jth item in the ith category. Now, we have a knapsack with capacity $M$. Want to know the maximum value when putting items into the knapsack without exceeding the capacity. You can choose at most one item from a particular category. <br> <br>

    Define $F[i,j]$ be the maximal value for the first $i$ categories with capacity $j$. The Bellman equation is:
    
    $$\begin{align*}
    F[i,j] = \max 
    \begin{cases}
        F[i-1,j] \\
        \underset{1\le k \le C_i}{\max} \{F[i-1, j-V_{ik}] + W_{ik}\} \\
    \end{cases}
    \end{align*}$$

    Similar to 0-1 Knapsack, we can ignore the first dimension. The bottom-up implementation then would be:

    ``` cpp linenums="1"
    memset(f, 0xcf, sizeof(f));
    f[0] = 0;
    for (int i=1;i<=n;i++)
        for(int j=m;j>=0;j--)
            for(int k=1;k<=c[i];k++)
                if(j>=w[i][k])
                    dp[j] = max(dp[j], dp[j-w[i][k]] + w[i][k]);    
    ```

Please be aware of the order of loops. Multiple-Choice Knapsack Problem is the foundation of many Tree DP problems. We'll discuss them in later chapters.
    
    
    
    
