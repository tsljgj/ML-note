# Interval DP

Interval DP is a special kind of linear DP. It can be viewed as "DP for String Problems" which use substring as subproblems.

!!! ex "**Exercise** (Merging Stones)"
    Given $n$ piles of rocks, the ith pile has weight $A_i$, merging two adjacent piles cost the sum of their weight, find the minimum cost to merge all the piles into one pile.

    ??? sl "**Solution**"
        Note that every pile of rocks can be viewed as a merge of several piles. Define $f[l,r]$ be the minimum cost to merge lth pile and rth pile. The Bellman equation is:
        
        $$\begin{align*}
        f[l,r] = \underset{l\le k<r}{\min}{\{f[l,k] + f[k+1,r]\} + \sum_{i=1}^{r}A_i}
        \end{align*}$$

        Note that when we use substrings as subproblems, we update array from smallest interval to largest interval. So given `l`, we'll use `r = l + len -1`. We'll use prefixes sum to calculate $\sum_{i=1}^{r}A_i$.
        
        ``` cpp linenums="1"
        memset(f, 0x3f, sizeof(f));  // infinitely large
        for(int i=1;i<=n;i++){
            f[i][i]=0;
            sum[i]=sum[i-1]+a[i];
        }
        for(int len=2;len<=n;len++){
            for(int l=1;l<=n-len+1;l++){
                int r = l + len -1;
                for(int k=l;k<r;k++)
                    f[l][r] = min(f[l][r], f[l][k]+f[k+1][r]);
                f[l][r] += sum[r] - sum[l-1];
            }
        }

        cout<<f[1][N];
        ```
        
        
        
        
    
