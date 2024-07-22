# Prime

## Sieve of Eratosthenes
!!! df "**Algorithm** (Sieve of Eratosthenes)"
    To find all prime numbers up to any given limit $N$, the _Sieve of Eratosthenes_ is an ancient algorithm based on the fact that any multiplication of integer $x$ is not prime. The algorithm is to scan each integer $x_i$ within the limit $N$ starting at $2$ and "erase" $\{2x_i, 3x_i, \cdots, &lfloor;N/x_i&rfloor; * x_i\}$. Note that when an integer $x_j$ is scanned but not marked, we know $x_j$ is a prime. We can proof this by contradiction easily.

    <figure markdown="span">
    ![Image title](https://upload.wikimedia.org/wikipedia/commons/9/94/Animation_Sieve_of_Eratosth.gif){ width="400" }
    <figcaption>Sieve of Eratosthenes [^1]</figcaption>
    </figure>

    [^1]: [Wikipedia](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)

    Code:

    ``` cpp linenums="1"
    void primes(int n){
        memset(v,0,sizeof(v));
        for(int i=2;<=n;i++){
            if(v[i]) continue;
            cout<<i<<endl;
            for(int j=i;j<=n/i;j++)
                v[i*j]=1;
        }
    }
    ```

    The time complexity of Sieve of Eratosthenes is $\mathcal{O}(\sum_{\text{prime} \; p \le N}{\frac{N}{p}}) = \mathcal{O}(N\log\log N)$.