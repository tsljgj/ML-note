# Basic Data Structure

## Stack
!!! st "**Strategy** (Opposite Stacks)"
    Putting two stacks on the opposite direction to simulate Editor-like effect.

!!! ex "**Exercise** (In/out Stack)"
    Given $n$ integers $1, 2, \cdots, n$, we want every number to be pushed into an infinite stack once and then popped once. If the in-stack order is $1, 2, \cdots, n$, then how many possible out-stack orders are there?

    ??? sl "**Solution 1** (Iteration)"
        Consider the position of $1$ in the out-stack order. If $1$ is on position $k$, then the process is:
        <br>&nbsp;&nbsp;&nbsp;&nbsp; 1. push $1$ into the stack
        <br>&nbsp;&nbsp;&nbsp;&nbsp; 2. push $2 ~ k$ into the stack and then pop them in some unknown order
        <br>&nbsp;&nbsp;&nbsp;&nbsp; 3. pop $1$
        <br>&nbsp;&nbsp;&nbsp;&nbsp; 4. push $k+1 ~ n$ and then pop them in some unknown order
        Then we get the iterative formula:
        
        $$\begin{align*}
        S_n = \sum^{n}_{k=1}{S_{k-1} * S_{n-k}}
        \end{align*}$$

        The time complexity is $\mathcal{O}(n^2)$.
    
    ??? sl "**Solution 2** (DP)"
        Define $f[i,j]$ be the number of possibility that there are $i$ elements not pushed and $j$ elements left in the stack. Under any situation, there are two choices: push an element or pop an element. The Bellman Equation is:
        
        $$\begin{align*}
        f[i,j] = f[i-1,j+1] + f[i,j-1]
        \end{align*}$$

        The time complexity is $\mathcal{O}(n^2)$.
    
    ??? sl "**Solution 3** (Math)"
        This question is equivalent to calculating the nth Catalan number, i.e. 
        
        $$\begin{align*}
        \frac{C^n_{2n}}{n+1}
        \end{align*}$$

        The time complexity is $\mathcal{O}(n)$.
    
!!! df "**Definition** (Infix, Prefix, Postfix Notation)"
    _Infix Notation_: operators are in-between every pair of operands. e.g. $3 * (1 - 2)$ <br>
    _Prefix Notation_ (Polish Notation): operators are before two expressions. e.g. $* \; 3 - 1 2$ <br>
    _Postfix Notation_ (Reverse Polish Notation): operators are after two expressions. e.g. $1 2 - 3 \; *$

!!! im "**Important Note** (Computing Postfix Expression)"
    Use a stack. Scan the expression following these steps:
    <br>&nbsp;&nbsp;&nbsp;&nbsp; 1. if encounter a number, push it into the stack
    <br>&nbsp;&nbsp;&nbsp;&nbsp; 2. if encounter an operator, pop out two elements from the stack, calculate and push the result into the   stack
    <br> The time complexity is $\mathcal{O}(n)$.

!!! im "**Important Note** (Computing Infix Expression)"
    The fastest way to compute an infix expression is to first turn it into a postfix expression. We can do this following these steps:
    <br>&nbsp;&nbsp;&nbsp;&nbsp; 1. if encounter a number, output it
    <br>&nbsp;&nbsp;&nbsp;&nbsp; 2. if encounter a `(`, push it into the stack
    <br>&nbsp;&nbsp;&nbsp;&nbsp; 3. if encounter a `)`, pop and output elements until we popped a `(`
    <br>&nbsp;&nbsp;&nbsp;&nbsp; 4. if encounter an operator, pop and output elements until the priority of the operator > the element we intend to pop, then push the operator into the stack; The priority ranking is: `*/` > `+-` > `(`.
    <br> <br> After scanning all elements in the expression, we pop all the elements from the stack.

    The key idea here is to use stack to "wait" for `)` if encounter a `(`, or another number if encouter an operator.

## Monotonic Stack

!!! ex "**Exercise** (Largest Rectangle in a Histogram)"
    Find the maximum area of the rectangle that can be outlined in a histogram.

    <figure markdown="span">
    ![Image title](https://cdn.acwing.com/media/article/image/2019/01/14/19_eac6c46017-2559_1.jpg){ width="500" }
    </figure>

    ??? sl "**Solution**"
        Consider if the each small rectangle in the histogram is increasing height, then we can simply enumerate heights and ignore all the rectangle on the left for a certain height. When a shorter rectangle comes, we can first view all previous rectangles as increasing height histogram, and then ignore the heights that are larger than the new shorter rectangle. In other word, we are maintaining a _Monotonic Stack_. To implement this, we add a rectangle of height 0 at the end to activate final pop.

        ``` cpp linenums="1"
        a[n+1] = 0; // adding a 0 height rectangle at the end of the histogram
        position = 0; 
        for(int i=1;i<=n;i++){
            if(a[i]?stack[position]){
                // push a rectangle into the stack if the stack is monotonic
                stack[++p] = a[i]; 
                // the width of small rectangle can vary as 
                // it might be the combination of multiple rectangle
                width[position] = 1; 
            }else{
                int width_ = 0;
                while(stack[position] > a[i]){
                    // accumulate width while popping
                    width_ += width[position]; 
                    // calculate the area if using rectangle 'position' as height
                    ans = max(ans, width_ * stack[position]); 
                    p--; // pop
                }
                // push the new rectangle
                stack[++p] = a[i]; 
                // the new rectangle has the accumulated width + 1
                width[position] = width_ + 1; 
            }
        }
        ```

        This is the famous monotonic stack, with time complexity $\mathcal{O}(N)$. The key idea here is to erase impossible choices and maintain the set to be effective and in order.
    
    
    
    
    
    

        
        
        
        
    
        
        
     

