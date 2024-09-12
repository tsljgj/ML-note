# Introduction to Probability Theory

!!! df "**Definition** (Event Space $\mathcal{F}$)"
    The collection $\mathcal{F}$ of subsets of the sample space $\Omega$ is called an _**event space**_ if 
    
    $$\begin{align*}
    & \mathcal{F} \text{ is non-empty},\tag{1} \\
    & \text{if } A \in \mathcal{F} \text{ then } \Omega \setminus A \in \mathcal{F}, \tag{2} \\
    & \text{if } A_1, A_2, \dots \in \mathcal{F} \text{ then } \bigcup_{i=1}^{\infty} A_i \in \mathcal{F}. \tag{3}
    \end{align*}$$

    i.e. $\mathcal{F}$ is closed under the operations of taking complements and countable unions.

!!! rm "**Remark** (Intuitive Understanding of Event Space)"
    We can call a non-empty set $\mathcal{F}$ *Event Space* as long as any event $A$ in $\mathcal{F}$ satisfies: <br>
    
    &nbsp;&nbsp;&nbsp;&nbsp;1. $A$ not happening is also an event in $\mathcal{F}$ by the axiom (2). <br>
    &nbsp;&nbsp;&nbsp;&nbsp;2. Either $A$ or any other event $B \in \mathcal{F}$ happening is also an event in $\mathcal{F}$ by the axiom (3).

!!! nt "**Note** (Properties of Event Space $\mathcal{F}$)"
    (a) An event space $\mathcal{F}$ must contain the empty set $\varnothing$ and the whole set $\Omega$. 
    ??? pf "**Proof**"
        By (1), there exists some $A \in \mathcal{F}$. By (2), $A^c \in \mathcal{F}$. We set $A_1 = A$, $A_i = A^c$ for $i \geq 2$ in (3), and deduce that $\mathcal{F}$ contains the union $\Omega = A \cup A^c$. By (2) again, the complement $\Omega \setminus \Omega = \varnothing$ lies in $\mathcal{F}$ also.

    (b) An event space is closed under the operation of *finite* unions.
    ??? pf "**Proof**"
        Let $A_1, A_2, \dots, A_m \in \mathcal{F}$, and set $A_i = \varnothing$ for $i > m$. Then $A := \bigcup_{i=1}^{m} A_i$ satisfies $A = \bigcup_{i=1}^{\infty} A_i$, so that $A \in \mathcal{F}$ by (3).

    &#40;c&#41; An event space is also closed under the operations of taking finite or countable *intersections*. 
    ??? pf "**Proof**"
        Note that $(A \cap B)^c = A^c \cup B^c$.
    
??? eg "**Example** (Examples of Event Space $\mathcal{F}$)"
    **Example 1** <br> $\Omega$ is any non-empty set and $\mathcal{F}$ is the power set of $\Omega$. △

    **Example 2** <br> $\Omega$ is any non-empty set and $\mathcal{F} = \{\varnothing, A, \Omega \setminus A, \Omega\}$, where $A$ is a given non-trivial subset of $\Omega$. △

    **Example 3** <br> $\Omega = \{1, 2, 3, 4, 5, 6\}$ and $\mathcal{F}$ is the collection 
    $\{\varnothing, \{1, 2\}, \{3, 4\}, \{5, 6\}, \{1, 2, 3, 4\}, \{3, 4, 5, 6\}, \{1, 2, 5, 6\}, \Omega\}$ of subsets of $\Omega$.   △

!!! df "**Definition** (Probability Measure)"
    A mapping $\mathbb{P} : \mathcal{F} \rightarrow \mathbb{R}$ is called a _**probability measure**_ on $(\Omega, \mathcal{F})$ if

    (a) $\mathbb{P}(A) \geq 0$ for $A \in \mathcal{F}$,  
    (b) $\mathbb{P}(\Omega) = 1$ and $\mathbb{P}(\varnothing) = 0$,  
    &#40;c&#41; if $A_1, A_2, \dots$ are disjoint events in $\mathcal{F}$ (in that $A_i \cap A_j = \varnothing$ whenever $i \neq j$) then  
    
    $$\mathbb{P}\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \mathbb{P}(A_i). \tag{4}$$ 

!!! nt "**Note** (Why $\mathbb{P}(\varnothing) = 0$ ?)"
    Define the disjoint events $A_1 = \Omega$, $A_i = \varnothing$ for $i \geq 2$. By condition &#40;c&#41;,

    $$
    \mathbb{P}(\Omega) = \mathbb{P}\left(\bigcup_{i=1}^{\infty} A_i\right) = \mathbb{P}(\Omega) + \sum_{i=2}^{\infty} \mathbb{P}(\varnothing).
    $$

!!! df "**Definition** (Probability Space)"
    A *probability space* is a triple $(\Omega, \mathcal{F}, \mathbb{P})$ of objects such that

    &#40;a&#41; $\Omega$ is a non-empty set,  
    &#40;b&#41; $\mathcal{F}$ is an event space of subsets of $\Omega$,  
    &#40;c&#41; $\mathbb{P}$ is a probability measure on $(\Omega, \mathcal{F})$.

!!! im "**Important Note** (Properties of Probability Space)"
    **Property 1**  If $A, B \in \mathcal{F}$, then $A \setminus B \in \mathcal{F}$. 

    ??? pf "**Proof**"
        The complement of $A \setminus B$ equals $(\Omega \setminus A) \cup B$, which is the union of events and is therefore an event. Hence $A \setminus B$ is an event. □
    
    **Property 2**  If $A, B \in \mathcal{F}$, then $\mathbb{P}(A \cup B) + \mathbb{P}(A \cap B) = \mathbb{P}(A) + \mathbb{P}(B)$.

    ??? pf "**Proof**"
        The set $A$ is the union of the disjoint sets $A \setminus B$ and $A \cap B$, and hence

        $$\mathbb{P}(A) = \mathbb{P}(A \setminus B) + \mathbb{P}(A \cap B) \quad \text{IMPORTANT}.$$

        A similar remark holds for the set $B$, giving that

        $$\begin{aligned}
        \mathbb{P}(A) + \mathbb{P}(B) &= \mathbb{P}(A \setminus B) + 2\mathbb{P}(A \cap B) + \mathbb{P}(B \setminus A) \\
        &= \mathbb{P}((A \setminus B) \cup (A \cap B)) \cup (B \setminus A)) + \mathbb{P}(A \cap B) \\
        &= \mathbb{P}(A \cup B) + \mathbb{P}(A \cap B).
        \end{aligned}$$

    **Property 3** If $A_1, A_2, \dots \in \mathcal{F}$, then $\bigcap_{i=1}^{\infty} A_i \in \mathcal{F}$.

    ??? pf "**Proof**"
        The complement of $\bigcap_{i=1}^{\infty} A_i$ equals $\bigcup_{i=1}^{\infty} (\Omega \setminus A_i)$, which is the union of the complements of events and is therefore an event. Hence the intersection of the $A_i$ is an event also, as before. □
    
    **Property 4** If $A, B \in \mathcal{F}$ and $A \subseteq B$ then $\mathbb{P}(A) \leq \mathbb{P}(B)$.

    ??? pf "**Proof**"
        We have that $\mathbb{P}(B) = \mathbb{P}(A) + \mathbb{P}(B \setminus A) \geq \mathbb{P}(A)$. □

!!! df "**Definition** (Conditional Probability)"
    If $A, B \in \mathcal{F}$ and $\mathbb{P}(B) > 0$, the _**conditional probability**_ of $A$ given $B$ is denoted by $\mathbb{P}(A \mid B)$ and defined by

    $$
    \mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}. \tag{5}
    $$

!!! rm "**Remark**"
    Formula (5) is a **definition** rather than a theorem. An intuition to define conditional probability is that $\mathbb{P}(A \mid A) = 1$. (Draw Venn Diagram)

!!! tm "**Theorem**"
    If $B \in \mathcal{F}$ and $\mathbb{P}(B) > 0$ then $(\Omega, \mathcal{F}, \mathbb{Q})$ is a probability space where $\mathbb{Q} : \mathcal{F} \rightarrow \mathbb{R}$ is defined by $\mathbb{Q}(A) = \mathbb{P}(A \mid B)$.

    ??? pf "**Proof**"
        We need only check that $\mathbb{Q}$ is a probability measure on $(\Omega, \mathcal{F})$. Certainly $\mathbb{Q}(A) \geq 0$ for $A \in \mathcal{F}$ and

        $$
        \mathbb{Q}(\Omega) = \mathbb{P}(\Omega \mid B) = \frac{\mathbb{P}(\Omega \cap B)}{\mathbb{P}(B)} = 1,
        $$

        and it remains to check that $\mathbb{Q}$ satisfies &#40;4&#41;. Suppose that $A_1, A_2, \dots$ are disjoint events in $\mathcal{F}$. Then 
        
        $$\begin{align*}
        \mathbb{Q}\left(\bigcup_i A_i\right) &= \frac{1}{\mathbb{P}(B)} \mathbb{P}\left(\left(\bigcup_i A_i\right) \cap B\right) \\
                                             &= \frac{1}{\mathbb{P}(B)} \mathbb{P}\left(\bigcup_i (A_i \cap B)\right) \\
                                             &= \frac{1}{\mathbb{P}(B)} \sum_i \mathbb{P}(A_i \cap B) \quad \text{since $\mathbb{P}$ satisfies &#40;4&#41;} \\
                                             &= \sum_i \mathbb{Q}(A_i).
        \end{align*}$$

        Therefore, $\mathbb{Q}$ is a probability measure on $(\Omega, \mathcal{F})$. □

!!! df "**Definition** (Naive Definition of Independent Events)"
    We call two events $A$ and $B$ *independent* if the occurrence of one of them does not affect the probability that the other occurs. More formally, we **define**: <br>
    
    if $\mathbb{P}(A), \mathbb{P}(B) > 0$, then

    $$\mathbb{P}(A \mid B) = \mathbb{P}(A) \quad \text{and} \quad \mathbb{P}(B \mid A) = \mathbb{P}(B). \tag{6}$$

!!! df "**Definition** (Generalized Definition of Two Independent Events)"
    Writing $\mathbb{P}(A \mid B) = \mathbb{P}(A \cap B)/\mathbb{P}(B)$, we have:

    Events $A$ and $B$ of a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ are called *independent* if

    $$
    \mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B), \tag{7}
    $$

    and *dependent* otherwise.

    This definition is slightly more general since it allows the events $A$ and $B$ to have zero probability. 

It is easily generalized as follows to more than two events.:

!!! df "**Definition** (Definition of Independent Events)"
    A family $\mathcal{A} = (A_i \mid i \in I)$ of events is called _**independent**_ if, for all finite subsets $J$ of $I$,

    $$
    \mathbb{P}\left(\bigcap_{i \in J} A_i\right) = \prod_{i \in J} \mathbb{P}(A_i). \tag{8}
    $$

!!! df "**Definition** (Pairwise Independent)"
    The family $\mathcal{A}$ is called _**pairwise independent**_ if &#40;4&#41; holds whenever $|J| = 2$. 

There are families of events which are pairwise independent but not independent.

??? eg "**Example** (Pairwise Independent but Dependent)"
    Suppose that we throw a fair four-sided die (you may think of this as a square die thrown in a two-dimensional universe). We may take $\Omega = \{1, 2, 3, 4\}$, where each $\omega \in \Omega$ is equally likely to occur. The events $A = \{1, 2\}$, $B = \{1, 3\}$, $C = \{1, 4\}$ are pairwise independent but not independent.

!!! tm "**Theorem** (Partition Theorem)"
    If $\{B_1, B_2, \dots \}$ is a partition of $\Omega$ with $\mathbb{P}(B_i) > 0$ for each $i$, then

    $$
    \mathbb{P}(A) = \sum_i \mathbb{P}(A \mid B_i)\mathbb{P}(B_i) \quad \text{for } A \in \mathcal{F}.
    $$

    ??? pf "**Proof**"
        We have that

        $$\begin{align*}
        \mathbb{P}(A) &= \sum_i \mathbb{P}(A \cap B_i) \tag{Venn Diagram} \\
                      &= \sum_i \mathbb{P}(A \mid B_i)\mathbb{P}(B_i).
        \end{align*}$$
    
!!! rm "**Remark** (Intuition of Partition Theorem)"
    The partition theorem is saying if we have a set of event that partition $\Omega$, we can use it to crop an arbitrary event $E$ into several pieces, and the combination of all those pieces is exactly $E$.
    
!!! ex "**Exercise** (Application of Partition Theorem)"
    Tomorrow there will be either rain or snow but not both; the probability of rain is $\frac{2}{5}$ and the probability of snow is $\frac{3}{5}$. If it rains, the probability that I will be late for my lecture is $\frac{1}{5}$, while the corresponding probability in the event of snow is $\frac{3}{5}$. What is the probability that I will be late?

    ??? sl "**Solution**"
        Let $A$ be the event that I am late and $B$ be the event that it rains. The pair $B, B^c$ is a partition of the sample space (since exactly one of them must occur). By Partition Theorem,

        $$\begin{align*}
        \mathbb{P}(A) &= \mathbb{P}(A \mid B)\mathbb{P}(B) + \mathbb{P}(A \mid B^c)\mathbb{P}(B^c) \\
                    &= \frac{1}{5} \cdot \frac{2}{5} + \frac{3}{5} \cdot \frac{3}{5} = \frac{11}{25}.
        \end{align*}$$

!!! tm "**Theorem** (Bayes' Theorem)"
    Let $\{B_1, B_2, \dots\}$ be a partition of the sample space $\Omega$ such that $\mathbb{P}(B_i) > 0$ for each $i$. For any event $A$ with $\mathbb{P}(A) > 0$,

    $$
    \mathbb{P}(B_j \mid A) = \frac{\mathbb{P}(A \mid B_j)\mathbb{P}(B_j)}{\sum_i \mathbb{P}(A \mid B_i)\mathbb{P}(B_i)}.
    $$

    ??? pf "**Proof**"
        By the definition of conditional probability,

        $$\begin{align*}
        \mathbb{P}(B_j \mid A) &= \frac{\mathbb{P}(A \mid B_j)\mathbb{P}(B_j)}{\mathbb{P}(A)}, \\
                               &= \frac{\mathbb{P}(A \mid B_j)\mathbb{P}(B_j)}{\sum_i \mathbb{P}(A \mid B_i)\mathbb{P}(B_i)} \tag{Partition Theorem}
        \end{align*}$$

!!! ex "**Exercise** (False Positives)"
    A rare but potentially fatal disease has an incidence of 1 in $10^5$ in the population at large. There is a diagnostic test, but it is imperfect. If you have the disease, the test is positive with probability $\frac{9}{10}$; if you do not, the test is positive with probability $\frac{1}{20}$. Your test result is positive. What is the probability that you have the disease?

    ??? sl "**Solution**"
        Write $D$ for the event that you have the disease, and $P$ for the event that the test is positive. By Bayes' theorem, 

        $$\begin{align*}
        \mathbb{P}(D \mid P) &= \frac{\mathbb{P}(P \mid D)\mathbb{P}(D)}{\mathbb{P}(P \mid D)\mathbb{P}(D) + \mathbb{P}(P \mid D^c)\mathbb{P}(D^c)} \\
                            &= \frac{\frac{9}{10} \cdot \frac{1}{10^5}}{\frac{9}{10} \cdot \frac{1}{10^5} + \frac{1}{20} \cdot \frac{10^5-1}{10^5}} \approx 0.0002.
        \end{align*}$$