
# Matchings in Graph

<figure markdown="span">
  ![Image title](https://pandanotes-bucket.s3.amazonaws.com/cs251s24/chapter_files/Matchings_in_Graphs/chapter-cover-matchings_ba551327-42a4-470f-82_GWyP82w.jpg){ width="300" }
</figure>
<br> <br>
In this chapter, we continue our discussion on graphs and turn our attention to finding matchings in graphs. Algorithms to find matchings are used a lot in real-world applications, and we discuss some of these applications in lecture. This chapter will expand your toolkit for reasoning about graphs and help you build more intuition about them.

## 1&nbsp;&nbsp;&nbsp;Maximum Matchings
!!! def "**Definition** (Matching – maximum, maximal, perfect)"
    A matching in a graph $G=(V,E)$ is a subset of the edges that do not share an endpoint. A maximum matching in $G$ is a matching with the maximum number of edges among all possible matchings. A maximal matching is a matching with the property that if we add any other edge to the matching, it is no longer a matching.(1) A *perfect matching* is a matching that covers all the vertices of the graph. 

??? eg "**Example** (Examples of matchings)"
    Consider the following graph.
    <figure markdown="span">
    ![Image title](https://pandanotes-bucket.s3.amazonaws.com/cs251s24/chapter_files/Matchings_in_Graphs/matchings-example_3d0e2e77-c246-4b32-8c3c-a1c_6fAOxJH.png){ width="300" }
    </figure> 
    Note that the empty set and a set with only one edge is always a matching. The set $M = \{\{v_1, v_5\}, \{v_4,v_7\}\}$ is a maximal matching with 2 edges, since we if we tried to add another edge to this set, it would no longer be a matching. On the other hand, this maximal matching is not a maximum matching because there is another matching with 3 edges: $M' = \{ \{v_1,v_6\}, \{v_3,v_5\}, \{v_4,v_7\} \}$. This graph does not have a perfect matching. One easy way to see this is that it has an odd number of vertices, and any graph with an odd number of vertices cannot have a perfect matching.

!!! imnote "**Note** (Size of a matching)"
    The size of a matching $M$ refers to the number of edges in the matching, and is denoted by $|M|$. Note that this coincides with the size of the set that $M$ represents.

!!! thm "**Theorem** (Handshake Lemma)"
    handshake lemma.

??? ex "**Exercise**"
    1 + 1 = ?
    ??? sol "**Solution**"
        Consider the following graph.
        <figure markdown="span">
        ![Image title](https://pandanotes-bucket.s3.amazonaws.com/cs251s24/chapter_files/Matchings_in_Graphs/matchings-example_3d0e2e77-c246-4b32-8c3c-a1c_6fAOxJH.png){ width="300" }
        </figure> 
        Note that the empty set and a set with only one edge is always a matching. The set $M = \{\{v_1, v_5\}, \{v_4,v_7\}\}$ is a maximal matching with 2 edges, since we if we tried to add another edge to this set, it would no longer be a matching. On the other hand, this maximal matching is not a maximum matching because there is another matching with 3 edges: $M' = \{ \{v_1,v_6\}, \{v_3,v_5\}, \{v_4,v_7\} \}$. This graph does not have a perfect matching. One easy way to see this is that it has an odd number of vertices, and any graph with an odd number of vertices cannot have a perfect matching.
