---
layout: distill
title: Permutation Symmetric Neural Nets from the Ground Up
description: Dealing with lists, matrices or symbols? When your problem has obvious permutation symmetries, carefully designed permutation symmetric neural networks can enable highly efficient learning, sometimes solving the task from just a few examples. In this post, we'll walk through a first principle derivation of permutation symmetric network parameterization using Taylor series, a practical network design using einsum pooling for various types of permutation symmetry, and from scratch implementations on several problems, including a Sudoku solver, a few ARC-AGI challenge problems with permutation symmetry and knowledge graph reasoning.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

#Anonymize when submitting
authors:
   - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-permutation-symmetric-nns.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: I. In Theory - Paramterizing Symmetric Functions in Taylor Series
  - name: II. Engineering a Network with Permutation Symmetry

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


## I. In Theory - Paramterizing Symmetric Functions in Taylor Series

Speaking of permutation invariance, you may already have your faviorite ways to design invariant neural networks for certain types of problems. 
But here we'll introduce a simple yet general Taylor series-based technique necessary for studying complex equivariance patterns. Specifically, we need to enable efficient universal learners -- that can represent any such invariant or equivariant functions. 
For advanced readers, the general idea follows [Equivariant Multilayer Perceptrons (EMLP)](https://github.com/mfinzi/equivariant-MLP) but with a high-order twist. 

Given the desired input-output shapes and symmetry constraints, we would proceed as the following: 
1) Express a general function that matches the input-output shapes in Taylor series form, 
2) Map the symmetry constraints into equations about the Taylor series coefficients,
3) Solve the equations for free parameters and the parameter sharing patterns, and parameterize the function using the free parameters,
4) Simplify the parameterization for efficient computation.

Let's start from 1D permutation invariance as an example to demonstrate how the technique works.


### I.1 Example: 1D permutation invariance

Let's say we want to make a function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)
$$
invariant to permutation of $x_0$, $x_1$, $x_2$.

Consider the Taylor series

$$
\begin{aligned}
f & \left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right) \\
= & a + 
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2}\end{bmatrix}
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \end{bmatrix}
+ \ldots
\end{aligned}
$$

Since we want $f(\cdot)$ to be invariant to any permutation matrix $P$, the invariant constraint says 

$$
f\left(\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix}\right)=f\left(
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} 
    \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix}\right)
$$

For our Taylor series form, because of the uniqueness of Taylor series, all order-k coefficients on the left hand side need to match the corresponding order-k coefficients on the right hand side. That is for any permutation matrix $P$ we have

$$
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix}  = &
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix} 
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    &   &  \\
    & P^T &  \\
    &   &  
\end{bmatrix} 
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} \\
& \ldots
\end{aligned}
$$

These equations are all linear equations about coefficients $a$, $b_i$ and $c_{ij}$. So we can just enumerate all $P$ to get all the equations, and then solve them. For $b_i$ for example, enumerating different permutations $P$ would give

$$
\begin{bmatrix} b_0 \\ b_1 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_0 \\ b_2 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_0 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_2 \\ b_0\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_0 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_1 \\ b_0\end{bmatrix} 
$$

That is more than enough to say $b_0=b_1=b_2$. So the order-1 term has only 1 degree of freedom.

For $c_i$ there are more equations, but it turns out that solving the equations across all permutations would yield 
$$c_{00}=c_{11}=c_{22}$$ and $$c_{01}=c_{10}=c_{10}=c_{12}=c_{20}=c_{21}$$. So the order 2 term has 2 degrees of freedom, one for the diagonal and one for everywhere else.

Applying what we have learned, we can now write

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{1}\\
    c_{1} & c_{0} & c_{1}\\
    c_{1} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \end{bmatrix}
+ \ldots
\end{aligned}
$$
For a total of 4 free parameters up to order 2, instead of 13 free parameters without the invariance constraint. More generally, for $N$ inputs, we still only need 4 parameters to express any permutation invariant function, whereas a non-invariant function needs $N^2+N+1$ parameters. In practice, parameterizing with symmetry often **reduces parameter count** exponentially. 

We can further simplify by focusing on the free parameters
$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)\\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 \sum_{i} \sum_{j} x_i x_j + \ldots \\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 (\sum_{i}x_i )^2 + \ldots
\end{aligned}
$$
An important effect of this simplification is **reduced compute**. It now requires $O(N)$ compute for $N$ inputs instead of $O(N^2)$ for order-2.  

In math terms, the number of free parameters is the dimensionality of the null space of the symmetry equations. The free parameters can be numerically solved from the basis of this null space which is one of the many innovations in [1]. But note that as the basis is often not unique, numerical solutions can vary by a linear combination and therefore may not be compute-optimal, so further simplification is still needed.

Although we didn't unroll order-3 and higher terms because they are difficult to visualize, they can still be analyzed with the same approach. Just imagine a cube or a hypercube of parameters, apply the symmetry transformations simultaneously along all dimensions and solve for the parameter sharing pattern.

### I.2 Exercises
If you are interested in going a little deeper, test yourself on the following list of exercises and gain new insights. Click to expand the reference solution. 

**A. 1D translation.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix} \right) =f\left( \begin{bmatrix} x_{3} & x_{0} & x_{1} & x_{2}\end{bmatrix} \right)
$$

<details>

<summary> 
Solution
</summary>


According to equivariant constraints, the coefficients of the Taylor series satisfy
$$
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix}  = &
\begin{bmatrix} b_1 & b_2 & b_3 & b_0\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{11} & c_{12} & c_{13} & c_{10} \\
    c_{21} & c_{22} & c_{23} & c_{20} \\
    c_{31} & c_{32} & c_{33} & c_{30} \\
    c_{01} & c_{02} & c_{03} & c_{00} \\
\end{bmatrix} 
& \ldots
\end{aligned}
$$
Which means there are 6 free parameters up to order-2.
$$
\begin{aligned}
b_0=b_1=&b_2=b_3 \\
c_{00}=c_{11}=&c_{22}=c_{33} \\
c_{01}=c_{12}=&c_{23}=c_{30} \\
c_{02}=c_{13}=&c_{20}=c_{31} \\
c_{03}=c_{10}=&c_{21}=c_{32} \\
\end{aligned}
$$
Considering Hessian transpose symmetry, we would additionally have $c_{01}=c_{03}$ and reduce number of free parameters to 5. For now, let us assume that multiply among $x_i$ does not commute.

The parameterization with 6 parameters has an unrolled circular convolution on the order-2 term.

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b &b\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2} & x_3\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{3} & c_{0} & c_{1} & c_{2}\\
    c_{2} & c_{3} & c_{0} & c_{1}\\
    c_{1} & c_{2} & c_{3} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3} \end{bmatrix}
+ \ldots
\end{aligned}
$$

Computing the 2nd order term naively would require $O(N^2)$ multiplies for length-$N$ input. But because the coefficients implement a circular convolution operation,  fast fourier transform would reduce compute complexity down to $N\log{N}$.


</details>


**B. Scale.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) =f\left(\alpha \begin{bmatrix}x_{1} & x_{2} & x_{0}\end{bmatrix} \right)
$$
For any $\alpha\ne0$.

<details>

<summary> 
Solution
</summary>

With Taylor series, you'll run into a conclusion that no terms could exist and $y=a$. That is because scale invariant functions are often not smooth at $x_i=0$ so Taylor series could not capture them. Let us instead look into Laurent Series

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) \\
=&\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} c_{ijk} x_0^i x_1^j x_2^k 
\end{aligned}
$$

Applying the invariant constraint

$$
\begin{aligned}
\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} c_{ijk} x_0^i x_1^j x_2^k 
=
\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} \alpha^{i+j+k} c_{ijk} x_0^i x_1^j x_2^k 
\end{aligned}
$$

This only holds when the coefficients match, that is for any $(i,j,k)$
$$
\begin{aligned}
c_{ijk}=\alpha^{i+j+k} c_{ijk}
\end{aligned}
$$
That is only terms with $i+j+k=0$ would have non-zero coefficients. For example, $\frac{xy}{z^2}$. Within terms up to order-2, that is $i,j,k\in \left\{ -2,-1,0,1,2 \right\}$, the degrees of freedom is $19$ out of $5^3=125$ as the following

| (i,j,k) | DoF |
|:-------:|:---:|
| 0,0,0   | 1   |
| -1,0,1  | 6   |
| -2,0,2  | 6   |
| -2,1,1  | 3   |
| 2,-1,-1 | 3   |

The full parameterization is
$$
\begin{aligned}
f&\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) \\
=&a + \sum_{i}\sum_{j\ne i} b_{ij} \frac{x_i}{x_j} + \sum_{i}\sum_{j\ne i} c_{ij} \frac{x_i^2}{x_j^2} + d_0 \frac{x_1 x_2}{x_0^2} + d_1 \frac{x_0 x_2}{x_1^2} + d_2 \frac{x_0 x_1}{x_2^2} + e_0 \frac{x_0^2}{x_1 x_2} + e_1 \frac{x_1^2}{x_0 x_2} + e_2 \frac{x_2^2}{x_0 x_1} 
\end{aligned}
$$

Nevertheless, for scale invariance it is easier to reparameterize the input with
$$
z_0=\frac{x_0}{\sqrt{x_0^2+x_1^2+x_2^2}} \quad
z_1=\frac{x_1}{\sqrt{x_0^2+x_1^2+x_2^2}} \quad
z_2=\frac{x_2}{\sqrt{x_0^2+x_1^2+x_2^2}}
$$
and express

$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) =g\left(z_1,z_2\right)
$$

</details>

**C. 1D permutation with latent.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} \\ x_{2} & x_{3}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{0} & x_{1} \\ x_{2} & x_{3}\end{bmatrix} \right)
$$

<details>

<summary> 
Solution
</summary>

According to the equivariant constraint, the coefficients of the Taylor series satisfy
$$
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix}  = &
\begin{bmatrix} b_2 & b_3 & b_0 & b_1\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{22} & c_{23} & c_{20} & c_{21} \\
    c_{32} & c_{33} & c_{30} & c_{31} \\
    c_{02} & c_{03} & c_{00} & c_{01} \\
    c_{12} & c_{13} & c_{10} & c_{11} \\
\end{bmatrix} 
\end{aligned}
$$
Solving the equations gives
$$
\begin{aligned}
\begin{bmatrix}b_0 & b_1\end{bmatrix}=&\begin{bmatrix}b_2 & b_3\end{bmatrix} \\

\begin{bmatrix} 
    c_{00} & c_{01} \\
    c_{10} & c_{11} \\
\end{bmatrix} 
=&
\begin{bmatrix} 
    c_{22} & c_{23} \\
    c_{32} & c_{33} \\
\end{bmatrix}  \\
\begin{bmatrix} 
    c_{02} & c_{03} \\
    c_{12} & c_{13} \\
\end{bmatrix} 
=&
\begin{bmatrix} 
    c_{20} & c_{21} \\
    c_{30} & c_{31} \\
\end{bmatrix} 
\end{aligned}
$$
If we view the rows of inputs as vectors, the coefficients can be partitioned into blocks that process those vectors, and the row-permutation invariant constraint leads to parameter sharing at the block level. We can parameterize

$$
\begin{aligned}
f&\left( \begin{bmatrix}x_{0} & x_{1}\end{bmatrix}, \begin{bmatrix}x_{2} & x_{3}\end{bmatrix} \right) \\
=&a + 
\begin{bmatrix}b_{0} & b_{1}\end{bmatrix}
\begin{bmatrix}x_{0}+x_2 \\ x_{1}+x_3\end{bmatrix}
+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}c_0 & c_1\\c_2 & c_3\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}c_0 & c_1\\c_2 & c_3\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}
\\
&+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}\\
=&a + 
\begin{bmatrix}b_{0} & b_{1}\end{bmatrix}
\begin{bmatrix}x_0+x_2 \\ x_1+x_3\end{bmatrix}
+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}c_0' & c_1'\\c_2' & c_3'\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}c_0' & c_1'\\c_2' & c_3'\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}\\
&+ 
\begin{bmatrix}x_0+x_2 & x_1+x_3\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_0+x_2 \\ x_1+x_3\end{bmatrix}
\end{aligned}
$$
The size of order-$k$ coefficient blocks for processing length-$H$ latent vectors is $H^k$. This is already much better than the full coefficients $(NH)^k$ for a set of $N$ vectors but is still large. Now, the bread and butter of deep learning comes in, namely 1) stacking more layers, 2) low-rank factorization and 3) non-linearities which we'll discuss more in Section II.

</details>

**D. 2D permutation.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) = f\left(\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix}\begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \right)
$$

<details>

<summary> 
Solution
</summary>

According to the equivariant constraint, the coefficients of the Taylor series satisfy
$$
\begin{aligned}
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix} = &
\begin{bmatrix} b_2 & b_3 & b_0 & b_1\end{bmatrix} =
\begin{bmatrix} b_1 & b_0 & b_3 & b_2\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{22} & c_{23} & c_{20} & c_{21} \\
    c_{32} & c_{33} & c_{30} & c_{31} \\
    c_{02} & c_{03} & c_{00} & c_{01} \\
    c_{12} & c_{13} & c_{10} & c_{11} \\
\end{bmatrix} 
=
\begin{bmatrix} 
    c_{11} & c_{10} & c_{13} & c_{12} \\
    c_{01} & c_{00} & c_{03} & c_{02} \\
    c_{31} & c_{30} & c_{33} & c_{32} \\
    c_{21} & c_{20} & c_{23} & c_{22} \\
\end{bmatrix} 
\end{aligned}
$$

Solving the equations gives the following parameterization with 6 degrees of freedom
$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b &b\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix} +
\begin{bmatrix} x_{00} & x_{01} & x_{10} & x_{11}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{1} & c_{0} & c_{3} & c_{2}\\
    c_{2} & c_{3} & c_{0} & c_{1}\\
    c_{3} & c_{2} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix}
+ \ldots
\end{aligned}
$$

Let us perform a bit of merging and simplification
$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + b \sum_i \sum_j x_{ij} + (c_0-c_1-c_2-c_3) \sum_i \sum_j x_{ij}^2\\
&+ (c_1-c_3) \sum_i (\sum_j x_{ij})^2+(c_2-c_3) \sum_j (\sum_i x_{ij})^2  + c_3 (\sum_i \sum_j x_{ij})^2
+ \ldots
\end{aligned}
$$

An interesting pattern emerges, that all terms are some forms of tensor contractions. In fact, this seems to be true for all flavors of permutation symmetry and the motivation behind Section II. Don't believe it? Try another case below!

</details>


**E. 2D joint permutation.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\right)
$$

<details>

<summary> 
Solution
</summary>

According to the equivariant constraint, the coefficients of the Taylor series satisfy
$$
\begin{aligned}
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix} = &
\begin{bmatrix} b_3 & b_2 & b_1 & b_0\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{33} & c_{32} & c_{31} & c_{30} \\
    c_{23} & c_{22} & c_{21} & c_{20} \\
    c_{13} & c_{12} & c_{11} & c_{10} \\
    c_{03} & c_{02} & c_{01} & c_{00} \\
\end{bmatrix} 
\end{aligned}
$$

Solving the equations gives the following parameterization with 11 degrees of freedom
$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b_0 & b_1 & b_1 &b_0\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix} +
\begin{bmatrix} x_{00} & x_{01} & x_{10} & x_{11}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{4} & c_{5} & c_{6} & c_{7}\\
    c_{7} & c_{6} & c_{5} & c_{4}\\
    c_{3} & c_{2} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix}
+ \ldots
\end{aligned}
$$
With Hessian transpose symmetry, we may further have $c_1=c_4$ and $c_2=c_7$ which reduces free parameters count down to 9, still 3 more than regular 2D permutation invariance. If you squint really hard (and maybe try Exercise D), there exists a tensor contraction form:
$$
\begin{aligned}
f&\left( \{x_{ij}\} \right)\\
= & a + b_0' \sum_i x_{ii} + b_1' \sum_i \sum_j x_{ij} + 
c_0' \sum_i x_{ii}^2 
+ c_1' \sum_i x_{ii}\sum_j x_{ij} \\
&+ c_2' \sum_i x_{ii}\sum_j x_{ji} 
+ c_3' \sum_i x_{ii}\sum_j x_{jj}
+ c_5' \sum_i \sum_j x_{ij}^2
+ c_6' \sum_i \sum_j x_{ij} x_{ji}  
+ \ldots
\end{aligned}
$$
What's different from regular 2D permutation invariance are terms involving diagonal and transpose. Also all tensor contractions here are at or below $O(N)$ compute for input size $\sqrt{N}\times \sqrt{N}$, which is exponentially less compute than $O(N^2)$ for the default Taylor series.

</details>


**F. 1D permutation equivariance.** Parameterize function 
$$
\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) =F\left( \begin{bmatrix}&&\\&P&\\&&\end{bmatrix}\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right)
$$
For any permutation $P$.


<details>

<summary> 
Solution
</summary>

The Taylor series up to order 1 can be expressed as
$$
\begin{aligned}
&\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) \\
&=\begin{bmatrix}a_0 \\ a_1 \\a_2\end{bmatrix}
+ 
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} 
+
\ldots
\end{aligned}
$$
The equivariant constraints are for any $P$
$$
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
=
\begin{bmatrix}
 & & \\ 
 & P^T & \\ 
 & & \\ 
\end{bmatrix}
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
\begin{bmatrix}
 & & \\ 
 & P & \\ 
 & & \\ 
\end{bmatrix}
$$
Which is identical to the invariant constraints on order-2 terms. In general, the parameterization of an equivariant function up to order-k is very much the same as an invariant function up to order-(k+1). In the case of 1D permutation equivariance, the order-1 parameterization would be

$$
\begin{aligned}
&\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) \\
&=\begin{bmatrix}a \\ a \\a\end{bmatrix}
+ 
\begin{bmatrix}
b_{0} & b_{1} & b_{1} \\ 
b_{1} & b_{0} & b_{1} \\
b_{1} & b_{1} & b_{0}
\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} 
+
\ldots
\end{aligned}
$$
Rewriting in tensor contraction form using [einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) notations

$$
\begin{aligned}
Y=&F\left( X \right) \\
=& a \cdot \text{einsum(`i->i',X)} 
+ (b_0-b_1) \cdot \text{einsum(`i,i->i',X,X)} \\
&+ b_1 \cdot \text{einsum(`i,j->i',X,X)} 
\end{aligned} 
$$


</details>


### I.3 What we have learned so far

In this section, we have learned that
1) Symmetry constraints reduce the number of free parameters.
2) A Taylor-series technique can be used to parameterize symmetric functions.
3) Different symmetries can have different impacts on degrees of freedom.
4) Certain parameterizations can reduce compute.
5) Parameterization of equivariant functions are tied to parameterization of invariant functions
6) Permutation invariant and equivariant functions can be parameterized solely using tensor contraction terms.

A Taylor series parameterization is sound in theory. In practice however, functions compound and high order interactions are common. Taylor series often provides too little relevant capacity and too much irrelevant capacity to be useful. Engineering is key in the journey to create universal learners of equivariant functions. In the next section, we'll focus on permutation symmetry and design a family of practical invariant and equivariant networks for various flavors of permutation symmetry.


## II Engineering a Network with Permutation Symmetry

From matrices to sets to symbolic processing, permutation symmetry is found in many problems and requires extra attention during modeling. When handled properly however, permutation symmetry is also a blessing. As we have learned in the previous section, if parameterized properly, permutation symmetry has the potential to exponentially reduce parameter count and compute for highly efficient learning. At the other end of the spectrum, reciting the success recipe of deep learning, we can scale the latent dimension and stack equivariant layers to create exponentially more expressive networks at the same parameter count and compute as a regular network.

![Network with Equivariant Backbone](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png)


Devil's in the details, in this section we'll walk through the design of permutation symmetric neural networks for various types of permutation symmetry.


There are many places where you'll see permutation symmetry and they often come in different forms. So in Section II.1 we'll first start from a summary of common types of permutation symmetry. 
And then Section II.2 will discuss the design of permutation equivariant layers. Permutation symmetry turns out to be closely tied to tensor contractions. That would allow us to synthesize efficient high-order permutation equivariant layers automatically in a procedural manner.
And finally in Section II.3 we discuss further optimizations that helps practical implementation. 

### II.1 Common types of permutation symmetry

In the following table we list a few common problems with different types of permutation symmetry.

| Problem       | Illustration  | Symmetry type  | Dependency |
| ------------- |:-------------:|:--------------:|:---------- |
|               |               |                |            |

Multiple dimensions, joint permutation and dependency are common themes here. To aid discussions, we also use a custom notation to describe the specific type of permutation symmetry, to capture both the input shape and the unique permutation axes. A fully independent batch dimension Z and a non-symmetric latent dimension H may be added optionally.


### II.2 Creating a permutation equivariant layer with einsum pooling

Across all types of permutation symmetry, as we learned in Section I through Taylor series, it turns out that tensor contractions are are all you need for parameterizing permutation invariant and equivariant layers, which can then be stacked into a deep network.

**How to create an equivariant layer given permutation symmetry type?**
Our answer is two fully connected layers with a pooling layer in between. 

<details>

<summary>
Primer: Tensor contractions and the einsum notation
</summary> 

Intuitively, tensor contractions like
$$
Y_{ij}=\sum_k \sum_l X_{ik} X_{lk} X_{lj} 
$$
create a new tensor that has the same shape as the input while summing over unused dimensions. They achieve a permutation equivariant effect. And tensor contractions like
$$
y=\sum_i \sum_j \sum_k \sum_l X_{ik} X_{lk} X_{lj} 
$$
that sum over all dimensions achieve a permutation invariant effect. 

As the math equations can get quite long, we will use the [einsum notation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) which represents a tensor contraction using the indices involved. It is widely used across deep learning frameworks to denote tensor contractions. For example,
```python
Y=einsum('Zik,Zlk,Zlj->Zij',X,X,X)
y=einsum('Zik,Zlk,Zlj->Z',X,X,X)
```
Here a batch dimension Z is added to make sure the right hand side is not empty.  

</details>  


Let us use a 1D + latent `aH`-type equivariant constraint as an example to illustrate the design.

The Taylor Series parameterization up to order 2 is
```python
Y_abH=einsum('a->ba',a_H)
     +einsum('ab,ca->cb',b0_HH,X_aH)
     +einsum('ab,ca->db',b1_HH,X_aH)
     +einsum('abc,da,db->dc',c0_HHH,X_aH,X_aH)
     +einsum('abc,da,db->ec',c1_HHH,X_aH,X_aH)
     +einsum('abc,da,eb->dc',c2_HHH,X_aH,X_aH)
     +einsum('abc,da,eb->fc',c3_HHH,X_aH,X_aH)
     +...
```

We can immediately see that the order-1 terms have $H^2$ parameters and order-2 terms have $H^3$ parameters, which would naturally need a low-rank($=K$) treatment, such as

```python
Y_abH=einsum('a->ba',a_H)
     +einsum('ka,kb,ca->cb',b0U_KH,b0V_KH,X_aH)
     +einsum('ka,kb,ca->db',b1U_KH,b1V_KH,X_aH)
     +einsum('ka,kb,kc,da,db->dc',c0U,c0V,c0W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,db->ec',c1U,c1V,c1W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,eb->dc',c2U,c2V,c2W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,eb->fc',c3U,c3V,c3W,X_aH,X_aH)
     +...
```

We can move the order-0 parameters, as well as $U$, $V$, $W$ matrices into fully connected layers along $H$ that perform input preprocessing and output post_processing. So the end result is two linear layers with pooling in between, and for pooling we need 

```python
Y_abH_0V=einsum('ck->ck',X_aH_0U)
Y_abH_1V=einsum('ck->dk',X_aH_1U)
Y_abH_0W=einsum('dk,dk->dk',X_aH_0U,X_aH_0V)
Y_abH_1W=einsum('dk,dk->ek',X_aH_1U,X_aH_1V)
Y_abH_2W=einsum('dk,ek->dk',X_aH_2U,X_aH_2V)
Y_abH_3W=einsum('dk,ek->fk',X_aH_3U,X_aH_3V)
...
```

Notice that `'dk,ek->fk'` can be composed with `'ck->dk'` for each operand individually, and then combine using `'dk,dk->dk'`. As we can stack more layers, not all pooling operations are needed and less pooling operations would reduce network complexity. In fact, this might be a good point to step back and ask: **Given equivariance type, e.g. `aH`, how can we identify the minimum yet sufficient set of pooling operations?**

The following recipe might be helpful for designing pooling operations given equivariance type in practice:

1) Enumerate all valid and unique einsum operations up to order-k that are compatible with the given equivariance type. For example `einsum('ab,bc->ac',X_ab,X_ab)` is compatible with `aa`-type equivariance, but not compatible with `ab`-type equivariance. Also notice that `ba,ac->bc` is the just a renaming of `ab,bc->ac`. There is a graph homomorphism problem under the hood for listing unique einsum operations and interested readers can dig deeper.

2) Filter einsum operations based on dependency requirement of the given equivariance type. For example `einsum('ab,cb->cb',X_ab,X_ab)` satisfy `b->a` dependency but does not satisfy `a->b` dependency for `ab`-type equivariance.

3) Filter out order-2+ "breakable" operations that can be divided into two lower order terms with a simple pointwise multiplication. For example `ab,cb,cd,ad->ad` can be divided into `ab,cb,cd->ad` and `ad->ad` which can the be put together with `ab,ab->ab`, so it is not necessary as long as the lower order terms exist. 

4) Normalize the rotation of input/output terms. For example for `aa`-type equivariance, `ab,cb->ca` is not necessary because it can be achieved with `ab,bc->ac`, through applying rotations `ab->ba` on the input and output.

5) Remove order-2+ operations that expand new dimensions in the output term. For example `ab,bc->ad` is redundant because it can be achieved through `ab,bc->ac` followed by a dimension expansion operation `ab->ac`.

An algorithm that properly de-duplicates through compositions remains to be developed. But after all the filtering listed here, there is usually a quite compact initial set of pooling operations for further optimizations.

The following is a quick lookup table of pooling operations for a few common equivariance types.


| Symmetry type | Order | Pooling operation(s) |
| ------------- |-------|:------------------:|
| `aH`          | 1     | `aH->aH`, `aH->bH` |
|               | 2     | `aH,aH->aH`  |
|               | 3+    | No need  |
| `abH`         | 1     | `abH->abH`, `abH->cbH`, `abH->acH` |
|               | 2     | `abH,abH->abH` |
|               | 3     | `abH,cbH,cdH->adH` |
|               | 4     | No need |
|               | 5     | `abH,acH,dbH,dcH,deH->aeH`, `abH,acH,dbH,dcH,ecH->ebH` |
| `aaH`         | 1     | `abH->abH`,`aaH->aaH`,`abH->baH`,`abH->cbH` |
|               | 2     | `abH,bcH->acH`,`abH,abH->abH` |
|               | 3+    | No need  |


*Did you know: `'ab,cb,cd->ad'` which is linear self-attention is an order-3 term for `ab`-type equivariance. Self-attention operation by itself is equivariant to not only token permutation but also latent permutation, although other linear layers in the transformer architecture do not retain the latent symmetry.*


### II.3 Putting everything together: The Equivariant Einsum Network

The following diagram shows the final network design.

![Complete design](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png)

In addition to layer stacking of equivariant layesr, the following features are further added:

1) Softmax normalization before einsum pooling to prevent high order einsum from exploding.
2) GELU nonlinearity between equivariant layers to add to the depth and create bottlenecks.
3) Residual connections for better optimization dynamics

Another consideration in practice is [einsum path optimization](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html). For example, the einsum string `ab,dc,ae,ac,db->de` by default is programmed to be computed pairwise from left to right. By the third term, a large factor `abcde` would be created and stress the memory. Instead, if we compute pairwise via path `ab,db->ad`, `ac,dc->ad`, `ad,ad->ad` and `ad,ae->de`, the largest intermediate factor would only be 2-dimensional and the computation can also be done much faster. For modeling complex higher-order interations under certain types of symmetries, large einsums may be unavoidable, and computing them might be an interesting compute challenge.

Putting it all together, here is a reference Pytorch implementation of an Equivariant Einsum Network that would served as the backbone, to be followed by averaging for dimensions that need invariance. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#Implements minimal aH-type pooling
class einpool_a(nn.Module):
    fan_in=4
    fan_out=3
    ndims=1
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-2:]) # Apply pooling only to the last 2 dims, supposedly `aH`
        N,KH=x.shape[-3:]
        H=KH//self.fan_in
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].sum(-2,keepdim=True).repeat(1,N,1)
        y2=x[2]*x[3]
        y=torch.cat((y0,y1,y2),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y

#Implements minimal aaH-type pooling
class einpool_aa(nn.Module):
    fan_in=8
    fan_out=6
    ndims=2
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-3:]) # Apply pooling only to the last 3 dims, supposedly `aaH`
        N,M,KH=x.shape[-3:]
        H=KH//self.fan_in
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].diagonal(dim1=-2,dim2=-3).diag_embed(dim1=-2,dim2=-3)
        y2=x[2].transpose(-2,-3)
        y3=x[3].sum(-2,keepdim=True).repeat(1,1,M,1)
        y4=x[4]*x[5]
        y5=torch.einsum('ZabH,ZbcH->ZacH',x[6],x[7])
        y=torch.cat((y0,y1,y2,y3,y4,y5),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y

#Implements order-3 abH-type pooling
class einpool_ab(nn.Module):
    fan_in=8
    fan_out=5
    ndims=2
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-3:]) # Apply pooling only to the last 3 dims, supposedly `abH`
        N,M,KH=x.shape[-3:]
        H=KH//self.fan_in
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].sum(-2,keepdim=True).repeat(1,1,M,1)
        y2=x[2].sum(-3,keepdim=True).repeat(1,N,1,1)
        y3=x[3]*x[4]
        y4=torch.einsum('ZacH,ZbcH,ZadH->ZbdH',x[5],x[6],x[7])
        y=torch.cat((y0,y1,y2,y3,y4),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y

#2-layer mlp with GELU
def mlp2(ninput,nh,noutput):
    return nn.Sequential(nn.Linear(ninput,nh),nn.GELU(),nn.Linear(nh,noutput))

#Equivariant EinNet backbone
class einnet(nn.Module):
    #Instantiate the network
    #    ninput/noutput -- number of input/output dimensions
    #    nh0 -- pooling dimensions, like head_dim in transformers
    #    nh -- latent dimensions 
    #    nstacks -- number of einsum pooling stacks
    #    pool -- einsum pooling operation. Needs to provide fan_in, 
    #            fan_out factors, and ndims
    def __init__(self,ninput,nh0,nh,noutput,nstacks,pool):
        super().__init__()
        self.t=nn.ModuleList()
        self.t.append(mlp2(ninput,nh,nh0*pool.fan_in))
        for i in range(nstacks-1):
            self.t.append(mlp2(nh0*pool.fan_out,nh,nh0*pool.fan_in))
        
        self.t.append(mlp2(nh0*pool.fan_out,nh,noutput))
        self.pool=pool
    
    # Forward call
    #    x: tensor shape matches equivariance type, e.g. *abH
    def forward(self,x):
        h=self.t[0](x)
        for i in range(1,len(self.t)):
            hi=F.softmax(h.view(*h.shape[:-self.pool.ndims-1],-1,h.shape[-1]),dim=-2).view(*h.shape)
            hi=self.t[i](self.pool(hi))
            #Residual connection
            if i<len(self.t)-1:
                h=h+hi
            else:
                h=hi
        
        return h

#Example usage
#    net=einnet(1,16,64,1,2,einpool_ab())
```
