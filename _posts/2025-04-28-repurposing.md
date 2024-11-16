---
layout: distill
title: "Repurposing in AI: A Distinct Approach or an Extension of Creative Problem Solving?"
description: "Creativity is defined as the ability to produce novel, useful, and surprising ideas. A sub area of creativity is creative problem solving, the capacity of an agent to discover novel and previously unseen ways to accomplish a task, according to its perspective. However, there is a related concept, repurposing, that has often been overlooked in the broader context of creative problem solving in AI. Repurposing involves identifying and utilizing existing objects, resources, or processes in innovative ways to address different problems. While these two concepts may seem distinct at first glance, recent advancements in AI research suggest that they may be more closely intertwined than previously thought. By examining the underlying mechanisms and cognitive processes involved in both creative problem solving and repurposing, we can begin to understand how these approaches complement and enhance each other in the realm of artificial intelligence."
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-repurposing.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Why Repurposing?
  - name: What is Repurposing?
  - name: Creative Problem Solving
  - name: Conceptual Spaces
  - name: Incorporating Resources and Adaptability
  - name: The Repurposing Function
  - name: Evaluating Repurposing Success
  - subsections:
    - name: Task Solvability
    - name: Success Criteria
  - name: Differences and Advantages
  - name: Repurposing vs CPS
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.

---

This blog post is based on the paper Nair et al. <d-cite key="nair-etal-2024-creative"></d-cite>. The original paper contains a wider overview of computational creativity and creative problem solving. The blog post aims to questions about similarity and differences between creative problem solving and repurposing.  

## Why Repurposing?

In an era of rapid technological advancement and resource constraints, repurposing has emerged as a crucial strategy for innovation and problem-solving. The ability to adapt existing solutions to new challenges promotes efficiency. Repurposing allows us to maximize the utility of our resources, reduce waste, and find novel solutions to complex problems. From repurposing drugs for new medical treatments to transforming industrial waste into valuable products, the applications are vast and impactful.


## What is Repurposing?

Repurposing is the process of adapting or transforming an existing concept, object, or solution to serve a new purpose or solve a different problem. It involves creative thinking to identify potential new uses for established ideas or resources. At its core, repurposing is about seeing beyond the original intent and recognizing latent potential.

In the context of mathematical framework, repurposing is formalized as a function that maps concepts from one domain to another, while considering resource optimization and adaptability. This process incorporates various aspects of creativity, including exploration of existing features of concepts, combination of different ideas, and transformation of concepts into entirely new forms.

Repurposing goes beyond simple reuse; it involves a deliberate reimagining of purpose and function. By providing a structured approach to this creative process, our framework aims to enhance our ability to innovate and solve problems efficiently, making the most of our existing knowledge and resources.

## Creative Problem Solving

### Forms of Creativity

Creativity is defined as “...the ability to come up with an idea which, relative to the pre-existing domain space in one’s mind, one could not have had before. Whether any other person (or system) has already come up with it on an earlier occasion is irrelevant.” <d-cite key="boden1998creativity"></d-cite>. In computational creativity, the theory by <d-cite key="boden2004creative"></d-cite> gives a definition of this creative process in terms of manipulation of a conceptual space.
>  A “[conceptual space] is the generative system that underlies the domain and defines a certain range of possibilities: chess moves, or molecular structures, or jazz melodies. ... in short, any reasonably disciplined way of thinking" .

In terms of (neural based) agents, the conceptual space is essnetially the embedding space, the space where the agent organizes data points to simplify how real-world data is represented.

This theory divides creativity into three types:

1. _Exploratory Creativity_: Searching within an established conceptual space for new ideas or solutions.
2. _Combinational Creativity_: Making novel associations between familiar ideas.
3. _Transformational Creativity_: Altering one or more dimensions of the conceptual space itself, often leading to paradigm shifts.

We will use these concepts and categorization to draw parallels between creative problem solving (CPS) and repurposing.

### Definition of Creative Problem Solving

We borrow and adapt the mathematical formalization from the creative problem solving framework proposed by <d-cite key="gizzi2022creative"></d-cite> and further developed by <d-cite key="nair-etal-2024-creative"></d-cite>. 

In this framework, concepts are represented as points or regions in a multidimensional space, where each dimension corresponds to a quality or feature of the concept. Hence, the conceptual space $$C$$ is defined as:

$$C = C_S \cup C_A$$

Where:
- $$C_S$$ represents the set of environmental states
- $$C_A$$ represents the set of agent actions

This representation captures the dual nature of creative problem solving: it involves both understanding the current context (environmental states) and the potential modifications or applications (agent actions).

<aside style="padding:20px;background-color:green;" class="box-important l-body"> 
  <p> We borrow the simplified example from <d-cite key="nair-etal-2024-creative"></d-cite> illustrating a robot with the goal $G$ of transferring beans from a jar to a cooker: $G = in $(beans, cooker). The initial state is defined as $C_S = in$(beans, jar), $hasContainability$(spoon). The actions are defined as $C_A = scoop$(beans, $X, loc_s, loc_d)$, where $X$ refers to an object that satisfies $hasContainability(\cdot)$. In the creative problem-solving framework, when the robot doesn't have a spoon but has a glass, it uses a function $f(\cdot)$ to discover a new conceptual space: $f(C_S) = C'_S = C_S \cup hasContainability$ (glass), allowing it to solve the previously unsolvable task.</p> 

</aside>



Another way of seeing this problem is through the lenses of repurposing. Before illustrating this, let us introduce the concepts of _resources_ and _adaptability_ for repurposing.


## Incorporating Resources and Adaptability

To account for practical constraints and flexibility in repurposing, we introduce two key concepts: _resources_ and _adaptability_.

- Resources ($$R$$) represent the set of all available items, tools, materials, or even skills that can be utilized in the repurposing process. These are the building blocks from which new solutions are crafted. Formally, we define:

$$R = {r_1, r_2, ..., r_n}$$: Set of available resources

Where each $$r_i$$ represents a distinct resource. This explicit consideration of resources ensures that our repurposing solutions are grounded in what's actually available, promoting practical and implementable outcomes.

- Adaptability ($$a$$) quantifies how easily the existing resources can be repurposed to meet the new goal. It's a measure of the flexibility of our resources and the efficiency of our repurposing solution. We define the adaptability score as:
$$a = 1 - \frac{n}{N}$$
Where:

  - $n$ is the number of resources that need to be replaced or added to achieve the repurposing goal
  - $N$ is the total number of resources in the original concept

This adaptability score ranges from 0 to 1, where 1 indicates maximum adaptability (no resources need to be changed) and 0 indicates minimum adaptability (all resources need to be changed). A higher adaptability score suggests a more efficient repurposing solution, as it requires fewer changes to the existing resources. This metric helps us evaluate and compare different repurposing strategies, favoring those that make the most of what's already available.

### Determining Resource Transformation

To determine how many resources need to be transformed, we follow these steps:

1. Resource Mapping: Create a detailed list of all resources involved in the original concept.
2. Goal Analysis: Clearly define the repurposing goal and identify the requirements to achieve this goal.
3. Resource Comparison: Compare the original resource list with the requirements for the repurposed concept.
4. Transformation Count: Count the number of resources that need to be modified, replaced, or added.

5. Adaptability score: $$a = 1 - n/N = 1 - 2/5 = 0.6$$ where $$N$$ is the number of original resources and $$n$$ is the number of number of resources changed (or added).

## The Repurposing Function

We define the repurposing function as:

$$R: C \times R \times a \rightarrow C' \times R' \times a'$$

Where $$C'$$ is the new conceptual space after repurposing, $$R'$$ is the updated resource set, and $$a'$$ is the updated adaptability score.

This function encapsulates the core of our repurposing process, transforming concepts while accounting for resource usage and adaptability changes.

By formalizing repurposing in this way, we create a structured approach to creative problem-solving that balances innovation with practical considerations. 

The key differences between this repurposing framework and creative problem-solving framework are:

1. The **explicit inclusion of resources** ($$R$$) and **adaptability** ($$a$$) in the conceptual space and repurposing function.
2. The consideration of both environmental states ($$C_S$$) and agent actions ($$C_A$$) in the conceptual space, reflecting the context-dependent nature of repurposing.
3. The focus on transforming existing concepts for new purposes, rather than creating entirely new concepts.
4. The quantification of adaptability based on resource changes, providing a practical measure of repurposing feasibility.

These distinctions highlight how repurposing, while related to general creativity, has its own unique characteristics and constraints that require a specialized framework for analysis and application. 

If we were to frame the previous example about the robot in terms of repurposing, we would approach it as follows:
- Initial state:

  - $$C_S = in$$(beans, jar), $$hasContainability$$(glass), $$isContainer$$(jar), $$isContainer$$(cooker)
  - $$C_A = scoop$$(beans, $$ X, loc_s, loc_d$$), $$pour$$(beans, $$X, Y$$)
  - $$R$$ = $$\{$$jar, beans, glass, cooker$$\}$$
  - $a = 1$ (no changes needed initially)

- Repurposing goal: Transfer beans from jar to cooker without a spoon
- Repurposing process: 
  - Identify the glass as a potential tool for transferring beans
  - Redefine the use of the glass within the existing conceptual space:

    - From: $$use$$(glass, drinking)
    - To: $$use$$(glass, scooping)


- Apply existing actions in a new context: Use $$scoop$$(beans, glass, jar, cooker) instead of $$scoop$$(beans, spoon, jar, cooker)

- Update the resource set: $$R' = R$$ (no new resources added)
- Calculate new adaptability score: $$a' = 1 - \frac{1}{4} = 0.75$ (1 resource repurposed out of 4)

- Repurposing function:
$$R(C, R, a) \rightarrow (C', R', a')$$


**Key differences in this repurposing approach**:

- We don't add $$hasContainability$$(glass)$ to $$C_S$$ because this property is already known.
- The focus is on redefining the use of the glass, not discovering new properties.
- The existing action $$scoop$$ is applied with the glass instead of a spoon.
- The conceptual space isn't expanded; rather, existing concepts are applied in new ways.

Using the repurposing framework, we explicitly consider the available resources and quantify the adaptability of the solution. The focus is on how existing resources (the glass) can be repurposed to meet the goal. This approach emphasizes practical problem-solving within resource constraints, which is often more aligned with real-world scenarios where creating entirely new solutions may not be feasible. CPS aims at discovering something new and the newly discovered knowledge is applied to solve a previously impossible task.

## Evaluating Repurposing Success

### Task Solvability

To assess the effectiveness of repurposing, we introduce a task solvability function:
$$S(G, C, R, a) = \frac{1}{|K|} \sum_{k \in K} w_k \cdot s_k(G, C, R, a)$$
Where:

- $K$ is the set of criteria that define goal satisfaction
- $w_k$ is the weight of criterion $k$ (with $\sum_{k \in K} w_k = 1$)
- $s_k(G, C, R, a)$ is the satisfaction score of criterion $k$, ranging from 0 to 1

This function returns a value between 0 and 1, representing the degree to which the solution satisfies the goal criteria.

### Success Criteria

A repurposing is successful if $$S(G, C', R', a') > S(G, C, R, a)$$ and $$S(G, C', R', a') \geq \theta$$
Where $$\theta$$ is a threshold value (e.g., 0.7) that defines the minimum acceptable satisfaction level.
An example of set of criteria $K$ for the pasta example above is:

- Low-carb ($k_1$): Carbohydrate content below 20g per serving
- Protein content ($k_2$): At least 20g of protein per serving
- Taste similarity ($k_3$): Flavor profile similar to original dish

## Forms of Repurposing

In the same way as for creative problem solving, it is possible to frame repurposing according to the three types. 

1. *Exploratory Repurposing*: consists of exploring minor modifications or new applications of an existing concept within its current domain.
Example: Using a smartphone as a flashlight. The core function (emitting light) was already present, but it's explored for a new purpose.
2. *Combinational Repurposing*: Consists of combining two or more existing concepts to create a new application or solution.
Example: The spork, which combines the functionalities of a spoon and a fork into a single utensil.
3. *Transformational Repurposing*: Is defined as changing the function or application of an existing concept, often moving it to an entirely new domain.
Example: Repurposing old tires as playground equipment or garden planters. This transforms the tire from a vehicle component to a completely different use in a new context.


## Comparing Creative Problem-Solving and Repurposing Frameworks

Let's consider the example of transforming a classic Italian pasta dish into a low-carb option.

### Creative Problem-Solving Framework

1. Define the Conceptual Space:
   Let $$C$$ be the space of all possible dishes, where each dimension represents attributes like ingredients, cooking methods, flavors, textures, etc.

2. Initial Concept:
   $$c$$ = Spaghetti Bolognese, represented as a point in $$C$$

3. Goal:
   $$G$$ = Create a low-carb alternative

4. Creative Operators:
   We'll use a transformational operator $$T$$ that replaces high-carb ingredients with low-carb alternatives.
   
5. Apply Transformation:
   $$c'$$ = $$T(c)$$ = Zucchini Noodle Bolognese

6. Evaluate:
   Define a function $$f(c, G)$$ that measures how well the new concept $$c'$$ meets the goal $$G$$.
   If $$f(c', G) > f(c, G)$$, then the creative transformation is considered successful.

### Repurposing Framework

1. Define the Conceptual Space:
   $$C = C_S \cup C_A$$
   Where:
   $$C_S$$ = {pasta dish, Italian cuisine, high-carb meal}
   $$C_A$$ = {cook, prepare ingredients, serve}

2. Initial Resources:
   $$R$$ = \{pasta, ground beef, tomatoes, herbs, cooking equipment\}
   $$N$$ = 5$ (total number of resources)

3. Initial Adaptability Score:
   $$a = 1$$ (no changes made yet)

4. Goal:
   $$G$$ = Create a low-carb alternative

5. Repurposing Process:
   a) Identify resource to replace: pasta and tomatoes
   b) Repurpose existing ingredients:
    - Reduce pasta quantity by 2/3
    - Increase tomato quantity, cooking them down to a thicker sauce
    - Finely chop herbs to increase volume and flavor intensity
    c) Update resources:
    $$R'$$ = \{reduced pasta, ground beef, increased tomatoes, increased herbs, cooking equipment1\}
    d) Calculate new adaptability score:
    $$n = 3$$ (pasta, tomatoes, and herbs used differently)
    $$a' = 1 - \frac{n}{N} = 1 - \frac{3}{5} = 0.4$$

6. Apply Repurposing Function:
   $$R(C, R, a) \rightarrow (C', R', a')$$
   Where:
   $$C' = C_S' \cup C_A'$$
   $$C_S'$$ = \{low-carb dish, Italian-inspired cuisine, vegetable forward meal\}
   $$C_A'$$ = \{reduce pasta, increase vegetables, cook, prepare ingredients, serve\}

7. Evaluate:
Define a satisfaction function $$S(G, C, R, a, K)$$ that measures how well the concept meets the goal $$G$$ and the set of criteria $$K$$, given the conceptual space $$C$$, resources $$R$$, and adaptability score $$a$$.
$$K = {k_1, k_2, k_3}$$, where:

$$k_1$$: Lower-carb (Reduced carbohydrate content compared to original)
$$k_2$$: Protein content (Maintain protein content per serving)
$$k_3$$: Taste similarity (Flavor profile similar to original dish)

8. Calculate satisfaction scores:
$$S(G, C, R, a, K)$$ for the original Spaghetti Bolognese
$$S(G, C', R', a', K)$$ for the repurposed Zucchini Noodle Bolognese
A repurposing is successful if:
$$S(G, C', R', a', K) > S(G, C, R, a, K)$$ and $$S(G, C', R', a', K) \geq \theta$$
Where $$\theta$$ is a threshold value (e.g., 0.7) that defines the minimum acceptable satisfaction level.


### Differences and Advantages of Repurposing Framework

1. *Focus*:
   - CPS: Primarily focused on generating novel solutions, which may or may not involve existing resources.
   - Repurposing: Specifically focused on finding new uses for existing resources or concepts.

2. *Constraint consideration*:
   - CPS: May consider constraints, but is not inherently bound by them.
   - Repurposing: Explicitly works within given resource constraints as a core principle.

3. *Outcome*:
   - CPS: Can result in entirely new inventions or discoveries.
   - Repurposing: Always results in a new use for an existing item or concept.

4. *Process*:
   - CPS: May involve broader exploration of possibilities, including those outside current resources.
   - Repurposing: Starts with available resources and explores possibilities within that scope.

5. *Applicability*:
   - CPS: Useful in a wide range of scenarios, especially when novel solutions are needed.
   - Repurposing: Particularly valuable in resource-constrained or sustainability-focused contexts.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/cps_vs_rep.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Some examples of the concepts defining creative problem solving and repurposing.
</div>


## Repurposing vs. Creative Problem Solving

While CPS is a powerful and essential approach in many scenarios, there are numerous problems that can benefit significantly from being framed as repurposing challenges. The dataset presented in <d-cite key=tian2024macgyver></d-cite> provides an excellent example of such problems.
The MACGYVER dataset, consisting of over 1,600 real-world problems designed to trigger innovative usage of objects and necessitate out-of-the-box thinking, can be viewed through the lens of repurposing rather than pure creativity. These problems often involve using existing objects or resources in novel ways to solve unexpected challenges - a hallmark of repurposing.

Framing these as repurposing problems offers several advantages:

- *Resource Constraints*: Repurposing explicitly considers available resources, which is crucial in the MacGyver-style problems where solutions must be crafted from limited, on-hand materials.
- *Adaptability Focus*: Our repurposing framework's emphasis on adaptability aligns well with the need to adapt existing objects for new purposes in these challenges.
- *Practical Feasibility*: The repurposing approach inherently considers the practicality of solutions, addressing the issue of physically-infeasible actions proposed by LLMs in the study.
- *Structured Approach*: Repurposing provides a more structured framework for tackling these problems, potentially bridging the gap between human intuition and AI's broad knowledge base.

By viewing such problems through the repurposing lens, we can potentially develop more effective strategies for both human problem-solvers and AI systems. This approach complements creative problem solving, offering a structured method for innovation within constraints - a common scenario in real-world challenges.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/macgyver.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Sample illustration from the MacGyver dataset <d-cite key=tian2024macgyver></d-cite>. The problems in this dataset require innovative usage of objects among the ones available which is the scope of repurposing.
</div>


## Conclusion

The exploration of repurposing through this mathematical framework has shed light on its relationship to creative problem-solving. While repurposing shares many characteristics with general creative problem-solving, our analysis reveals that it can be viewed as a specialized subset with distinct features:

- *Constraint-Driven Creativity*: Repurposing is inherently constrained by existing resources and structures, forcing creativity within defined boundaries.
- *Efficiency Focus*: Unlike open-ended creative problem-solving, repurposing emphasizes resource optimization and adaptability of existing solutions.
- *Transformation Emphasi*s*: While creative problem-solving can start from scratch, repurposing always begins with an existing concept or object, focusing on transformation and recontextualization.

These distinctions suggest that repurposing, while related to creative problem-solving, is a unique process that warrants specific attention and methodologies.
Regarding the question of whether AI efforts should prioritize repurposing over general creative problem-solving, our analysis suggests several compelling reasons to focus on repurposing:

- **Resource Efficiency**: In a world of limited resources, repurposing offers a more sustainable approach to innovation.

- **Structured Exploration:** The constraints inherent in repurposing provide a more structured problem space for AI systems to explore, potentially leading to more practical and immediately applicable solutions.

- **Cross-Domain Innovation**: Repurposing encourages the transfer of ideas across different domains, a process that AI could potentially excel at by identifying non-obvious connections.

In conclusion, while repurposing and creative problem solving share common ground, repurposing emerges as a distinct and valuable approach. The structured nature of repurposing, combined with its focus on efficiency and transformation of existing solutions, makes it a particularly promising area for AI research and development. As we face increasingly complex global challenges, AI-driven repurposing could offer a powerful tool for innovation, potentially yielding more immediate and practical solutions than broader creative problem-solving approaches.
Future work in this area could focus on developing AI systems that can effectively navigate the repurposing process. Additionally, further exploration of how humans and AI can collaborate in repurposing tasks could lead to powerful hybrid approaches, combining human intuition with AI's vast knowledge and processing capabilities.

<d-bibliography src="/2025/assets/bibliography/2025-04-28-distill-example.bib"></d-bibliography>










