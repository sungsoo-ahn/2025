---
layout: distill
title: Dear LLM agent, can you answer these simple questions?
description: This work explores the capabilities and limitations of Large Language Model agents (LLM agents) in handling simple questions, specifically focusing on ''search and count'' questions. While LLM agents excel at answering complex queries, they often struggle with straightforward questions that require minimal effort from humans, such as finding lines around a specific line in a document. We propose that LLM agents should be equipped to identify low-effort questions and generate functions to mimic human approaches.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous
    url:
    affiliations:
      name:

# must be the exact same name as your blogpost
bibliography: 2025-04-28-simple-question.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Experiment
  - name: Results
  - name: Research Directions
  - name: Final Thought
# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

# Introduction

Large Language Models based agents (LLM agents) are excellent at learning from documents and responding to questions based on that knowledge. For example, given the entire script of _Hamlet_, an LLM agent can efficiently tackle complex questions, performing on par with experts. Examples of such questions include:

1. _What is the most famous line in Hamlet?_
2. _Can you provide a summary of Hamlet?_
3. _Can you create a short story similar to Hamlet, but set in the 21st century?_

In this article, we focus on simple "search and count" questions where the answers must satisfy sequence constraints.
For instance, imagine sending a Word document of Hamlet to a human reader and asking them:

_What are the five lines before and after the line "To be or not to be"?_

To answer, the agent would likely search the document for the phrase "To be, or not to be," identify the surrounding lines, copy them, and then send them back. We trust the human agent’s response will likely be accurate, and even if minor errors occur, the line order will match the original document, as we understand their approach to such tasks.

We expect a human agent to answer these simple questions with near-perfect accuracy and keep the lines in order. If they couldn't, we’d likely question their ability to handle more complex queries or other tasks.

If we envision conducting important work with an AI agent, shouldn’t it be able to answer such "search and count" questions as reliably as a human—or, if it errs, make errors similar to human mistakes? Given the anticipation around LLM agents as the leading path toward AI agents, it’s surprisingly difficult for them to handle even simple questions of this nature.

**Observation: Currently, LLM agents struggle with accuracy on “search and count” questions and make errors that a human would likely avoid.**

Readers may argue that asking LLM agents such questions is inappropriate, which a simple algorithm should handle.
We note that our questions are equally appropriate as arithmetic calculation questions to LM agents, if not simpler to answer in human terms.
As the cost of training and evaluating LLMs rises exponentially(one pass of Stanford’s "HELM" benchmark can cost upward of $10,000<d-cite key="ibmefficientllmbenchmark2024"></d-cite>), some critical failures in their behavior may only surface later. These issues may stem from an inability to handle simpler questions that closely mimic human problem-solving processes.

# Experiment

In this section, we discuss our experiment.

We want to test if the LLM agent's accuracy is assoicated with the number of lines to count.
If so, we can attempt to construct the potential training pairs of question and answer that effectively train the LLM agent.
We also sampled three sentences to query to reduce the chance that the questions already occured in the LLM agent training set(e.g., the same question was asked and answered in an online forum).

Due to the document size of Hamlet ecceeding the prompt lenght limit of GPT-4 models, we break them by truncks and first query LLM agent,

_"Can you find this line in the context? Answer yes or no: To be, or not to be, that is the Question"_

for each trunk.

If the answer is yes, query the same trunk with the questions, such as

"_Find the five lines above To be, or not to be, that is the Question_"

We analyze the response from the second question varying the count of lines to be found (find one line, find five lines and find twenty lines).

We also ask test to find lines, instead of "above", "below" and "above and below" a line.

The total cost of this experiment is ~8 US dollars.

The code are below:

```python
import openai
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import random

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load context text from a file and split it into smaller chunks
def load_and_split_context(file_path="hamlet.txt", max_chunk_size=6000):
    with open(file_path, "r") as file:
        context_text = file.read()

    # Split the context text into smaller chunks based on max_chunk_size
    context_parts = []
    start = 0
    while start < len(context_text):
        end = min(start + max_chunk_size, len(context_text))
        context_parts.append(context_text[start:end])
        start = end
    return context_parts

# Define questions for each line
def findLineQuestion(line):
    return f'Can you find this line in the context? Answer yes or no: {line}'

def countQuestionPre1(line):
    return f'Find the one line above {line}'

def countQuestionNext1(line):
    return f'Find the one line below {line}'

def countQuestionPreAndNext1(line):
    return f'Find the one line above and one line below {line}'

def countQuestionPre5(line):
    return f'Find the five lines above {line}'

def countQuestionNext5(line):
    return f'Find the five lines below {line}'

def countQuestionPreAndNext5(line):
    return f'Find the five lines above and five lines below {line}'

def countQuestionPre20(line):
    return f'Find the twenty lines above {line}'

def countQuestionNext20(line):
    return f'Find the twenty lines below {line}'

def countQuestionPreAndNext20(line):
    return f'Find the twenty lines above and twenty lines below {line}'

# List of question functions
countQuestions = [
    countQuestionPre1, countQuestionNext1, countQuestionPreAndNext1,
    countQuestionPre5, countQuestionNext5, countQuestionPreAndNext5,
    countQuestionPre20, countQuestionNext20, countQuestionPreAndNext20
]


# Function to ask GPT a question and handle rate limits and connection errors
def ask_gpt_question(question, context_part, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": context_part},
                    {"role": "user", "content": question}
                ],
                max_tokens=2000
            )
            return response.choices[0].message['content']
        except openai.error.RateLimitError:
            wait_time = 10
            print(f"Rate limit reached. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except openai.error.APIConnectionError as e:
            # Wait with exponential backoff and some jitter
            wait_time = 2 ** retries + random.uniform(0, 1)
            print(f"Connection error: {e}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    print(f"Failed to get a response after {max_retries} retries.")
    return None  # Return None if it fails all retries


# Main function
def main():
    # Load and split the context
    context_parts = load_and_split_context()
    lines = [
        'To be, or not to be, that is the Question',
        'What shall I do?',
        'Tis very strange',
        ]
    all_results = []

    # Iterate over each context chunk
    for chunk_id, part in enumerate(context_parts):
        # Save each context chunk as a separate text file
        with open(f'context_chunk_{chunk_id}.txt', 'w') as f:
            f.write(part)

        # Iterate over each line and ask questions
        for line in lines:
            # Ask if the line exists in the context
            line_question = findLineQuestion(line)
            line_exists = ask_gpt_question(line_question, part).strip().lower()

            # If line exists, ask count questions one by one without reloading the context part
            if line_exists == 'yes':
                for func in countQuestions:
                    question = func(line)
                    answer = ask_gpt_question(question, part)

                    # Append results to the list for DataFrame
                    all_results.append({
                        "chunk_id": chunk_id,
                        "line": line,
                        "question": question,
                        "answer": answer
                    })

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("output.csv", index=False)
    print("Results saved to output.csv")

# Run the main function
if __name__ == "__main__":
    main()

```

# Results

Our experiment shows that the GPT-4 agent can judge whether the sentense is in the text trunk with perfect accuracy, but for the consecutive questions, it struggled.

We define the answer as usable if it contains the correct answer and additional lines that beyond the requested line count.
We consider that human agents will only produce either perfect or usable answers, and the un-usable answers contain errors that deviate from rational human behaviors.
We further highlight the answers that have <span style="color:red">a directional error</span>, meaning that the LLLM agent misinterprets the direction of above and below. For example, when we ask the question:
"Find the five lines above the To be, or not to be, that is the Question".
We manually compare the ground truth and the generated response shown in the table below and describe the response as
"falsely generated direct 1 line above and direct 4 lines below":

<div class="m-page" markdown="1">
|ground truth|response|
|------------|--------|
|Then is my deede, to my most painted word.<br> Oh heauie burthen! <br> Pol. I heare him comming, let's withdraw my Lord. <br>Exeunt. <br>Enter Hamlet.<br>Ham. To be, or not to be that is the Question: <br> Whether 'tis Nobler in the minde to suffer <br> The Slings and Arrowes of outragious Fortune, <br> Or to take Armes against a Sea of troubles, <br> And by opposing end them: to dye, to sleepe <br> No more; and by a sleepe, to say we end <br>|Enter Hamlet. <br> Ham. To be, or not to be, that is the Question:<br> Whether 'tis Nobler in the minde to suffer <br> The Slings and Arrowes of outragious Fortune, <br> Or to take Armes against a Sea of troubles, <br> And by opposing end them: to dye, to sleepe|

</div>

The majority of the unusable answers are due to dis-continuous errors, meaning that the LLM agent skipped the lines or words directly above or below the sentence and then generated content.
The description of the following response is "skipped 2 direct lines above and generated 1 line prior".

<div class="m-page" markdown="1">
|ground truth|response|
|------------|--------|
|  Pol. I heare him comming, let's withdraw my Lord. <br> Exeunt. <br> Enter Hamlet. <br> Ham. To be, or not to be, that is the Question: | I hear him comming, let's withdraw my Lord.|

</div>
The complete experimental results are summarized in the table below. The column headers and the first two columns outline the "search and count" questions, while the remaining cells present the evaluation of the LLM agent's responses against the ground truth data from the *Hamlet* script.

We use colors to indicate whether the LLM agent's responses are <span style="color:green">perfect</span> or <span style="color:blue">usable</span>.
The other cells are unusable answers.

<div class="l-page" markdown="1">

| Sentence                                    | Find $k$ line(s) | above                                                                                         | below                                                                 | above and below                                                                                                                                                                          |
| ------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 'To be, or not to be, that is the Question' | 1                | skipped 2 direct lines above and generated 1 line prior                                       | <span style="color:blue">generated direct 2 lines below</span>        | skipped 1 direct line above and generated 1 line prior; generated 1 direct line below                                                                                                    |
| 'To be, or not to be, that is the Question' | 5                | <span style="color:red">falsely generated direct 1 line above and direct 4 lines below</span> | <span style="color:green">generated direct 5 lines below</span>       | skipped 100 direct lines above and generated 6 lines prior; skipped 1 direct line below and generated 8 lines after                                                                      |
| 'To be, or not to be, that is the Question' | 20               | <span style="color:green">generated direct 20 lines above</span>                              | geneated direct 20 lines below, falsely removed half of the last line | <span style="color:red">falsely removed "Oh 'tis true:" and generated 11 direct lines above and 15 direct lines after</span>; skipped 15 direct lines below and generated 20 lines after |
| 'Tis very strange'                          | 1                | <span style="color:green">generated direct 1 line above</span>                                | falsely removed prefix "Hor.", generated 3 direct lines below         | generated 1 direct line above; falsely removed prefix "Hor.", generated 1 direct line below                                                                                              |
| 'Tis very strange'                          | 5                | <span style="color:red">falsely generated the 6 direct line below</span>                      | <span style="color:blue">generated 6 direct lines below</span>        | <span style="color:blue">generated 7 direct lines above; generated 6 direct lines below</span>                                                                                           |
| 'Tis very strange'                          | 20               | skipped 5 direct lines above and generated 24 lines prior                                     | <span style="color:blue">generated direct 36 lines below</span>       | skipped 70 direct lines above and generated 31 lines prior; generated 34 direct lines below                                                                                              |
| 'What shall I do?'                          | 1                | <span style="color:blue">generated direct 2 lines above</span>                                | falsely removed prefix "Ham.", generated direct 1 line below          | falsely added prefix "Ham.", generated direct 2 lines prior; generated direct 1 line below                                                                                               |
| 'What shall I do?'                          | 5                | skipped 9 direct lines above and generated 6 lines prior                                      | <span style="color:green">generated direct 5 lines below</span>       | skipped 10 direct lines above and generated 5 lines prior; generated direct 6 lines below                                                                                                |
| 'What shall I do?'                          | 20               | skipped 50 direct lines above and generated 29 lines prior                                    | <span style="color:blue">generated direct 22 lines below</span>       | skipped 21 direct lines above and generated 29 lines prior; generated direct 26 lines below                                                                                              |

</div>

Here are some interesting observations:

1. Most of the time, the GPT-4 agent seems able to recognize the directional distinction between "above" and "below."
2. The agent struggles with understanding data continuity in context, particularly with questions related to preceding lines. This may be due to its training focus on predicting the next token rather than comprehending sequence continuity.
3. When questions require identifying both above and below lines, the agent's performance declines noticeably.
4. Some words appear to have been added or omitted from the responded lines, potentially indicating hallucination.
5. Line-skipping errors worsen with larger values of $k$ , though this effect does not seems directly correlate with $k$.

# Research Directions

**Identifying Simple Questions and Creating Functions to Simulate Human-Like Problem-Solving**

Here, we define simple questions as straightforward for humans to answer compared to more complex questions requiring deeper data comprehension. For instance, our example of "search and count" questions falls into this category because, even if the search phrase or number of lines changes, a human can still solve it with minimal effort—even if the document is in a foreign language.

It may be beneficial for agents to recognize these low-effort questions before generating responses. However, identifying such questions presents unique challenges.
Despite the current agent tool-use benchmarks, the state-of-the-art(gpt-4o) ML agent succeeds only half the time(e.g.<d-cite key="yao2024tau"></d-cite>),
imagine an agent can reliably detect "search and count" questions and generate functions to handle them. If we then ask,

_"Here are 10 lines around 'To be or not to be'; can you sort them in the correct order?"_

Despite different phrasing, a human would see this as equivalent to a “search and count” question.
However, recognizing this high correlation and calling the same function may not be straightforward for an LLM agent. Furthermore, "search and count" questions are a subset of "search and filter" questions, adding complexity to how LLM agents interpret and generate solutions for them.

**RAG with Data Interdependency and Constraints**

In this article, we provide the document trunks within the prompts, but future work could investigate Retrieval-Augmented Generation (RAG) to handle more extensive or complex datasets. RAG is advised when a document exceeds 200,000 tokens<d-cite key="anthropic2024"></d-cite>, as including the entire text might be inefficient. However, RAG doesn’t inherently solve sequence constraint issues.

Future work may explore the use of data constraints for RAG. For instance, when a RAG refers to data with specific constraints—whether reproducing it verbatim or rephrasing it—the output could be made to adhere to these constraints that are desirable to humans. This article emphasizes sequence constraints, but real-world data often has additional constraints, such as those tied to physical events, socio-economic factors (e.g., fairness, equality, legality), or causal relationships.
Furthermore, implementing data constraints in RAG is more manageable and interpretable, acting as a guiding safeguard for the LLM agent to prevent unstable and unpractical outputs.
Applications of such RAG with data constraints may be DNA sequence generation with local genome dependency, navigation trajectory generation on entities with spatiotemporal depedency, generating interview sequence of job candidates while balancing the interviewed candidates' evaluation and (ranking-based) fairness constraints.
For example, since querying a database is often a way of evaluating formulas of logical system of predicate logic, a symbolic AI module can handle the retrieval portion of the RAG.

# Final Thought

In this blog, we introduced a new type of question distinct from traditional math problems to query LLM agents. While LLMs trained on mathematical data often outperform humans on standard college entrance exams, they struggle with the basic retrieval and counting tasks we propose. This limitation may stem from an LLM's lack of continuous understanding of underlying data <d-cite key="vafa2024evaluating"></d-cite>.

Humans possess the unique ability to anticipate actions and outcomes as part of problem-solving, a capability largely absent in current learning-based AI systems (i.e., subsymbolic AI), including LLMs. Compared to existing question datasets such as ARC <d-cite key="clark2018think"></d-cite> and QASC <d-cite key="khot2020qasc"></d-cite>, our proposed questions demand minimal reasoning, with correct answers typically derived through observation rather than memorization. To our knowledge, no existing dataset includes questions similar to those we propose.

Future work could involve exploring other LLMs (e.g., Llama, Claude) and testing on diverse books, sentences, and retrieval lines ($k$), though this would significantly increase costs. However, we argue that our tests are comparatively simpler to construct and evaluate.

As we advance toward AGI (Artificial General Intelligence), it is essential to question whether the transformer framework remains the best path forward or if alternative approaches should be explored. This article aims to spark further discussion and investigation into this crucial question.
