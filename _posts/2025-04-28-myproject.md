---
layout: distill
title: "Lost in Prediction: The Missing Link Between LLMs and Narrative Economics. Lost in Prediction: Why Social Media Narratives  Don't Help Macroeconomic Forecasting?"
description: Your blog post's abstract 2.
  Please add your abstract or summary here and not in the main body of your text. 
  Do not include math/latex or hyperlinks.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Albert Einstein
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: IAS, Princeton
  - name: Boris Podolsky
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: IAS, Princeton
  - name: Nathan Rosen
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

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



## What is Narrative Economics?

Narrative Economics is the study of how popular stories and ideas (aka **narratives**) formed about the state of the economy could have real effects in the world. In this context, a “narrative” is a belief about the state of the world that is shared by the population, regardless of the actual state of the economy. For example, a narrative might be the belief that housing prices are increasing, whereas in-reality (according to economic indicators) they are stagnating. 

The central idea is that the spread of viral narratives can influence individual and collective economic behavior, leading to fluctuations in markets, changes in investment patterns, and even broader economic shifts.

This term is heavily attributed to Robert J. Shiller, a Nobel laureate economist and the founder of Narrative Economics which defined it as:

> *Spread through the public in the form of popular stories, **ideas can go viral and move markets** —whether it’s the belief that tech stocks can only go up, that housing prices never fall, or that some firms are too big to fail. Whether true or false, stories like these—transmitted by word of mouth, by the news media, and increasingly by social media—**drive the economy by driving our decisions about how and where to invest, how much to spend and save, and more.***  
  *- Robert J. Shiller*
  TODO: cite

### The Ambiguity of the Term "Narrative"

The term "narrative" itself has different connotations in NLP compared to economics, which might lead to some confusion. 

In NLP, narrative is XXX.

In Narrative Economcis, as stated in the above section, narrative is a shared belief or idea that spreads through the population and potentially influences economic behavior. 

Our research uses the term "narrative" in the economic sense. We're interested in how shared beliefs about the economy can be used to predict market trends. 

**But, how can we capture such narratives?**
The economic term is wide and it is undefined what requirements a story or idea must have in order to be considered as "narrative". Yet, we can look at some characteristics Shiller's mentions about narratives to have a better understanding:

* First, the story should be viral, publicly believed, in order to change a large enough audience to move the market.
* Second, Shiller's uses some propositions while explaining the term. He states that "the economic impact of narratives may change through time" and "narrative constellations have more impact than any one narrative". 
* Lastly, Shiller mentions social media as a source of narratives and Google Ngram as a tool for tracking them. 
**TODO: cite coursera

Combined together, to capture a narrative, one would need a good measure of what many people are discussing about, over time. Twitter (X), in this case, is an almost ideal source of information for capturing this distribution of opinions. 

**And how to extract the "narrative" aspect from tweets?** Aligning with Shiller’s arguments and with existing literature, the extraction might be (for example) a sentiment, a topic, or a specific economic outlook. TODO: citations



### Social Media Narratives

We have collected two comprehensive datasets of tweets from Twitter (X), a platform chosen for its real-time reflection of public opinions and ability to capture diverse economic narratives. These datasets cover a broad time-frame, and a wide range of topics relevant to the economy, including economics, business, politics, and current events, ensuring a broad and comprehensive representation of narratives.  

Both datasets were carefully curated using targeted queries with the inclusion of specific keywords, and were analyzed to ensure quality and relevant for capturing economic narratives. 

**Pre-Pandemic Twitter Dataset:** Utilizing Twitter API (TODO: cite), we collected 2.4 million tweets from Twitter's early days (January 2007) to COVID-19 pandemic (December 2020). To prioritize viral tweets, we retrieved the daily top 200 tweets based on follower count,  then we randomly sampled 100 to mitigate potential bias towards highly active accounts typically associated with news outlets. This process yeilded a dataset contributed by about 250,000 users per collected topic, each boasting an average follower count of 100 million, including global leaders, news outlets and other influencers.

**Post-2021 Twitter Dataset:** We wanted to test the predictive power of our employed LLM (Chat Completion API with GPT-3.5 TODO:cite) on a timeframe beyond the LLM's data cutoff date (September 2021). Therefore, we collected a second dataset ranging from September 2021 until July 2023. This assures the LLM relies solely on provided tweets and pre-existing knowledge, unaware of "future" knowledge. Here we collected the tweets monthly, using Twitter Advanced Search, restricting to users with at least 1,000 followers. Overall we curated 2,881 tweets <d-footnote>As this data collection method is more restricted than the
previous, the resulting dataset is relatively smaller.</d-footnote> contributed by 1,255 users including politicians, CEOs, activists, and academics. 

An example tweet can be:

{% twitter https://x.com/CNLiberalism/status/1525672295775223808 %}

### Did We Really Collect Narratives?
To confirm the presence of narratives within our Twitter dataset, we conducted an analysis using RELATIO, a tool designed to "capture political and economic narratives" by mapping relationships and interactions among entities within a corpus.  Upon processing our dataset with RELATIO, we obtained "narrative statements" (as defined in their paper) and visualized their temporal distribution:

<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/relatio_plot.jpg" class="img-fluid z-depth-2" %}
    </div>

### LLMs can Learn Narratives
A more advanced technique to extract and analyze narratives is using LLMs. Prompting ChatGPT (TODO: cite) with monthly tweets and prices of economic indicator from matching dates, we generated LLM-based narratives analysis, one for each date in the post-2021 dataset, containing a component of summarized analysis of the tweets and a component of potential effect on the given financial indicator. 

Here's a snippet of such an LLM-based narrative analysis for inputs of dates 29/08/2022 to 28/09/2022. In this time period the Federal Reserve raised the interest rates in an effort to combat inflation, the US Supreme Court ruled that the Biden administration could not extend the pause on student loan payments:

<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/chatgpt_snippet.jpg" class="img-fluid z-depth-2" %}
    </div>

This snippet demonstrates the LLM’s ability to aggregate information, condensing and distinguishing between opinions and occurrences conveyed in the tweets. Moreover, the LLM links its insights to potential future consequences for the financial indicator, a pivotal initial move towards prediction.

## Why Macroeconomics?

Now that we have social media narratives in hand, let's focus on *macroeconomics*.  

**Macroeconomics** studies the behavior of the economy as a whole, examining factors like inflation, unemployment, and economic growth. **Microeconomics**, on the other hand, is concerned with the decision-making of individuals and firms, examining indicators like a certain stock.

A core concept in Narrative Economics is that narratives can drive economics flunctuations. This is especially intriguing at the macroeconomic level, as the theory suggests that widely shared stories can influence the collective decisions of millions of individuals. Additionaly, existing research focuses on microeconomic indicators within the context of Narrative Economics, while the application in macroeconomics remains relatively unexplored. TODO:cite

However, studying this macroeconomically is more complax than microeconomically due to the complex interplay of various factors, the need for broad-covering narratives, and the inherent difficulty in isolating causal relationships. 

### Our Macroeconomic Indicators

We focus on predicting three key macroeconomic indicators:

**Federal Funds Rate (FFR):** The interest rate at which depository institutions, such as banks, lend reserve balances overnight to meet reserve requirements. The FFR serves as a Federal Reserve monetary policy tool, is influenced by public perception of economic stability, and its fluctuations impact various sectors, making it widely monitored.

**S&P 500:** A stock market index measuring the performance of the 500 largest publicly traded companies in the U.S. It reflects collective investor confidence in economic growth and risk appetite and is widely regarded as a barometer of the overall health of the US stock market. 

**CBOE Volatility Index (VIX):** Measures market expectations of future volatility based on S&P 500 options prices, often referred to as the 'fear gauge' as it tends to rise during market stress and fall during market stability.

These indicators are well-suited for testing the predictive power of narratives for macroeconomics due to their daily frequency, which aligns with the rapid pace of Twitter, and their sensitivity to public sentiment and behavior.


## Connecting the dots: Testing Narratives' Effectiveness for Macroeconomic Forecasting

The previous two sections discussed the theory of Narrative Economics and our curated Twitter dataset, which holds narratives within them, and the distinction between macroeconomics and microeconomics, highlighting why it is interesting to research the theory at the macroeconomic level and what macroeconomic indicators we chose.

We can now delve into the series of experiments we tested to assess the central question - **can economic narratives can provide valuable insights for future macroeconomic movements?**

Each experiment test the predictive power of narratives from the curated Dataset, for macroeconomic prediction of one (or more) of the financial targets intriduced before: FFR, S&P 500, and VIX. 

We won't be able to cover all the details of the experiments in this blog, but it is available in our paper. 

### Experimental Setup

**Prediction tasks:** 

We test the predictive power of narratives on three tasks commonly used in macroeconomic literature (TODO:cite):
* Next value: predicts the target’s value at the specified horizon.
* Percentage change: predicts the target’s percentage change between the specified horizon and
the day before.
* Direction change: classifies the target’s direction of change (increase or decrease) between the
specified horizon and the day before.


**Models Categories:** 

We differ our models into 3 categories based on their input signal:
* Financial (F): utilizes historical financial data, from the past week or month.
* Textual (T): leverages solely textual data, either raw tweets or tweets’ analyses.
* Textual & Financial (TF): draws upon both textual and financial data as input.

  Our goal is to effectively leverage insights from both textual narratives and historical financial patterns to improve prediction accuracy.  The added value of incorporating textual narratives can be demonstrated if a model that utilizes both text and financial data (TF model) outperforms a model that relies solely on financial data (F model). 

**Baselines:**

*Financial baselines:*
* As/inverse-previous: next value is the same/inverse as previous.
* majority: next value is the majority vote of the previous week/training data.
* up/down: always predict "increase"/"decrease".

*Counterfactual textual baselines:*
* random texts: feeding the LLM with randomly generated sentences comprised of varying random words. This baseline evaluate wether the LLM actually utilize the content of tweets.
* Shuffled tweets: feeding the LLM with chronologically disordered tweets, isolating the impact of temporal narratives from confounding patterns or memoraization. This baseline assess the model reliance on temporal narratives.
* Synthetic `narratives`: Fedding the LLM with generated narrative-like sentences experssing positive or negative cues, aligned with subsequent changes in the financial target. This baseline assess the LLM's ability to infer relationships between aligned narratives and the following market changes.

**Models:**

Our model selection progresses from simpler models, frequently employed in the financial literature (TODO:cite), to more advanced architectures. This progression serves two purposes: 
1. Achieving positive results with simpler models provides a stronger evidence for the predictive signal of narratives.
2. It allows us to build upon existing research in Narrative Economics, which is primarily rooted in finance and often utilizes relatively simple models, before exploring more advanced NLP approaches.

*Financial models:* these include traditional ML models (e.g., Linear Regression, SVM), DA_RNN (TODO:cite), and T5 (TODO:cite) which receives financial input in a text format. Each model is fed with a sequence of historical financial values of the target indicator, either as individual features per day or as a time-series.  

*Textual models:*
* Daily sentiment: a simple method, commonly used in the literature, is presenting each tweet with its sentiment score. Then, we average the scores of individual tweets of the same dates to recive a daily sentiment, and concatenate over a week.
<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/models_diagram_1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    
* LLM's representations of individual/joint tweets: we derive embeddings of individual or concatenated tweets using pre-trained languge models (BERT,RoBERTa and T5 TODO:cite). In the individual case, tweets' embeddings are aggregated daily by averaging or concatenating embeddings of same-date tweets. In the joint case, tweets are concatenated daily to create a single daily embedding, potentially capturing their collective meaning without explicit aggregation, avoiding potential information loss.
<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/models_diagram_2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
  
* LLM-generated analyses for prediction/as input to a subsequent prediction model: First, we feed GPT-3.5 (TODO:cite) with a month of tweets and corresponding financial values of the target indicator to create monthly analyses.Then, these analyses are either used directly for prediction or as an input to a subsequent T5 model.  
*since the LLM receives both tweets and financial data to enable analyzing relationships, this method applies only for a TF model type.
<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/models_diagram_3.jpg" class="img-fluid rounded z-depth-1" %}
    </div>

*Fusing textual and financial models:* we experiment with several strategies for combining the representations from the T and F models for a unified prediction:
* concatenation: the simplest approach is concatenating the T and F representations.
* DA-RNN(TODO:cite): The dual-stage attention-based RNN model predicts the current value of a time-series based on its previous values and those of exogenous series. We feed historical financial representations (F) as the time series and
textual representations (T) as the exogenous series.
* Prompt-based fusion: LLM-based analysis of given tweets and historical financial values of the target indicator are fed together with raw historical values of the target to a T5 model as separate segments.

  Given a TF model we can derive a T or F model by ommiting or zeroing either F or T component, respectively.



### The Challenges in Improving Models with Narratives

## What Can We Take Away?

## Citations

bla



