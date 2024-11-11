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
  - name: What is Narrative Economics?
    subsections:
    - name: The Ambiguity of the Term "Narrative"
    - name: Social Media Narratives
      subsections:
      - name: Did We Really Collect Narratives?
      - name: LLMs Can Understand Narratives
  - name: Why Macroeconomics?
    subsections:
    - name: Our Macroeconomic Indicators
  - name: "Connecting the Dots: Testing Narratives' Effectiveness for Macroeconomic Forecasting"
    subsections:
    - name: Experimental Setup
      subesctions:
      - name: Prediction Tasks
      - name: Models Categories
      - name: Baselines
      - name: Models
    - name: The Challenges in Improving Models with Narratives
      subesctions:
      - name: Sentiment-Based Prediction
      - name: Embeddings for Time-Series Prediction
      - name: Predicting Using LLM Analyses
  - name: What Can We Take Away?
  - name: Citations

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



# What is Narrative Economics?

Narrative Economics is the study of how popular stories and ideas (aka **narratives**) formed about the state of the economy could have real effects in the world. In this context, a “narrative” is a belief about the state of the world that is shared by the population, regardless of the actual state of the economy. For example, a narrative might be the belief that housing prices are increasing, whereas in-reality (according to economic indicators) they are stagnating. 

The central idea is that the spread of viral narratives can influence individual and collective economic behavior, leading to fluctuations in markets, changes in investment patterns, and even broader economic shifts.

This term is heavily attributed to Robert J. Shiller, a Nobel laureate economist and the founder of Narrative Economics which defined it as:

> *Spread through the public in the form of popular stories, **ideas can go viral and move markets** —whether it’s the belief that tech stocks can only go up, that housing prices never fall, or that some firms are too big to fail. Whether true or false, stories like these—transmitted by word of mouth, by the news media, and increasingly by social media—**drive the economy by driving our decisions about how and where to invest, how much to spend and save, and more.***  
  *- Robert J. Shiller*
  TODO: cite

## The Ambiguity of the Term "Narrative"

The term "narrative" itself has different connotations in NLP compared to economics, which might lead to some confusion. 

In NLP, narrative is  XXX TODO.

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



## Social Media Narratives

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
        {% include figure.html path="assets/img/2024-05-07-myproject/relatio_plot.jpg" class="img-fluid z-depth-2" style="width: 60%;" %}
    </div>

### LLMs Can Understand Narratives
A more advanced technique to extract and analyze narratives is using LLMs. Prompting ChatGPT (TODO: cite) with monthly tweets and prices of economic indicator from matching dates, we generated LLM-based narratives analysis, one for each date in the post-2021 dataset, containing a component of summarized analysis of the tweets and a component of potential effect on the given financial indicator. 

Here's a snippet of such an LLM-based narrative analysis for inputs of dates 29/08/2022 to 28/09/2022. In this time period the Federal Reserve raised the interest rates in an effort to combat inflation, the US Supreme Court ruled that the Biden administration could not extend the pause on student loan payments:

<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/chatgpt_snippet.jpg" class="img-fluid z-depth-2" %}
    </div>

This snippet demonstrates the LLM’s ability to aggregate information, condensing and distinguishing between opinions and occurrences conveyed in the tweets. Moreover, the LLM links its insights to potential future consequences for the financial indicator, a pivotal initial move towards prediction.

# Why Macroeconomics?

Now that we have social media narratives in hand, let's focus on *macroeconomics*.  

**Macroeconomics** studies the behavior of the economy as a whole, examining factors like inflation, unemployment, and economic growth. **Microeconomics**, on the other hand, is concerned with the decision-making of individuals and firms, examining indicators like a certain stock.

A core concept in Narrative Economics is that narratives can drive economics flunctuations. This is especially intriguing at the macroeconomic level, as the theory suggests that widely shared stories can influence the collective decisions of millions of individuals. Additionaly, existing research focuses on microeconomic indicators within the context of Narrative Economics, while the application in macroeconomics remains relatively unexplored. TODO:cite

However, studying this macroeconomically is more complax than microeconomically due to the complex interplay of various factors, the need for broad-covering narratives, and the inherent difficulty in isolating causal relationships. 

## Our Macroeconomic Indicators

We focus on predicting three key macroeconomic indicators:

**Federal Funds Rate (FFR):** The interest rate at which depository institutions, such as banks, lend reserve balances overnight to meet reserve requirements. The FFR serves as a Federal Reserve monetary policy tool, is influenced by public perception of economic stability, and its fluctuations impact various sectors, making it widely monitored.

**S&P 500:** A stock market index measuring the performance of the 500 largest publicly traded companies in the U.S. It reflects collective investor confidence in economic growth and risk appetite and is widely regarded as a barometer of the overall health of the US stock market. 

**CBOE Volatility Index (VIX):** Measures market expectations of future volatility based on S&P 500 options prices, often referred to as the 'fear gauge' as it tends to rise during market stress and fall during market stability.

These indicators are well-suited for testing the predictive power of narratives for macroeconomics due to their daily frequency, which aligns with the rapid pace of Twitter, and their sensitivity to public sentiment and behavior.


# Connecting the Dots: Testing Narratives' Effectiveness for Macroeconomic Forecasting

The previous two sections discussed the theory of Narrative Economics and our curated Twitter dataset, which holds narratives within them, and the distinction between macroeconomics and microeconomics, highlighting why it is interesting to research the theory at the macroeconomic level and what macroeconomic indicators we chose.

We can now delve into the series of experiments we tested to assess the central question - **can economic narratives provide valuable insights for future macroeconomic movements?**

Each experiment tests the predictive power of narratives from the curated datasets, for macroeconomic prediction of one (or more) of the financial targets intriduced before: FFR, S&P 500, and VIX. 

We won't be able to cover all the details of the experiments in this blog, but it is available in our paper. 

## Experimental Setup

### Prediction Tasks

We test the predictive power of narratives on three tasks commonly used in macroeconomic literature (TODO:cite):
* Next value: predicts the target’s value at the specified horizon.
* Percentage change: predicts the target’s percentage change between the specified horizon and
the day before.
* Direction change: classifies the target’s direction of change (increase or decrease) between the
specified horizon and the day before.


### Models Categories

We differ our models into 3 categories based on their input signal:
* Financial (F): utilizes historical financial data, from the past week or month.
* Textual (T): leverages solely textual data, either raw tweets or tweets’ analyses.
* Textual & Financial (TF): draws upon both textual and financial data as input.

  Our goal is to effectively leverage insights from both textual narratives and historical financial patterns to improve prediction accuracy.  The added value of incorporating textual narratives can be demonstrated if a model that utilizes both text and financial data (TF model) outperforms a model that relies solely on financial data (F model). 

### Baselines

**Financial baselines:**
* As/inverse-previous: next value is the same/inverse as previous.
* majority: next value is the majority vote of the previous week/training data.
* up/down: always predict "increase"/"decrease".

**Counterfactual textual baselines:**
* random texts: feeding the LLM with randomly generated sentences comprised of varying random words. This baseline evaluate wether the LLM actually utilize the content of tweets.
* Shuffled tweets: feeding the LLM with chronologically disordered tweets, isolating the impact of temporal narratives from confounding patterns or memoraization. This baseline assess the model reliance on temporal narratives.
* Synthetic 'narratives': Fedding the LLM with generated narrative-like sentences experssing positive or negative cues, aligned with subsequent changes in the financial target. This baseline assess the LLM's ability to infer relationships between aligned narratives and the following market changes.

### Models

Our model selection progresses from simpler models, frequently employed in the financial literature (TODO:cite), to more advanced architectures. This progression serves two purposes: 
1. Achieving positive results with simpler models provides a stronger evidence for the predictive signal of narratives.
2. It allows us to build upon existing research in Narrative Economics, which is primarily rooted in finance and often utilizes relatively simple models, before exploring more advanced NLP approaches.

**Financial models:** these include traditional ML models (e.g., Linear Regression, SVM), DA_RNN (TODO:cite), and T5 (TODO:cite) which receives financial input in a text format. Each model is fed with a sequence of historical financial values of the target indicator, either as individual features per day or as a time-series.  

**Textual models:**
* Daily sentiment: a simple method, commonly used in the literature, is presenting each tweet with its sentiment score. Then, we average the scores of individual tweets of the same dates to recive a daily sentiment, and concatenate over a week.
<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/models_diagram_1.jpg" class="img-fluid rounded z-depth-1" style="width: 50%;" %}
    </div>
    
* LLM's representations of individual/joint tweets: we derive embeddings of individual or concatenated tweets using pre-trained languge models (BERT,RoBERTa and T5 TODO:cite). In the individual case, tweets' embeddings are aggregated daily by averaging or concatenating embeddings of same-date tweets. In the joint case, tweets are concatenated daily to create a single daily embedding, potentially capturing their collective meaning without explicit aggregation, avoiding potential information loss.
<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/models_diagram_2.jpg" class="img-fluid rounded z-depth-1" style="width: 50%;" %}
    </div>
  
* LLM-generated analyses for prediction/as input to a subsequent prediction model: First, we feed GPT-3.5 (TODO:cite) with a month of tweets and corresponding financial values of the target indicator to create monthly analyses.Then, these analyses are either used directly for prediction or as an input to a subsequent T5 model.  
*since the LLM receives both tweets and financial data to enable analyzing relationships, this method applies only for a TF model type.
<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/models_diagram_3.jpg" class="img-fluid rounded z-depth-1" style="width: 50%;" %}
    </div>



**Fusing textual and financial models:** we experiment with several strategies for combining the representations from the T and F models for a unified prediction:
* concatenation: the simplest approach is concatenating the T and F representations.
* DA-RNN(TODO:cite): The dual-stage attention-based RNN model predicts the current value of a time-series based on its previous values and those of exogenous series. We feed historical financial representations (F) as the time series and
textual representations (T) as the exogenous series.
* Prompt-based fusion: LLM-based analysis of given tweets and historical financial values of the target indicator are fed together with raw historical values of the target to a T5 model as separate segments.

  Given a TF model we can derive a T or F model by ommiting or zeroing either F or T component, respectively.



## The Challenges in Improving Models with Narratives

TL;DR:
* Models incorporating narratives (TF) show limited improvement over those using solely financial data (F).
* Gains were inconsistent, marginal and statistically insignificant -> we regard it as negative results.

### Sentiment-Based Prediction:

We fed classic ML models with daily sentiments for FFR 'next value' and 'direction change' prediction (as separate tasks).

| Type | Model | Accuracy |
|---|---|---|
| F baselines | As-previous | 0.812 |
| F | Random Forest Numeric | **0.936** | 
| TF | Random Forest Numeric | **0.939** | 
| T | Logistic Regression | 0.885 |
| T | SVM | 0.885 |


In classifying 'direction change', models with financial input (F & TF) outperform text-only models (T), with a 5% accuracy improvement (0.94 vs. 0.89). Models with both textual and financial signals (TF) achived similar accuracy to F models (0.939 vs. 0.936), suggesting that text has little impact. The T models achieve comparable accuracy to the F baselines (0.89 vs. 0.81).

| **Type** | **Model** | **MSE** |
|---|---|---|
| F baseline | Train-mean | 15.661 |
| F | SVM | 15.416 |
| TF | SVM | 15.416 | 
| T | SVM | 15.36 | 

In predicting 'next value', best F, TF and T models yield comparable MSE to the 'train-mean' baseline (15.4, 15.6).

**What can we learn?** Sentiment analysis, lacks the nuance necessary for accurate financial prediction, and traditional ML models have limitations in capturing complex market dynamics.  This is evident as several models did not even outperform non-learned baselines, highlighting the need for more improved text representations and advanced prediction models.  


### Embeddings for Time-Series Prediction:

Here we turn to embedding-representations (as explained in the Experimental Setup) and to DA-RNN (TODO:cite) model, which is designed to capture temporal dynamics and complex relationships within its given data.

We extensively evaluated various model configurations, targer indicators (FFR and VIX), tasks ('next value', 'percentage change', 'direction change' and the last two together), prediction horizons (next-day, next-week), LLM architectures (see Experimental Setup), aggregation methods, and the daily number of tweets given as input. Additionaly, we assessed the models' reliance on temporal context and relevant narratives using the counerfactual textual baselines.

To keep it short, we present results only for predicting the VIX 'next value' of the next-day and next-week (as separate tasks). Additional experiments showed a recurring pattern to the presented results.

<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-myproject/embeddings_exp_results.jpg" class="img-fluid rounded z-depth-1" style="width: 80%;" %}
    </div>

For the next-day horizon, the 'as-previous' F baseline outperforms all other models (3.079 MSE). This suggests that the input information might not be beneficial for such a short-term prediction. 

For the next-week horizon, initially both TF models (13.148, 13.147) appeared to outperform the F model (13.463) and F baseline (16.172), implying a potential influence of the textual content. 
However, the 'random texts' TF baseline (13.056), which replaced actual tweets with random texts, outperformed all others, indicating the improvement was not due to meaningful textual content. We hypothesize that the presence of text improves performance, even when random, due to spurious correlations or random noise aiding generalization, similar to regularization techniques. 

A contributing factor may arise from the difficulty of effectively capturing and representing aggregated tweet information for financial prediction, as well as the inherent challenges in predicting future values of a volatile financial indicator, characterized by frequent random movements and fluctuations, using its historical values.    

**What can we learn?** Our models struggled to leverage tweets for the prediction, indicating that implicitly capturing and aggregating latent narratives within LLMs remains a challenge.

### Predicting Using LLM Analyses:

**Can LLMs generate an accurate prediction?** We first tried to directly predict the financial indicator (average weekly VIX or S&P 500) as a generative response of the chat version of GPT (TODO: cite) prompted with a monthly window of tweets and corresponding values of the financial target. This resulted in limited inconsistent success. While the LLM consistently generated insightful narrative analyses and demonstrated comprehension of financial implications, it exhibited inconsistencies when applying these insights for prediction.  For instance, it occasionally refused to provide predictions, or it would simply mirror input ranges, neglecting the potential impact of the narratives it successfully analyzed. When presented with 'synthetic narratives', it recognized the change direction but struggled to quantify the magnitude of it.

**Repurpusing the LLMs' analyses for a subsequent prediction model:** The previous experiment revealed the LLM's ability to generate insightful analyses of tweets and financial data. To leverage this capability, we utilize these analyses as inputs for a dedicated prediction model to predict the S&P 500 'direction change'.

This approach addresses limitations of both previous experiments:
* Instead of relying on the embedding-based approach, which struggled to aggregate diverse narratives, we leverage the LLM's ability to produce concise analyses.   
* Instead of directly using the LLM for prediction, which exhibited inconsistencies, we utilize a separate fine-tuned model for downstream prediction.  

| Type | Model | Accuracy | F<sub>1</sub>-Score | 
|---|---|---|---|
| F-baselines | Train-majority | 0.424 | 0.0 |  
| F-baselines | Week-majority | 0.484 | 0.598 | 
| F-baselines | As-previous | 0.484 | 0.552 | 
| F-baselines | Inverse-previous | 0.517 | 0.511 | 
| F-baselines | Up-predictor | 0.576 | 0.731 | 
| F-baselines | Down-predictor | 0.424 | 0.0 |  
|-----|-----|-----|-----| 
| F | T5 Base | **0.604** | 0.723 |  
| F | T5 Large | 0.593 | **0.727** | 
|---|---|---|---| 
| TF | T5 Base | 0.626 | 0.738 |  
| TF | T5 Large | **0.627** | **0.742** |  
|---|---|---|---| 
| T | T5 Large | 0.587 | 0.726 | 
|---|---|---|---| 
| T-baseline | Synthetic narratives | 0.489 | 0.254 |


So did it work? not really. Results show that there is no significant difference between the best TF and F models, with a performance gap of ~2% on the limited test set of ~90 samples. <d-footnote>As a reminder, we can only use the second Twitter dataset, of tweets that were posted after the LLM's training cutoff date, and our financial indicators are of daily frequency, therefore the small dataset for this type of experiments.</d-footnote>
We confirmed it using the McNemar's test (TODO:cite) which shows no statistically significant difference (p-value=0.48).
On the good side, this is the only approach were our models surpass all baselines. 

**What can we learn?** While TF and F models outperform all others, the difference between their performance is negligible.


# What Can We Take Away?

Despite the presence of narratives in our curated datasets and the development of NLP tools for narrative extraction, evaluating their impact on macroeconomic prediction remains challenging.  Our models incorporating narrative data showed limited improvement over those using only financial data, failing to consistently outperform baselines or financial models.  Any observed improvements were marginal and statistically insignificant and we regard it as a negative result.

The missing link between the successful narrative extraction demonstrated by the LLM's analyses and the limited improvement in macroeconomic prediction raises questions about the extent to which narratives alone can truly drive and forecast economic fluctuations, at least at the macroeconomic level.

This study serves as a foundation for further exploration, highlighting the need for new macroeconomic models or
tasks designed to assess the extracted narratives’ influence on the economy.

TODO: link to paper.

# Citations

bla



