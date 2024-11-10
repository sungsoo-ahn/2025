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

<blockquote>
  Spread through the public in the form of popular stories, **ideas can go viral and move markets** —whether it’s the belief that tech stocks can only go up, that housing prices never fall, or that some firms are too big to fail. Whether true or false, stories like these—transmitted by word of mouth, by the news media, and increasingly by social media—**drive the economy by driving our decisions about how and where to invest, how much to spend and save, and more.**  
  - Robert J. Shiller
  TODO: cite
</blockquote>

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



## Social Media Narratives

We have collected two comprehensive datasets of tweets from Twitter (X), a platform chosen for its real-time reflection of public opinions and ability to capture diverse economic narratives. These datasets cover a broad time-frame, and a wide range of topics relevant to the economy, including economics, business, politics, and current events, ensuring a broad and comprehensive representation of narratives.  

Both datasets were carefully curated using targeted queries with the inclusion of specific keywords, and were analyzed to ensure quality and relevant for capturing economic narratives. 

**Pre-Pandemic Twitter Dataset:** Utilizing Twitter API (TODO: cite), we collected 2.4 million tweets from Twitter's early days (January 2007) to COVID-19 pandemic (December 2020). To prioritize viral tweets, we retrieved the daily top 200 tweets based on follower count,  then we randomly sampled 100 to mitigate potential bias towards highly active accounts typically associated with news outlets. This process yeilded a dataset contributed by about 250,000 users per collected topic, each boasting an average follower count of 100 million, including global leaders, news outlets and other influencers.

**Post-2021 Twitter Dataset:** We wanted to test the predictive power of our employed LLM (Chat Completion API with GPT-3.5 TODO:cite) on a timeframe beyond the LLM's data cutoff date (September 2021). Therefore, we collected a second dataset ranging from September 2021 until July 2023. This assures the LLM relies solely on provided tweets and pre-existing knowledge, unaware of "future" knowledge. Here we collected the tweets monthly, using Twitter Advanced Search, restricting to users with at least 1,000 followers. Overall we curated 2,881 tweets <d-footnote>As this data collection method is more restricted than the
previous, the resulting dataset is relatively smaller.</d-footnote> contributed by 1,255 users including politicians, CEOs, activists, and academics. 

### Did We Really Collected Narratives?
We analyzed the tweets to assure the presence of narratives within them. First, we utilized RELATIO, a tool designed to "capture political and economic narratives" by mapping relationships and interactions among entities in a corpus (TODO:cite). We feed the algorithm with our Twitter dataset, receive “narrative statements” (as defined in RELATIO paper), and visualize their temporal distribution:

<div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-myproject/relatio_plot.png" class="img-fluid z-depth-2" %}
    </div>


### Why Macroeconomics?

bla

## Testing Narratives' Effectiveness for Macroeconomic Forecasting

### LLMs Can Learn Narratives

### The Challenges in Improving Models with Narratives

## What Can We Take Away?

## Citations

bla


## Tweets

An example of displaying a tweet:
{% twitter https://twitter.com/rubygems/status/518821243320287232 %}

An example of pulling from a timeline:
{% twitter https://twitter.com/jekyllrb maxwidth=500 limit=3 %}

For more details on using the plugin visit: [jekyll-twitter-plugin](https://github.com/rob-murray/jekyll-twitter-plugin)


## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>


