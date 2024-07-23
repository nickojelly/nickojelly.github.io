---
title: "Sub Component Embedding: Reducing Dimensionality and Improving Flexibility"
date: 2024-06-20T16:04:06-05:00
tags: ["projects", "machine learning", "data visualization"]
author: ["Nick Barlow"]
---

VectoredIn in a tool I developed to visualize the job market and job postings from LinkedIn utalising various different tool and techniques from the NLP and LLM space. To read more about the project, look [here]({{< ref "vectoredIn.md" >}})


###  Data

For this project, I built upon the data from this Kaggle dataset  [here](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings), which originally contained over 100,000 job postings, and I bought this up to just over 1 Million individual job postings.

Now, with each job posting having an average of 500 words for the description alone, alongside some additional metadata, this is a reasonbly large dataset.

The main goal of this project was to use LLM's to create a vector representation of each individual job posting, and store these in a open-source vector database from Weaviate. The way a classical embedding process works is you feed in a set of text (Document, Article, Paper) and you get out a vector representation of the text. This vector repreresents the semantic meaning behind the text, with similar articles of text being closer to each other in the vector space.

This is all well and good, but having to embed over 500 Millions tokens (if we were simply to feed in each full job posting to the model) is pretty inefficient and costly. So let's have a look at some of the techniques that can be used to reduce the dimensionality of the data.

#### Anatomy of a Job Posting




<div style="display: flex; align-items: center;">
  <div style="flex: 1;">
Job posting's as a dataset fall into the semi-structured format of data. They are all quite different from each other, but if we zoom out they all seem to be made up of the same sub components. We can expect a job posting to have a title, responsibility, skills, qualifications, tools, all in varying amount. This is alongside some additional text, that is not really relevant to the actual role, but more contextual to the company and the hiring process. 

The diagram here show's roughly what we would expect to see when we are looking at a job posting, and looking though the lense of trying to reduce the amount of data, we can see a fair bit of text that can be removed, such an general information about the company, the hiring process, and fair employment compliance information. While this information may be relevant when applying for a job, it is not really relevant to the role itself. So let's look at some techniques we can use to extract the key information we need.

  </div>
  <div style="flex: 1;">
    <img src="/img/job_anat_crop2.png" alt="Job Anatomy" width="350">
  </div>
</div>


### Techniques

There are several techniques that can be used to extract the key information from a body of text, the most popular being the Named Entity Recognition (NER). NER involves training a model, (LLM or in this case a statistical model) to identify and classify named entities in a text. The NER model we have used and finetuned is from the [spaCy](https://spacy.io/api/entityrecognizer) library, which is a popular NLP library.

#### NER

We defined a set of custom classes for this NER to encompass the relevant information we talked about above, so building on our diagram from above, this is a representation of what we are doing with this model.

<img src="/img/job_ner3.png" alt="3d Plot" width="900">

The benefits of this are twofold:
1. Reducing the amount of data. The amount of data we extracted amounts to a little over 25% of the original data, on average over the 1 Million job postings we have.
2. Removed unrelated information. We have removed a fair amount of the text that is not pertinent to the underlying role, which should increase the quality of the embeddings.

#### Sub Component Embedding

Sub Component Embedding builds on our NER technique above, and involves taking the output of the NER model and using it to create a new embedding for each sub component. All of these individual embeddings can then be used to create a new embedding for the entire job posting. This presenst us with three major benefits:

1. Reducing in data, again. As we now only have to create embeddings for each unique sub component, we can greatly reduce the amount of data we have to send through a large language model. Even further than this, we have also changed the way this scales with amount of data, scalling logarithmically.
2. Flexibility. Having the ability to recompile an embedding from a range of subcomponents means that we have in the way we combine the subcomponents, whether they are all weighted equally, or a weighted relative to their classification etc.
3. Visualisation. Having the ability to visualise the semantic distance between each subcomponent, and the overall job posting, we can see how each of the subcomponents relate to the job posting, and the axis we are visualising them on.

<img src="/img/sub_component3.png" alt="3d Plot" width="900">

Looking at how the amount of token's needed to embed for each of these method, we can see the impact of our techniques, and how the sub component embedding method is the most efficient. At 1 Million Job postings, embedding the full NER of the job posting uses 33% of the tokens that embedding the full description needs. Futher, the sub component embedding method uses only 6.5% of the tokens.

<div style="max-width: 800px; margin: 0 auto;">
    <iframe src="/embedding_methods.html" width="900" height="600" style="display: block; margin: 0 auto;"></iframe>
</div>

### Visualisation

One of the main benefits of utalising sub component embedding is that when we are visualising the semantic distance between each of the subcomponents, we can see how they relate to the overall job posting. This is shown in the diagram below, where we can see that the subcomponents are all related to the job posting, and that the subcomponents are related to each other.

<div style="max-width: 800px; margin: 0 auto;">
    <iframe src="/fullplot_tax.html" width="900" height="900" style="display: block; margin: 0 auto;"></iframe>
</div>

<div style="max-width: 800px; margin: 0 auto;">
    <iframe src="/sub_component_plot_tax.html" width="900" height="900" style="display: block; margin: 0 auto;"></iframe>
</div>