---
title: "Sub Component Embedding: Reducing Dimensionality and Improving Flexibility"
date: 2024-06-20T16:04:06-05:00
tags: ["projects", "machine learning", "data visualization"]
author: ["Nick Barlow"]
---

VectoredIn in a tool I developed to visualize the job market and job postings from LinkedIn utalising various different tool and techniques from the NLP and LLM space. To read more about the project, look [here]({{< ref "vectoredIn.md" >}})


###  Data

For this project, I built upon the data from this Kaggle dataset  [here](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings), which originally contained over 100,000 job postings, and I bought this up to just over 1 Million individual job postings.

Now, with each job posting having an average of 500 words (600 Tokens) for the description alone, alongside some additional metadata, this is a reasonbly large dataset.

The main goal of this project was to use LLM's to create a vector representation of each individual job posting, and store these in a open-source vector database from Weaviate. The way a classical embedding process works is you feed in a set of text (Document, Article, Paper) and you get out a vector representation of the text. This vector repreresents the semantic meaning behind the text, with similar articles of text being closer to each other in the vector space.

This is all well and good, but having to embed over 600 Millions tokens (if we were simply to feed in each full job posting to the model) is pretty inefficient and costly. So let's have a look at some of the techniques that can be used to reduce the dimensionality of the data.

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

Sub Component Embedding builds on our NER technique above, and involves taking the output of the NER model and using it to create a new embedding for each sub component. All of these individual embeddings can then be used to create a Composite Embedding Vector (CEV) for the entire job posting. This presenst us with three major benefits:

1. Reducing in data, again. As we now only have to create embeddings for each unique sub component, we can greatly reduce the amount of data we have to send through a large language model. Even further than this, we have also changed the way this scales with amount of data, scalling logarithmically.
2. Flexibility. Having the ability to recompile an embedding from a range of subcomponents means that we have in the way we combine the subcomponents, whether they are all weighted equally, or a weighted relative to their classification etc.
3. Visualisation. Having the ability to visualise the semantic distance between each subcomponent, and the overall job posting, we can see how each of the subcomponents relate to the job posting, and the axis we are visualising them on.

<img src="/img/sub_component3.png" alt="3d Plot" width="900">

Looking at how the amount of token's needed to embed for each of these method, we can see the impact of our techniques, and how the sub component embedding method is the most efficient. At 1 Million Job postings, embedding the full NER of the job posting uses 30% of the tokens that embedding the full description needs. Futher, the sub component embedding method uses only 6% of the tokens.

<div style="max-width: 800px; margin: 0 auto;">
    <iframe src="/embedding_methods.html" width="900" height="600" style="display: block; margin: 0 auto;"></iframe>
</div>


### Visualisation

One of the main benefits of utalising sub component embedding is that when we are visualising the semantic distance between each of the subcomponents, we can see how they relate to the overall job posting. This is shown in the diagram below, where we can see that the subcomponents are all related to the job posting, and that the subcomponents are related to each other.


<div style="max-width: 900px; margin: 0 auto;">
    <iframe src="/fullplot_tax.html" width="900" height = "650" style="display: block; margin: 0 auto 20px;"></iframe>
    <iframe src="/sub_component_plot_tax.html" width="900" height = "650" style="display: block; margin: 0 auto;"></iframe>
</div>

The first plot here is the main plot of job postings, and their relative relations to the 3 user defined axis. The point highlighted in red is the job posting we are looking at, and in the second plot we can see the subcomponents of the job posting, and their relative relations to the 3 user defined axis, along with the composite embedding vector of the job posting.

For example we can see that one of the subcomponents **Expeirence in data managment and automation** is more related to the **Data Science** and **Machine Learning Engineer** than the **Accountant** axis, despite this being a job posting very related to a **Accountant** role. But the composite embedding takes into account all the subcompoents, including some very related to the **Accountant** role, such as **Bachelors Degree in Accounting or Business related field**. 

An important point to note here, is that while we talk about the Composite Embedding Vector (CEV) as an average of the Sub Component Embeddings (SCE), we do not expect the cosine distance of the CEV to be the average of the cosine distance of the SCE's. This is because the CEV represents a new point in the high-dimensional embedding space that captures the aggregate semantic meaning of all sub-components. All the vectors here are represented in a vector space of 1536 dimensions, from the new OpenAI `'text-embedding-3-small'` model.


### Compositing Embedding Vectors

As mentioned above, when we are compositing a new vector from SCE's, there are many different methods we can use. The simplest of which would be to take the average of all the SCE's as they are, and treat them all equally. 


Here's an example of the SCE's for the job posting we visualised above.
```python
{    
    'embeddings': {
        'company_name': {
            'Deloitte': array([0.0426881, -0.03636974, 0.02759714, ... ])
        },
        'entities_EXPERIENCE': {
            "3+ years' experience in federal partnership tax compliance...": array([-0.0206065, 0.0104712, 0.06552737, ... ]),
            'Experience with data management and automation tools ': array([-0.02074211, 0.02936463, 0.06454881, ... ]),
            'Experience working in a fast-paced, team environment ..': array([-0.05655309, 0.01771902, 0.01467823, ... ])
        },
        'entities_QUALIFICATIONS': {
          'Bachelors degree in accounting or business-related field ': array([-0.02273333, 0.01862575, 0.04938539, ... ]),
        }
        'entities_RESPONSABILITY': {
            'Aptitude in Microsoft Office, especially Microsoft Excel': array([0.01134986, 0.00715933, 0.06455616, ... ]),
            'Lead client engagement teams and client delivery ': array([0.03342095, 0.01799032, 0.10103559, ... ]),
            'Lead the development of new features and ... ': array([0.01359465, -0.00747398, 0.0514259, ... ]),
            'of the current range is $82,880 to $175,500.': array([0.01902576, 0.01339932, 0.09241084, ... ]),
            'provide you with a great work/life fit and ...': array([0.00955491, 0.01953559, 0.04167593, ... ])
        },
        'entities_TOOLS': {
            'Deloitte Tax': array([0.01264361, 0.01308516, 0.03882795, ... ])
        },
        'industry_names': {
            'Accounting': array([-0.02227813, -0.02324294, 0.08319844, ... ]),
            'Business Consulting and Services': array([-0.03887666, 0.00595183, 0.0723723, ... ]),
            'IT Services and IT Consulting': array([-0.04406529, -0.01873075, 0.06618199, ... ])
        },
        'job_functions': {
            'Accounting/Auditing': array([-0.05215971, 0.01514608, 0.04201899, ... ])
        },
        'title': {
            'Tax Senior - PSG - Investment Management Reporting ...': array([-0.02505, 0.04683963, 0.05933352, ... ])
        }
    }
}
```

So to achieve the simplest method of composition, we would take the average of all the SCE's, and treat them all equally.

```python
import numpy as np

# Extract all the arrays
all_arrays = []
for category in embeddings.values():
    for embedding in category.values():
        all_arrays.append(embedding)

# Stack the arrays
stacked_arrays = np.stack(all_arrays)

# Calculate the mean
composite_embedding_vector = np.mean(stacked_arrays, axis=0)
```

This approach does work and provide reasonable results for examples like this, but what we gain in simplicity here, we lose in peformance. While not the case in this example, there's are many job listings were the balance between the categories is very heavily skewed in one direction. For example a job posting for a **Full Stack Developer** may only list 3 responsibilities, while listing over 20 tools and frameworks. A simple fix for this is to take the average of the SCE's for each category, and then take the average of the resulting vectors to create our CEV.

```python
import numpy as np

# Extract all the arrays
all_cat_means = []
for category in embeddings.values():
    # Extract all the arrays for each category
    category_embeddings = []
    for embedding in category.values():
        category_embeddings.append(embedding)
    # Calculate the mean of the arrays for each category
    category_embedding = np.stack((category_embeddings).mean(axis=0))
    all_cat_means.append(category_embedding)

# Stack the arrays
stacked_arrays = np.stack(all_cat_means)

# Calculate the mean
composite_embedding_vector = np.mean(stacked_arrays, axis=0)
```

The third approach is we can build on our second approach here and add in some manual weight's to each of the categories. This is a bit more complex, but can be a good way to get a good balance between the categories. The only important thing to note here is that all the weights here need to add up to 1, and not every category is required to be present in every listing.

```python
import numpy as np
weights = {
  title: 5,
  company_name: 1,
  industry_names: 1,
  job_functions: 2.5,
  entities_RESPONSABILITY: 2,
  entities_TOOLS: 2,
  entities_QUALIFICATIONS: 2,
  ...
}

# Extract all the arrays
all_cat_means = []
category_weights = []
for category in embeddings.values():
    # Extract all the arrays for each category
    cateogry_weight = weights[category]
    category_embeddings = []
    for embedding in category.values():
        category_embeddings.append(embedding)
    # Calculate the mean of the arrays for each category
    category_embedding = np.stack((category_embeddings).mean(axis=0))*category_weight
    all_cat_means.append(category_embedding)
    category_weights.append(category_weight)

# Stack the arrays
stacked_arrays = np.stack(all_cat_means)

# Calculate the mean
composite_embedding_vector = np.mean(stacked_arrays, axis=0)/np.sum(category_weights)
```

The approach we took in this project is the second one, as it gives us a good balance between simplicity and performance. But we will get into a comparison of the different approaches, how they influence the results, and what metrics we can use to evaluate them in the next post.

