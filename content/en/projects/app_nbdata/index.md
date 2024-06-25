---
title: "VectoredIn"
weight: 1
resources:
    - src: "/img/3dplot.png"
      params:
          weight: -100
project_timeframe: "2024"
---

{{< figure src="img/Plot_example.png" width="1200x" >}}

This project showcases the application hosted at [app.nbdata.co](https://app.nbdata.co). 

VectoredIn is a web application developed as a submission for the Weaviate Machine Learning Engineer challenge. It's designed to analyze and visualize job market data using advanced natural language processing and machine learning techniques.

Key features include:

3D Plot of Semantic Distance: An interactive visualization that allows users to explore job listings in a three-dimensional space based on user-defined semantic axes.

RAG (Retrieval-Augmented Generation) Search: Enhances large language model queries by providing additional context from the job data.

Plot Summary: Generates a summary of job listings, highlighting similarities and differences using Weaviate's generative search and OpenAI's language model.

Axis Alignment: Allows users to select specific points in the 3D scatter plot and view a detailed comparison of how well the selected job aligns with the three chosen axes.

HNSW & ANN Implementation: Utilizes the Hierarchical Navigable Small World algorithm for efficient similarity search, with custom modifications to return a range of results with varying levels of similarity.

The project is built using Python, Django, and JavaScript, with Plotly for visualizations and Weaviate for vector storage and retrieval. It demonstrates advanced techniques in semantic search, data visualization, and natural language processing applied to job market analysis.