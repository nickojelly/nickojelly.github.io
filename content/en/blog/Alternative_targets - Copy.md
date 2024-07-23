---
title: "Optimizing Targets for Predictive Models"
date: 2023-06-01T16:04:06-05:00
tags: ["features", "blog"]
series: ["quickstart"]
author: ["Nick Barlow"]
---


I was recently awarded a prize in a Datathon hosted by Betfair based around predicting the outcome of greyhound races using Machine Learning. I learnt a great deal over this experience and one piece of this was around optimizing the target's used in training a ML model. The goal of the competition was simple, to have the lowest log loss between your predictions and the outcome of the races, a standard classification task. 

One of the more interesting points of discovery was around the choice of target's when training a model. The obvious choice for a target for this 'classification' model would be a one-hot encoded win, however as we will explore, there are some challenges along this path that add some complexity to a simple choice. Because we are stepping slightly outside the bounds of a classical 'classification' problem, we need to consider the fact that we are not just predicting a single class, but a probability distribution over all the classes. For example if we have a look at this result from a race in NZ:

![Close Heat](/img/Closeheat.png)

The outcome of this race is that the dog with rug 3 is the winner, by a nose. But this outcome is so close that a photo is required to split the hairs between the two, so here our idea of using a simple one-hot encoded model breaks down, because in essence greyhound racing has some degree of randomness. That is if this race was repeated over and over again, I believe the result would change, and using a one-hot encoded target would not account for how close this is.

This blog is an extension of a previous blog I made about using multiple targets at the same time to counteract this issue: [Multiple Loss Targets](/blog/multiple_loss_targets/). In that post we define 3 options for the target:

1. Margin as a softmin tensor: The margin of victory in a race is a direct measure of how well each dog performed. By converting this margin to a softmin tensor, we transform the margins into a set of probabilities that sum to 1. The dog with the smallest margin (i.e., the winner) will have the highest probability. This target tensor directly relates to the outcome of the race and is crucial for the model to learn the relationship between the input features and the race outcome.

2. One-hot encoded win: This tensor represents the actual winner of the race. Each dog is represented by a binary value in the tensor, with the winner having a value of 1 and all others having a value of 0. This target tensor is essential for the model to learn to predict the categorical outcome of the race, i.e., which dog will win. It provides a clear and unambiguous target for the model to aim for.

3. Starting prices as probabilities: The starting prices in a race can be seen as the inverse of the market's perceived probability of each dog winning. By converting these prices to probabilities, we create a target tensor that represents the collective opinion of the betting market. This tensor is important for the model to learn to take into account the same factors that bettors consider when placing their bets. It provides a different perspective on the race outcome, complementing the other two target tensors.

So lets explore the behavior of model trained when using a single one of these models:

