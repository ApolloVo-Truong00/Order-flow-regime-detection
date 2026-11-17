23/09/2026

After the toy model, I had no direction on where to go next. The limitations of the model were obvious, and the extensions to deal with them were quite natural, however, they were grounded in (what I believe was) naive modelling choices rather than backed by literature. Reading Hasbrouck and Cartea et al. was a means to find additional gaps in my toy model that could be used as direction for my next model, with the aim of getting closer to answering the detection problem. Reading provided further intuition on the problem, vocabulary in the industry, and of course, direction and backing for some of the extensions listed in the limitations section. 

After some time reading, I came across the Markov chains in Cartea et al. and immediately saw how the model could be used in my project. I set out to begin building my model. As I was writing the motivation for the Markov chain, a few thoughts hit me. 
1. Prediction vs. detection: I've implicitly considered these to be the same in the sense that we would be able to detect regime changes by predicting when the regime changes, but there's a nuanced difference between the two. The Markov chain doesn't explicitly detect anything unless there's a decision rule. There is a modelling choice here -- we need a mechanism to compare two statistics and have a criteria that defines when a regime shift has occurred instead of simply waiting to observe when the order imbalance enters/transitions into another bin.
2. Markov chain: I'm thinking of varying the probability transition matrix over time, i.e. the Markov chain is inhomogeneous. I could do something like the next-state probability depends only on current order imbalance state. Or, I could force next-state probability to depend on the entire trajectory of order imbalance. The latter is no longer a Markov chain as we condition on the entire history instead of the current state. This may be more informative as we get a rolling window of sorts. As we get more data, the next-state probability updates to take into that new flow of information. 
3. What does regime actually mean? There needs to be some sort of notion of what a regime shift really means - are we looking very granularly by partitioning the order imbalance into fine bins? If so, when we test the model on real data, how are we able to verify the truth? There is no observable data telling us when regimes actually change on a fine level. On the other hand we could partition the order imbalance coarsely; real data that includes known shocks can be used as a proxy to test regime changes. The former is probably closer to the problems a market maker faces than the latter.
4. Time scale: a regime relevant to a market maker might live for only hundreds of milliseconds whilst a regime relevant from a slower view might live for hours. The time scale needs to interact well with the regime partition used. For example if we compare transition matrices, we need enough data to estimate the transition matrix reliably. Hence we'd need a lot of transitions. The issue is that short windows don't give us many transitions.

These are all modelling choices I have to make. I don't think reading more literature will help me come up with a solution. The decisions I make here will change the project drastically. 

24/09/2026

The modelling choices I have to make must weigh the following: feasibility given my knowledge and infrastructure, difficulty, and staying close to the original question (as the original question is what I'm most interested in). 

With that in mind, I think the option to go with a "simple" continuous-time, inhomogeneous Markov chain is the best route. Firstly, Markov chains were one natural extension following the limitations of the toy model; secondly, Cartea et al. explicitly use this model (albeit with different goals); lastly, Markov chains are quite intuitive at a high-level. 

This means that there needs to be an extra mechanism that sits on top of the Markov chain that allows the model to detect when a regime shift has actually occurred. Naively, we need some sort of rule that tells us A --> regime shift and not A --> regime has not changed. Truthfully this is something that I haven't ever come across outside of machine learning where different learning techniques have (relatively) strict decision rules. This detection model is a bit more flexible, with the decision rule being a modelling choice made by me. First glances at some papers shows that this is actually a known field (sequential change detection). 

With the actual foundational model being chosen, I'll place my focus towards reading on different detection methods in general and in the case of Markov chains to gain better understanding on the methods so that the eventual choice isn't arbitrary but backed by intuition and literature. 

From where I stand, I see the start (Markov chain) and end (detection choice) of the model. Whatever additional modelling choices I may have to make in the intermediate stages will be considered when I get there, one being point 4 (time scale).

30/09/2026

After reading a survey on detection methods, I think I have a basic, high-level understanding of how it fits into my project. There are a few important mechanisms for the detection part of my model: the statistic used to define the decision rule, and the threshold used to compare the statistic. Both of these come together to give a decision rule, i.e. it tells us when our model should tell us that the market regime has changed by comparing the latest statistic to the threshold. The way we choose a rule and threshold is via an optimisation problem. An important idea is the balance between speed of detection and false alarms -- this is a trade-off. We can, in general, solve the optimisation problem to choose a rule and a threshold, but can also solve the optimisation problem under a family of existing decision rules. The latter makes more sense for this project as we're able to choose existing rules that are set out in the paper. The former is too involved and the point of the project isn't to dive deeply into detection methods. 

The workflow for the detection model itself looks a bit like: choose a rule, solve optimisation under that family of rules to choose a threshold b. 

The overall workflow for the entire model looks like: estimate Markov chain parameters, compute the statistic used for detection, choose rule family, calibrate threshold using optimisation criteria, apply rule continuously.

We can think about this upcoming section like it is like two disconnected parts of a bigger model that we put together. First, focus on the Markov chain model and the detection methods independently. Then we can connect the two parts using the statistic we choose. 

14/11/2026

A central aspect that needs to be addressed is how a market regime change is actually defined. Of course, I plan to use limit order imbalance as a proxy. If I naively also define a market regime change to be a transition from one limit order imbalance state to another, then the detection problem becomes redundant because limit order imbalances are directly observable. This project aims to say something about the underlying, unobservable states in the market using something that is observable. 

The new model I plan to use is a time-homogeneous, continuous-time Markov chain. This model is completely determined by its generator. We can think of the generator as the driving force of the market, modelled by a CTMC. Hence we can assume that different market regimes corresponds to different generators, and changes in market regimes correspond to a change in generators. The change time, say gamma, will be fixed and known, allowing us to have a ground truth -- this is what enables us to determine how well the detection methods perform. 

The resulting model is a piecewise, time-homogeneous CTMC defined by the generators changing at fixed, known times gamma. This model serves as the source of truth. Synthetic data can be generated using the Markov chain. The regime change time is known; this is gamma. 

After thoughts: I think it may be important to also note (in future discussions) other methods of defining a regime other than through the generators. This specific definition uses the actual model to define the regime. What's interesting is that there is a way to change the underlying process through something that isn't directly observable, then try to detect when a regime changes through something that is observable. This may be close to reality where there are unobservable states that drive the process, forcing us to use a proxy to detect when the process has changed in some meaningful way.

17/11/2026

A few modelling additional modelling choices that I am faced with are listed below. 

Before diving into these, there is a practical element that I have been thinking about. This project has quite a lot of intermediate modelling choices that are important. Trying to optimise and rigorously justify each choice is not the objective of the project. Hence, I will try to keep the justification of each modelling choice at a minimum (but hopefully sufficient). 

1. Choosing how the LO imbalance is partitioned. 
- The generator of the CTMC is estimated using the number of transitions and total time spent in each state. The partitions shouldn't be too granular. I want to ensure that the generator can be estimated reliably. 
- I want to also keep the states/partitions as intuitive as possible to maintain economic interpretation. 
- Both of these considerations lead me to choosing something like [-1, -1/3), [-1/3, 1/3], (1/3, 1]. Interpretation is: ask-side dominance, relatively balanced, bid-side dominance (respectively). Additionally, each state has equal width. 

2. Choosing how quoted imbalance V is defined.
- Cartea presents a couple different ways of choosing how V is defined: at the touch (i.e. lowest ask and highest bid), all existing LO, best n levels (take the n distinct lowest asks and n distinct highest bids), orders within n ticks of midprice (fixed price distance around the midprice).
- At the touch is simple, interpretable, but very local as it ignores the distribution of the liquidity deeper in the book. 
- The other three capture more liquidity deeper in the book but they present additional modelling choices of n. 
- From a practical, simplicity, interpretable point of view, I am leaning toward choosing at the touch. 
