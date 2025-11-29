---
title: "Road to become an AI Engineer"
date: "2025-11-29"
description: "Learn how to include media files in your blog posts"
---

I just prompted so that the AI can suggest me the syllabus to learn AI, ML foundation

Day 1: Vectors and Norms

1. Theory Keywords

* Scalars, vectors, matrices notation
* Column vs row vectors
* Vector addition, scalar multiplication
* Dot product, geometric meaning
* Vector norms L1, L2, Linf
* Cauchy–Schwarz inequality
* Distance between vectors
* Cosine similarity basics

2. Hands-on Labs (short specs)

* Lab 1 — Implement vector operations and norms
* Lab 2 — Compare distances and cosine similarities

3. Mini-Project Idea

* Title: Vector Similarity Explorer
* Goal: Build a small tool computing distances and similarities between high-dimensional vectors
* Dataset keywords: random vectors, word embeddings, toy features
* Output keywords: similarity matrix, distance histogram, summary stats

4. Q&A Seeds (Questions Only)

* Q1: How are vector norms related?
* Q2: When does cosine similarity matter more than distance?
* Q3: How to choose a norm for feature scaling?
* Q4: What happens with zero vectors and cosine similarity?
* Q5: Tradeoffs between L1 and L2 norms?

---

Day 2: Matrices and Linear Maps

1. Theory Keywords

* Matrix shapes and notation
* Matrix addition, multiplication rules
* Linear maps and composition
* Identity and zero matrices
* Transpose, symmetric matrices
* Rank and range intuition
* Basis vectors and coordinates

2. Hands-on Labs (short specs)

* Lab 1 — Implement matrix multiplication and tests
* Lab 2 — Visualize linear maps in 2D

3. Mini-Project Idea

* Title: Linear Transform Visualizer
* Goal: Build a tool that applies 2D transformations and visualizes effect on points and grids
* Dataset keywords: synthetic 2D points, grids
* Output keywords: before/after plots, transformation parameters

4. Q&A Seeds (Questions Only)

* Q1: When is matrix multiplication defined?
* Q2: How does composition relate to matrix multiplication?
* Q3: How do basis changes affect coordinates?
* Q4: What if rank is less than dimension?
* Q5: Tradeoffs using different bases?

---

Day 3: Determinants and Inverses

1. Theory Keywords

* Determinant geometric meaning
* Invertible vs singular matrices
* Determinant properties and scaling
* Computing inverse for 2x2, 3x3
* Linear systems Ax = b
* Condition number intuition
* Overdetermined and underdetermined systems

2. Hands-on Labs (short specs)

* Lab 1 — Implement determinant for small matrices
* Lab 2 — Solve linear systems with Gaussian elimination

3. Mini-Project Idea

* Title: Linear System Solver
* Goal: Build a solver for small linear systems with checks for singularity and conditioning
* Dataset keywords: synthetic linear equations, random matrices
* Output keywords: solution vector, residual norms, condition estimates

4. Q&A Seeds (Questions Only)

* Q1: What does determinant zero imply?
* Q2: Why does large condition number matter?
* Q3: How to handle inconsistent linear systems?
* Q4: What if A is nearly singular numerically?
* Q5: Tradeoffs using explicit inverse vs solvers?

---

Day 4: Eigenvalues and Eigenvectors

1. Theory Keywords

* Eigenvalues, eigenvectors definition
* Characteristic polynomial concept
* Diagonalizable matrices
* Spectral decomposition intuition
* Symmetric matrices properties
* Spectral radius and stability
* Applications to dynamical systems

2. Hands-on Labs (short specs)

* Lab 1 — Compute eigenvalues, eigenvectors using library
* Lab 2 — Visualize repeated application of linear map

3. Mini-Project Idea

* Title: Power Method Demo
* Goal: Implement power iteration to approximate dominant eigenvalue and eigenvector
* Dataset keywords: random matrices, covariance matrices
* Output keywords: eigenvalue estimates, convergence plots, error curve

4. Q&A Seeds (Questions Only)

* Q1: When is a matrix diagonalizable?
* Q2: Why are eigenvalues important for ML?
* Q3: How does power iteration work intuitively?
* Q4: What if eigenvalues are complex?
* Q5: Tradeoffs using full eigendecomposition vs approximations?

---

Day 5: Orthogonality and Projections

1. Theory Keywords

* Inner product spaces basics
* Orthogonal and orthonormal vectors
* Orthogonal projections onto subspace
* Projection matrix properties
* Decomposing vector into components
* Least squares as projection
* Gram–Schmidt process idea

2. Hands-on Labs (short specs)

* Lab 1 — Implement projection onto 1D, 2D subspaces
* Lab 2 — Verify orthogonality and reconstruction numerically

3. Mini-Project Idea

* Title: Least Squares Fitter via Projections
* Goal: Implement simple linear regression using projection viewpoint
* Dataset keywords: synthetic 2D regression data
* Output keywords: fitted line, residuals plot, projection illustration

4. Q&A Seeds (Questions Only)

* Q1: Why is orthogonality useful in ML?
* Q2: How do projections relate to least squares?
* Q3: How does Gram–Schmidt construct an orthonormal basis?
* Q4: What happens if basis vectors are nearly dependent?
* Q5: Tradeoffs using orthogonal vs non-orthogonal bases?

---

Day 6: Matrix Factorizations I (LU, QR)

1. Theory Keywords

* LU decomposition concept
* Pivoting and numerical stability
* Solving Ax = b via LU
* QR decomposition basics
* Orthogonal matrices properties
* Least squares via QR
* Complexity considerations

2. Hands-on Labs (short specs)

* Lab 1 — Use library LU and compare solving speeds
* Lab 2 — Solve least squares using QR decomposition

3. Mini-Project Idea

* Title: Linear Solver Benchmark
* Goal: Compare direct solving, LU-based solving, and QR-based least squares on synthetic problems
* Dataset keywords: random systems, tall matrices
* Output keywords: timing results, accuracy metrics, summary table

4. Q&A Seeds (Questions Only)

* Q1: Why is pivoting needed in LU?
* Q2: How does QR help least squares?
* Q3: When choose LU vs QR in practice?
* Q4: What if matrix is ill-conditioned?
* Q5: Tradeoffs between accuracy and speed here?

---

Day 7: Matrix Factorizations II (SVD, PCA View)

1. Theory Keywords

* Singular value decomposition definition
* Relation to eigen decomposition
* Low-rank approximation concept
* Energy captured by singular values
* PCA as SVD of centered data
* Dimensionality reduction intuition
* Reconstruction error vs rank

2. Hands-on Labs (short specs)

* Lab 1 — Compute SVD of data matrix
* Lab 2 — Compare original vs low-rank reconstruction

3. Mini-Project Idea

* Title: PCA Playground
* Goal: Implement PCA using SVD and visualize projections in 2D
* Dataset keywords: toy 2D, MNIST subset, tabular data
* Output keywords: explained variance plot, projection scatter, reconstruction examples

4. Q&A Seeds (Questions Only)

* Q1: Why is SVD more stable than eigen decomposition?
* Q2: How does SVD enable low-rank approximations?
* Q3: How does PCA reduce dimensionality?
* Q4: What if components are not clearly separated?
* Q5: Tradeoffs choosing number of principal components?

---

Day 8: Single-Variable Calculus for ML

1. Theory Keywords

* Limits and continuity basics
* Derivatives as rates of change
* Common derivative rules
* Chain rule applications
* Exponential, logarithm derivatives
* Higher-order derivatives
* Taylor approximation intuition

2. Hands-on Labs (short specs)

* Lab 1 — Implement numerical derivatives and compare with analytic
* Lab 2 — Visualize function, derivative, tangent lines

3. Mini-Project Idea

* Title: Derivative Visual Tool
* Goal: Build a small visualizer showing function and derivative curves
* Dataset keywords: synthetic functions, polynomials, exponentials
* Output keywords: plots, derivative comparison, approximation error

4. Q&A Seeds (Questions Only)

* Q1: Why is differentiability important for optimization?
* Q2: How does chain rule appear in deep networks?
* Q3: How accurate is numerical differentiation?
* Q4: What happens near discontinuities or kinks?
* Q5: Tradeoffs between analytic and numerical derivatives?

---

Day 9: Multivariable Calculus Basics

1. Theory Keywords

* Functions of several variables
* Partial derivatives definition
* Gradient vector interpretation
* Directional derivatives basics
* Level sets and contours
* Jacobian matrix concept
* Hessian matrix intuition

2. Hands-on Labs (short specs)

* Lab 1 — Compute gradients numerically in 2D, 3D
* Lab 2 — Visualize contour plots and gradient directions

3. Mini-Project Idea

* Title: 2D Gradient Field Explorer
* Goal: Build tool to plot gradients and descent paths on 2D surfaces
* Dataset keywords: synthetic 2D functions, quadratic bowls
* Output keywords: contour plots, gradient arrows, path trajectories

4. Q&A Seeds (Questions Only)

* Q1: How does gradient generalize derivative?
* Q2: Why do gradients point in steepest ascent?
* Q3: How does Jacobian relate to vector-valued functions?
* Q4: What if Hessian is indefinite?
* Q5: Tradeoffs using full Hessian vs approximations?

---

Day 10: Optimization Landscapes

1. Theory Keywords

* Local vs global minima
* Saddle points definition
* Convex vs nonconvex functions
* Stationary points conditions
* Strong convexity intuition
* Condition number and curvature
* Role of scaling and reparameterization

2. Hands-on Labs (short specs)

* Lab 1 — Visualize different 2D landscapes
* Lab 2 — Identify stationary points numerically

3. Mini-Project Idea

* Title: Landscape Catalog
* Goal: Create a gallery of optimization surfaces and annotate minima, maxima, saddles
* Dataset keywords: synthetic functions, polynomials, sinusoids
* Output keywords: plots, annotated points, classification table

4. Q&A Seeds (Questions Only)

* Q1: Why are saddle points problematic in deep learning?
* Q2: How does convexity simplify optimization?
* Q3: How does scaling of variables affect landscape?
* Q4: What if multiple local minima exist?
* Q5: Tradeoffs between expressive nonconvex models and optimization difficulty?

---

Day 11: Gradient Descent and Variants

1. Theory Keywords

* Gradient descent update rule
* Learning rate step size role
* Convergence conditions basic
* Stochastic gradient descent concept
* Mini-batch gradients
* Momentum intuition
* Learning rate schedules basics

2. Hands-on Labs (short specs)

* Lab 1 — Implement batch gradient descent on quadratic
* Lab 2 — Compare GD, SGD, momentum on toy function

3. Mini-Project Idea

* Title: Optimizer Playground
* Goal: Build framework to compare optimizers on various 2D functions
* Dataset keywords: synthetic loss functions, bowls, saddles
* Output keywords: trajectories, convergence plots, metrics table

4. Q&A Seeds (Questions Only)

* Q1: How does learning rate affect convergence?
* Q2: Why use mini-batches instead of full batch?
* Q3: How does momentum accelerate convergence?
* Q4: What happens with too large learning rate?
* Q5: Tradeoffs between SGD noise and convergence stability?

---

Day 12: Probability Basics

1. Theory Keywords

* Sample spaces and events
* Probability axioms
* Conditional probability definition
* Independence and dependence
* Bayes’ rule formula
* Law of total probability
* Discrete vs continuous variables

2. Hands-on Labs (short specs)

* Lab 1 — Simulate simple probability experiments
* Lab 2 — Estimate conditional probabilities from data

3. Mini-Project Idea

* Title: Bayesian Intuition Simulator
* Goal: Implement simulations to demonstrate Bayes’ rule in simple scenarios
* Dataset keywords: synthetic categorical data, coin flips
* Output keywords: prior/posterior tables, frequency plots, comparisons

4. Q&A Seeds (Questions Only)

* Q1: When are events independent?
* Q2: How does Bayes’ rule update beliefs?
* Q3: How to estimate probabilities from data?
* Q4: What if sample size is very small?
* Q5: Tradeoffs between frequentist and Bayesian viewpoints?

---

Day 13: Random Variables and Distributions

1. Theory Keywords

* Random variable definition
* Probability mass function
* Probability density function
* CDF and quantiles
* Expected value, variance, moments
* Common discrete distributions
* Common continuous distributions

2. Hands-on Labs (short specs)

* Lab 1 — Sample from standard distributions and visualize
* Lab 2 — Empirically estimate expectation and variance

3. Mini-Project Idea

* Title: Distribution Explorer
* Goal: Build small tool to visualize different distributions and sample statistics
* Dataset keywords: synthetic samples from known distributions
* Output keywords: histograms, empirical vs theoretical stats, comparison plots

4. Q&A Seeds (Questions Only)

* Q1: How do PMF and PDF differ?
* Q2: Why are moments useful descriptors?
* Q3: How to pick a distributional model?
* Q4: What if distribution is heavy-tailed?
* Q5: Tradeoffs modeling with simple vs complex distributions?

---

Day 14: Joint, Marginal, Conditional Distributions

1. Theory Keywords

* Joint distribution definition
* Marginalization process
* Conditional distributions basics
* Covariance and correlation
* Independence vs uncorrelated
* Multivariate normal intuition
* Conditional expectation concept

2. Hands-on Labs (short specs)

* Lab 1 — Simulate joint distributions and compute marginals
* Lab 2 — Estimate covariance, correlation from samples

3. Mini-Project Idea

* Title: Correlation Analyzer
* Goal: Build tool to compute and visualize correlation matrices for datasets
* Dataset keywords: tabular datasets, synthetic multivariate Gaussians
* Output keywords: correlation heatmaps, scatter plots, summary report

4. Q&A Seeds (Questions Only)

* Q1: How to derive a marginal from a joint?
* Q2: Difference between independence and zero correlation?
* Q3: Why is covariance scale-dependent?
* Q4: What if correlations are spurious?
* Q5: Tradeoffs using correlation vs more robust dependence measures?

---

Day 15: Expectation, Variance, LLN, CLT

1. Theory Keywords

* Expectation linearity property
* Variance of sum basics
* Law of large numbers
* Central limit theorem idea
* Sample mean distribution
* Confidence intervals intuition
* Monte Carlo estimation basics

2. Hands-on Labs (short specs)

* Lab 1 — Empirically verify law of large numbers
* Lab 2 — Show CLT using sums of random variables

3. Mini-Project Idea

* Title: Monte Carlo Estimator Demo
* Goal: Approximate integrals using Monte Carlo and analyze convergence
* Dataset keywords: synthetic distributions, unit interval
* Output keywords: estimate curves, confidence bands, error plots

4. Q&A Seeds (Questions Only)

* Q1: Why does LLN justify empirical averages?
* Q2: How does CLT support normal approximations?
* Q3: How many samples needed for stable estimates?
* Q4: What happens with heavy-tailed distributions in CLT?
* Q5: Tradeoffs using Monte Carlo vs analytic integration?

---

Day 16: Basic Statistical Inference

1. Theory Keywords

* Population vs sample concepts
* Point estimators properties
* Bias, variance of estimators
* Maximum likelihood estimation idea
* Likelihood vs probability distinction
* Confidence intervals basics
* Hypothesis testing intuition

2. Hands-on Labs (short specs)

* Lab 1 — Implement MLE for simple distributions
* Lab 2 — Construct confidence intervals for mean

3. Mini-Project Idea

* Title: Simple Inference Toolkit
* Goal: Build small library for basic estimation and confidence intervals
* Dataset keywords: synthetic normal data, Poisson counts
* Output keywords: estimates, interval reports, diagnostic plots

4. Q&A Seeds (Questions Only)

* Q1: What makes an estimator unbiased?
* Q2: How does MLE relate to fitting ML models?
* Q3: How to interpret a confidence interval correctly?
* Q4: What if model assumptions are violated?
* Q5: Tradeoffs between bias and variance in estimators?

---

Day 17: Information Theory Basics

1. Theory Keywords

* Entropy definition
* Joint and conditional entropy
* Mutual information concept
* KL divergence formula
* Cross-entropy connection
* Bits, coding interpretation
* Relation to likelihood, losses

2. Hands-on Labs (short specs)

* Lab 1 — Compute entropies and mutual information from discrete distributions
* Lab 2 — Compare KL divergence and cross-entropy numerically

3. Mini-Project Idea

* Title: Information Metrics Explorer
* Goal: Build utility to compute information-theoretic quantities on discrete datasets
* Dataset keywords: categorical features, joint tables, text tokens
* Output keywords: entropy tables, MI scores, ranking plots

4. Q&A Seeds (Questions Only)

* Q1: How does entropy measure uncertainty?
* Q2: Why is KL divergence asymmetric?
* Q3: How does cross-entropy relate to classification loss?
* Q4: What if probability estimates are zero?
* Q5: Tradeoffs between MI-based feature selection and correlation-based?

---

Day 18: Numerical Stability and Floating Point

1. Theory Keywords

* Floating point representation basics
* Machine epsilon idea
* Overflow and underflow
* Catastrophic cancellation
* Stable vs unstable algorithms
* Log-sum-exp trick
* Scaling for numerical stability

2. Hands-on Labs (short specs)

* Lab 1 — Experiment with floating point rounding errors
* Lab 2 — Implement stable vs naive softmax

3. Mini-Project Idea

* Title: Stability Diagnostic Suite
* Goal: Create tests demonstrating unstable vs stable implementations for common operations
* Dataset keywords: extreme values, small differences, logits
* Output keywords: numerical error reports, comparison plots, recommendations

4. Q&A Seeds (Questions Only)

* Q1: Why is numerical stability crucial for deep learning?
* Q2: How does log-sum-exp prevent underflow?
* Q3: How to design stable algorithms for sums?
* Q4: What happens when gradients explode numerically?
* Q5: Tradeoffs between stability and performance?

---

Day 19: Convex Optimization Basics

1. Theory Keywords

* Convex sets, convex functions
* First-order characterization
* Subgradients concept
* Projected gradient descent idea
* Constraints and feasible region
* Duality high-level intuition
* Strong convexity and convergence

2. Hands-on Labs (short specs)

* Lab 1 — Check convexity numerically in 1D, 2D
* Lab 2 — Implement projected gradient descent on simple constrained problem

3. Mini-Project Idea

* Title: Convex Optimization Sandbox
* Goal: Solve small convex problems and visualize feasible sets and optimal points
* Dataset keywords: synthetic quadratic programs, simplex constraints
* Output keywords: solution points, constraint plots, convergence curves

4. Q&A Seeds (Questions Only)

* Q1: Why is convexity attractive for optimization?
* Q2: How do subgradients enable nonsmooth optimization?
* Q3: How does projection maintain feasibility?
* Q4: What if problem is not convex?
* Q5: Tradeoffs using convex approximations for nonconvex problems?

---

Day 20: Math Integration Day

1. Theory Keywords

* Linear algebra recap
* Calculus recap
* Probability recap
* Optimization recap
* Information theory recap
* Numerical stability recap
* Connections to ML objectives

2. Hands-on Labs (short specs)

* Lab 1 — Re-implement logistic regression loss and gradients from math
* Lab 2 — Explore conditioning and optimization on small regression problem

3. Mini-Project Idea

* Title: From Math to Logistic Regression
* Goal: Derive and implement logistic regression training using mathematical tools only
* Dataset keywords: synthetic classification data, 2D blobs
* Output keywords: trained model, decision boundary plots, loss curves

4. Q&A Seeds (Questions Only)

* Q1: How do gradients, probabilities, and linear algebra combine in ML?
* Q2: Which math assumptions are most fragile in practice?
* Q3: How does conditioning affect optimization and generalization?
* Q4: What if numerical issues dominate theoretical behavior?
* Q5: Tradeoffs spending time on theory vs implementation?

---

Day 21: Data Handling and Exploration

1. Theory Keywords

* Tabular data structure basics
* Feature types: numeric, categorical
* Missing values patterns
* Train/validation/test splits
* Data leakage concept
* Summary statistics and profiling
* Basic visualization practices

2. Hands-on Labs (short specs)

* Lab 1 — Load, clean, and profile a tabular dataset
* Lab 2 — Implement train/validation/test splitting with stratification

3. Mini-Project Idea

* Title: Data Audit Report
* Goal: Build a script that generates an initial exploratory report for any tabular dataset
* Dataset keywords: open tabular data, Kaggle datasets
* Output keywords: summary tables, plots, split files, textual report

4. Q&A Seeds (Questions Only)

* Q1: Why is data leakage dangerous?
* Q2: How to choose appropriate split ratios?
* Q3: How to handle mixed feature types systematically?
* Q4: What if classes are severely imbalanced at splitting?
* Q5: Tradeoffs between simple random split and more complex schemes?

---

Day 22: Linear Regression Fundamentals

1. Theory Keywords

* Linear model formulation
* Ordinary least squares objective
* Closed-form OLS solution
* Assumptions behind linear regression
* Bias term and intercept
* Residuals and error analysis
* Overfitting vs underfitting basics

2. Hands-on Labs (short specs)

* Lab 1 — Implement linear regression with closed-form solution
* Lab 2 — Plot residuals and analyze patterns

3. Mini-Project Idea

* Title: House Price Baseline Regressor
* Goal: Train simple linear regression on house price dataset and analyze residuals
* Dataset keywords: housing features, tabular data
* Output keywords: fitted model, error metrics, diagnostic plots

4. Q&A Seeds (Questions Only)

* Q1: When is linear regression appropriate?
* Q2: How does the bias term affect fit?
* Q3: How to interpret regression coefficients?
* Q4: What if residuals show strong patterns?
* Q5: Tradeoffs between simplicity and accuracy for linear models?

---

Day 23: Regularized Linear Models (Ridge, Lasso)

1. Theory Keywords

* L2 regularization (ridge)
* L1 regularization (lasso)
* Bias-variance tradeoff with regularization
* Shrinkage of coefficients
* Feature selection via lasso
* Hyperparameter lambda role
* Standardization before regularization

2. Hands-on Labs (short specs)

* Lab 1 — Implement ridge and lasso using libraries
* Lab 2 — Plot coefficient paths vs regularization strength

3. Mini-Project Idea

* Title: Regularization Tuning Study
* Goal: Evaluate ridge and lasso on real dataset across regularization grid
* Dataset keywords: housing, medical, or finance tabular data
* Output keywords: validation curves, coefficient plots, comparison table

4. Q&A Seeds (Questions Only)

* Q1: How does regularization reduce overfitting?
* Q2: Why should features often be standardized first?
* Q3: How to choose lambda in practice?
* Q4: What happens with very strong regularization?
* Q5: Tradeoffs between ridge and lasso in sparse settings?

---

Day 24: Logistic Regression and Classification Basics

1. Theory Keywords

* Binary classification problem setup
* Logistic function and odds
* Cross-entropy loss
* Decision boundary interpretation
* Probability calibration
* Thresholding for class decision
* Evaluation metrics overview

2. Hands-on Labs (short specs)

* Lab 1 — Train logistic regression on binary dataset
* Lab 2 — Plot decision boundary and probability contours

3. Mini-Project Idea

* Title: Binary Classifier for Customer Churn
* Goal: Build logistic regression model to predict churn with basic evaluation
* Dataset keywords: customer features, churn labels
* Output keywords: metrics report, ROC curve, confusion matrix

4. Q&A Seeds (Questions Only)

* Q1: Why is logistic loss preferred over squared loss for classification?
* Q2: How to interpret logistic regression coefficients?
* Q3: How does threshold choice affect precision and recall?
* Q4: What if predicted probabilities are poorly calibrated?
* Q5: Tradeoffs between interpretability and performance vs more complex models?

---

Day 25: Model Evaluation and Metrics I

1. Theory Keywords

* Train vs validation vs test roles
* Accuracy limitations
* Precision, recall, F1-score
* ROC curve, AUC
* PR curve for imbalanced data
* Confusion matrix analysis
* Metric selection based on business goals

2. Hands-on Labs (short specs)

* Lab 1 — Compute multiple metrics for binary classifier
* Lab 2 — Plot ROC and PR curves for different models

3. Mini-Project Idea

* Title: Metric Comparison Dashboard
* Goal: Build script to generate metric summary and plots for classification models
* Dataset keywords: any binary classification dataset
* Output keywords: metric tables, ROC/PR plots, comparison report

4. Q&A Seeds (Questions Only)

* Q1: When is accuracy misleading?
* Q2: How do precision and recall trade off?
* Q3: When is PR curve more informative than ROC?
* Q4: What if business costs are highly asymmetric?
* Q5: Tradeoffs choosing a single primary evaluation metric?

---

Day 26: Data Preprocessing and Feature Scaling

1. Theory Keywords

* Standardization vs normalization
* Robust scaling for outliers
* Categorical encoding basics
* One-hot vs ordinal encoding
* Train-set-only fit for preprocessing
* Pipeline concept high-level
* Leakage through preprocessing pitfalls

2. Hands-on Labs (short specs)

* Lab 1 — Implement scaling and encoding using library transformers
* Lab 2 — Compare model performance with and without proper scaling

3. Mini-Project Idea

* Title: Preprocessing Pipeline Builder
* Goal: Build reusable preprocessing pipeline for mixed tabular data
* Dataset keywords: real-world tabular datasets
* Output keywords: pipeline object, transformed features, evaluation report

4. Q&A Seeds (Questions Only)

* Q1: Why must preprocessing be fitted only on train data?
* Q2: How does scaling impact gradient-based models?
* Q3: How to handle high-cardinality categoricals?
* Q4: What if test categories differ from train categories?
* Q5: Tradeoffs between simpler preprocessing and complex feature engineering?

---

Day 27: Overfitting, Underfitting, Capacity

1. Theory Keywords

* Model capacity and complexity
* Bias-variance decomposition intuition
* Learning curves concept
* High-variance vs high-bias models
* Regularization as capacity control
* Early stopping as regularization
* Data augmentation idea

2. Hands-on Labs (short specs)

* Lab 1 — Plot learning curves for underfit vs overfit models
* Lab 2 — Study effect of regularization strength on validation performance

3. Mini-Project Idea

* Title: Bias-Variance Exploration Study
* Goal: Compare models of varying complexity on synthetic data with known ground truth
* Dataset keywords: synthetic regression, synthetic classification
* Output keywords: learning curves, bias-variance analysis, summary report

4. Q&A Seeds (Questions Only)

* Q1: How to detect overfitting empirically?
* Q2: How does more data affect bias and variance?
* Q3: How to tune model complexity systematically?
* Q4: What if validation performance fluctuates greatly?
* Q5: Tradeoffs between high-capacity models and interpretability?

---

Day 28: Cross-Validation and Model Selection

1. Theory Keywords

* K-fold cross-validation
* Stratified cross-validation for classification
* Nested cross-validation concept
* Hyperparameter tuning basics
* Grid search vs random search
* Validation set reuse risks
* Early stopping with validation splits

2. Hands-on Labs (short specs)

* Lab 1 — Implement K-fold cross-validation on regression model
* Lab 2 — Perform simple grid search on classification model

3. Mini-Project Idea

* Title: Cross-Validation Tuner
* Goal: Build utility to run cross-validation and hyperparameter search experiments
* Dataset keywords: tabular regression, tabular classification datasets
* Output keywords: CV scores, best hyperparameters, result tables

4. Q&A Seeds (Questions Only)

* Q1: Why does cross-validation reduce variance of estimates?
* Q2: When is stratification essential?
* Q3: How does random search compare to grid search?
* Q4: What if validation performance is noisy across folds?
* Q5: Tradeoffs between computation time and thorough hyperparameter search?

---

Day 29: k-NN and Distance-Based Methods

1. Theory Keywords

* k-nearest neighbors algorithm
* Distance metrics for features
* Curse of dimensionality
* Scaling effects on distances
* k-NN for classification vs regression
* Complexity and indexing structures
* Local vs global generalization

2. Hands-on Labs (short specs)

* Lab 1 — Implement k-NN classifier with different k values
* Lab 2 — Compare distance metrics on same dataset

3. Mini-Project Idea

* Title: k-NN Decision Boundary Explorer
* Goal: Visualize decision boundaries of k-NN on 2D synthetic datasets
* Dataset keywords: toy 2D blobs, moons, circles
* Output keywords: boundary plots, error rates, k sweeps

4. Q&A Seeds (Questions Only)

* Q1: Why is k-NN sensitive to feature scaling?
* Q2: How does k affect bias and variance?
* Q3: How to choose appropriate distance metric?
* Q4: What if data is very high-dimensional?
* Q5: Tradeoffs using k-NN vs parametric models?

---

Day 30: Decision Trees Fundamentals

1. Theory Keywords

* Decision tree structure basics
* Splitting criteria: Gini, entropy
* Information gain concept
* Tree depth and overfitting
* Handling numeric vs categorical features
* Pruning strategies overview
* Interpretability of trees

2. Hands-on Labs (short specs)

* Lab 1 — Train decision tree classifier and visualize structure
* Lab 2 — Study effect of depth and min samples split

3. Mini-Project Idea

* Title: Explainable Tree Classifier
* Goal: Build decision tree model and generate human-readable rules
* Dataset keywords: tabular classification datasets
* Output keywords: tree visualization, rule list, performance metrics

4. Q&A Seeds (Questions Only)

* Q1: How do trees choose split thresholds?
* Q2: Why do deep trees tend to overfit?
* Q3: How to interpret feature importance from trees?
* Q4: What if splits are unstable across samples?
* Q5: Tradeoffs between tree depth, accuracy, and interpretability?

---

Day 31: Ensemble Methods I (Bagging, Random Forests)

1. Theory Keywords

* Bagging concept
* Bootstrap sampling
* Variance reduction via averaging
* Random forests structure
* Feature sub-sampling in splits
* Out-of-bag error estimate
* Feature importance in forests

2. Hands-on Labs (short specs)

* Lab 1 — Train random forest and compare with single tree
* Lab 2 — Evaluate out-of-bag error vs validation error

3. Mini-Project Idea

* Title: Random Forest Baseline for Tabular Data
* Goal: Use random forests as strong baseline on several datasets
* Dataset keywords: multiple tabular datasets, classification and regression
* Output keywords: metrics report, feature importances, comparison plots

4. Q&A Seeds (Questions Only)

* Q1: How does bagging reduce variance?
* Q2: Why random feature selection in forests?
* Q3: How reliable are feature importance measures?
* Q4: What if base trees are too shallow or too deep?
* Q5: Tradeoffs between ensemble size and inference cost?

---

Day 32: Ensemble Methods II (Boosting, Gradient Boosting)

1. Theory Keywords

* Boosting concept
* Weak learners and additive models
* Gradient boosting framework
* Learning rate in boosting
* Overfitting in boosting
* XGBoost, LightGBM high-level idea
* Handling different loss functions

2. Hands-on Labs (short specs)

* Lab 1 — Train gradient boosted trees and tune depth, learning rate
* Lab 2 — Compare boosting vs random forest on same dataset

3. Mini-Project Idea

* Title: Boosting Benchmark
* Goal: Evaluate gradient boosting on multiple datasets against simpler baselines
* Dataset keywords: varied tabular benchmarks
* Output keywords: performance comparisons, tuning curves, summary report

4. Q&A Seeds (Questions Only)

* Q1: How does boosting differ from bagging?
* Q2: Why do small learning rates often work better?
* Q3: How to control overfitting in boosting?
* Q4: What if dataset is noisy with mislabeled points?
* Q5: Tradeoffs between interpretability and power for boosted trees?

---

Day 33: SVMs and Margins

1. Theory Keywords

* Linear SVM formulation
* Margin and support vectors
* Hinge loss concept
* Soft margin with C parameter
* Kernel trick basics
* RBF and polynomial kernels
* SVMs for classification vs regression

2. Hands-on Labs (short specs)

* Lab 1 — Train linear and kernel SVMs on 2D datasets
* Lab 2 — Visualize margins and support vectors

3. Mini-Project Idea

* Title: SVM Decision Boundary Gallery
* Goal: Compare linear and kernel SVM boundaries on multiple synthetic problems
* Dataset keywords: moons, circles, linearly separable data
* Output keywords: boundary plots, margin illustrations, metrics table

4. Q&A Seeds (Questions Only)

* Q1: How does margin relate to generalization?
* Q2: How does C control tradeoff in SVM?
* Q3: When are kernels particularly powerful?
* Q4: What if feature dimension is huge vs sample size?
* Q5: Tradeoffs between SVMs and tree ensembles on tabular data?

---

Day 34: Clustering Basics (k-Means, Hierarchical)

1. Theory Keywords

* Clustering vs classification
* k-means objective function
* Initialization and local minima
* Choosing k (elbow, silhouette)
* Hierarchical clustering basics
* Linkage criteria overview
* Cluster evaluation metrics

2. Hands-on Labs (short specs)

* Lab 1 — Implement k-means and visualize clusters
* Lab 2 — Apply hierarchical clustering and plot dendrogram

3. Mini-Project Idea

* Title: Customer Segmentation Prototype
* Goal: Cluster customers based on behavior and visualize segment characteristics
* Dataset keywords: transactional data, customer features
* Output keywords: cluster assignments, segment descriptions, plots

4. Q&A Seeds (Questions Only)

* Q1: How does k-means optimize its objective?
* Q2: How sensitive is k-means to initialization?
* Q3: How to choose number of clusters reasonably?
* Q4: What if clusters are non-spherical or imbalanced?
* Q5: Tradeoffs between k-means and hierarchical methods?

---

Day 35: Dimensionality Reduction (PCA, t-SNE, UMAP)

1. Theory Keywords

* Linear vs nonlinear dimensionality reduction
* PCA recap for visualization
* t-SNE intuition
* UMAP high-level idea
* Perplexity and local structure
* Global vs local structure preservation
* Pitfalls in interpreting embeddings

2. Hands-on Labs (short specs)

* Lab 1 — Apply PCA, t-SNE, UMAP to same dataset
* Lab 2 — Visualize clusters in embedding space

3. Mini-Project Idea

* Title: Embedding Explorer
* Goal: Build tool to compare dimensionality reduction methods interactively
* Dataset keywords: MNIST, textual embeddings, tabular data
* Output keywords: 2D embeddings plots, clustering overlays, comparison notes

4. Q&A Seeds (Questions Only)

* Q1: When is PCA sufficient?
* Q2: How does t-SNE distort global distances?
* Q3: How to choose perplexity or neighbors parameters?
* Q4: What if embeddings look noisy and tangled?
* Q5: Tradeoffs between interpretability and structure preservation across methods?

---

Day 36: Handling Missing Data and Outliers

1. Theory Keywords

* Types of missingness (MCAR, MAR, MNAR)
* Simple imputation strategies
* Advanced imputation high-level overview
* Outlier detection basics
* Robust statistics (median, IQR)
* Impact of missingness on bias
* Impact of outliers on models

2. Hands-on Labs (short specs)

* Lab 1 — Implement different imputation strategies and compare impact
* Lab 2 — Detect and analyze outliers in a dataset

3. Mini-Project Idea

* Title: Robust Data Cleaning Toolkit
* Goal: Build scripts for imputation and outlier handling with comparisons
* Dataset keywords: noisy tabular data, real-world datasets
* Output keywords: cleaned datasets, impact metrics, documentation

4. Q&A Seeds (Questions Only)

* Q1: How can missingness mechanism bias results?
* Q2: When is simple mean imputation unacceptable?
* Q3: How to detect influential outliers?
* Q4: What if removal of outliers loses important patterns?
* Q5: Tradeoffs between robust methods and preserving raw data structure?

---

Day 37: Imbalanced Classification

1. Theory Keywords

* Class imbalance definition
* Resampling techniques (oversampling, undersampling)
* Synthetic methods (SMOTE idea)
* Class-weighted losses
* Appropriate metrics for imbalance
* Threshold moving strategies
* Cost-sensitive learning basics

2. Hands-on Labs (short specs)

* Lab 1 — Train classifier on imbalanced data with different strategies
* Lab 2 — Compare metrics and confusion matrices for different approaches

3. Mini-Project Idea

* Title: Fraud Detection Prototype
* Goal: Build imbalanced classification pipeline with appropriate metrics and techniques
* Dataset keywords: credit card fraud, rare events
* Output keywords: metric reports, PR curves, method comparison

4. Q&A Seeds (Questions Only)

* Q1: Why is accuracy especially misleading for imbalanced problems?
* Q2: How do class weights influence training dynamics?
* Q3: When to prefer resampling over reweighting?
* Q4: What if oversampling leads to overfitting minority class?
* Q5: Tradeoffs between recall and precision for rare events?

---

Day 38: ML Pipelines and Reproducibility

1. Theory Keywords

* End-to-end pipeline concept
* Separation of preprocessing and modeling
* Random seeds and determinism
* Configuration management basics
* Data versioning high-level idea
* Reproducible evaluation setup
* Logging and experiment metadata

2. Hands-on Labs (short specs)

* Lab 1 — Build end-to-end scikit-learn pipeline for classification
* Lab 2 — Implement basic experiment logging with configs and seeds

3. Mini-Project Idea

* Title: Reproducible Baseline Project
* Goal: Create small repository with fully reproducible ML experiment
* Dataset keywords: any stable benchmark dataset
* Output keywords: pipeline code, config files, logs, README instructions

4. Q&A Seeds (Questions Only)

* Q1: Why are pipelines essential for real projects?
* Q2: How can randomness break reproducibility?
* Q3: What minimal metadata should experiments log?
* Q4: What if data changes between training and deployment?
* Q5: Tradeoffs between quick experiments and well-structured pipelines?

---

Day 39: Model Debugging and Error Analysis

1. Theory Keywords

* Systematic error analysis process
* Data slices and stratification
* Calibration and reliability diagrams
* Shortcut learning and spurious correlations
* Concept drift high-level idea
* Diagnostic plots for residuals
* Human-in-the-loop feedback

2. Hands-on Labs (short specs)

* Lab 1 — Perform structured error analysis on trained classifier
* Lab 2 — Build calibration plots and adjust thresholds

3. Mini-Project Idea

* Title: Error Analysis Notebook
* Goal: Implement notebook template for thorough error breakdown
* Dataset keywords: any ML dataset with labels
* Output keywords: slice metrics, confusion tables, qualitative examples

4. Q&A Seeds (Questions Only)

* Q1: How to prioritize which errors to fix?
* Q2: How does slicing by features reveal biases?
* Q3: How to detect calibration issues empirically?
* Q4: What if model relies on unintended shortcuts?
* Q5: Tradeoffs between fixing rare vs frequent error types?

---

Day 40: Classical ML Integration Day

1. Theory Keywords

* Supervised learning recap
* Regression vs classification recap
* Regularization, capacity, overfitting
* Evaluation metrics recap
* Pipelines and reproducibility
* Error analysis, debugging
* Preparing for deep learning

2. Hands-on Labs (short specs)

* Lab 1 — Build full tabular ML pipeline from raw data to evaluation
* Lab 2 — Compare several model families under same pipeline

3. Mini-Project Idea

* Title: End-to-End Tabular ML Project
* Goal: Deliver a complete classical ML solution on chosen dataset
* Dataset keywords: Kaggle tabular dataset, open benchmark
* Output keywords: preprocessing code, models, metrics, reports, plots

4. Q&A Seeds (Questions Only)

* Q1: Which model families worked best and why?
* Q2: Which preprocessing choices had biggest impact?
* Q3: How to transition this pipeline toward deep learning usage?
* Q4: What if classical models outperform deep models on tabular data?
* Q5: Tradeoffs between model complexity and pipeline maintainability?

---

Day 41: PyTorch Basics and Tensors

1. Theory Keywords

* PyTorch tensor abstraction
* Tensor shapes and dtypes
* CPU vs GPU tensors
* Broadcasting rules
* Autograd high-level overview
* Computational graph concept
* In-place operations warnings

2. Hands-on Labs (short specs)

* Lab 1 — Create and manipulate tensors, check shapes
* Lab 2 — Implement basic linear algebra operations with tensors

3. Mini-Project Idea

* Title: Tensor Utilities Library
* Goal: Build helper functions for common tensor operations and checks
* Dataset keywords: synthetic tensors, random data
* Output keywords: utility module, tests, small demos

4. Q&A Seeds (Questions Only)

* Q1: How do tensors differ from NumPy arrays?
* Q2: Why be careful with in-place operations?
* Q3: How does device (CPU/GPU) affect performance?
* Q4: What if tensor shapes are misaligned?
* Q5: Tradeoffs between convenience abstractions and tensor-level control?

---

Day 42: Autograd and Manual Gradients

1. Theory Keywords

* Autograd mechanism basics
* requires_grad flag usage
* backward() and gradient accumulation
* Computational graph lifetimes
* Detaching tensors
* Comparing autograd vs manual gradients
* Gradient checking idea

2. Hands-on Labs (short specs)

* Lab 1 — Implement simple function and verify gradients via autograd
* Lab 2 — Implement numerical gradient checker for small functions

3. Mini-Project Idea

* Title: Gradient Checker Toolkit
* Goal: Build small module to verify gradients of custom PyTorch operations
* Dataset keywords: synthetic inputs, random tensors
* Output keywords: gradient comparison logs, error metrics, examples

4. Q&A Seeds (Questions Only)

* Q1: How does PyTorch autograd build computation graphs?
* Q2: Why do gradients accumulate by default?
* Q3: When should you detach tensors from graph?
* Q4: What if numerical gradient and autograd disagree?
* Q5: Tradeoffs writing custom autograd vs using built-in modules?

---

Day 43: Building MLPs from Scratch

1. Theory Keywords

* Fully connected layer formulation
* Activation functions (ReLU, sigmoid, tanh)
* Network depth and width
* Forward pass structure
* Loss functions for regression, classification
* Parameter initialization basics
* Overparameterization intuition

2. Hands-on Labs (short specs)

* Lab 1 — Implement simple MLP using only tensors, no nn.Module
* Lab 2 — Train MLP on small classification dataset

3. Mini-Project Idea

* Title: Pure PyTorch MLP Classifier
* Goal: Build and train MLP without high-level abstractions
* Dataset keywords: toy 2D datasets, MNIST subset
* Output keywords: training loop, loss curves, decision boundaries

4. Q&A Seeds (Questions Only)

* Q1: How does depth affect expressive power of MLPs?
* Q2: Why are nonlinear activations necessary?
* Q3: How does initialization impact training dynamics?
* Q4: What if network is too wide for small data?
* Q5: Tradeoffs building networks from scratch vs using nn.Sequential?

---

Day 44: PyTorch Modules and Training Loops

1. Theory Keywords

* nn.Module abstraction
* forward method definition
* Parameters and named_parameters
* Optimizers API basics
* Training vs evaluation modes
* mini-batch iteration with DataLoader
* Saving and loading model state_dict

2. Hands-on Labs (short specs)

* Lab 1 — Rebuild MLP using nn.Module and DataLoader
* Lab 2 — Implement standard training and evaluation loop template

3. Mini-Project Idea

* Title: Training Loop Template Library
* Goal: Create reusable, configurable PyTorch training loop utilities
* Dataset keywords: MNIST, CIFAR subset, synthetic data
* Output keywords: reusable code, example scripts, logs

4. Q&A Seeds (Questions Only)

* Q1: Why use nn.Module instead of raw functions?
* Q2: How to structure clean training and evaluation loops?
* Q3: How does DataLoader enable batching and shuffling?
* Q4: What if checkpoint loading fails or mismatches shapes?
* Q5: Tradeoffs between flexible training code and simplicity?

---

Day 45: Optimization in Deep Learning

1. Theory Keywords

* SGD vs momentum vs Adam
* Learning rate schedules (step, cosine)
* Weight decay as regularization
* Gradient clipping basics
* Batch size effects
* Loss surfaces for deep nets
* Warmup strategies idea

2. Hands-on Labs (short specs)

* Lab 1 — Compare optimizers on same network and dataset
* Lab 2 — Experiment with different learning rate schedules

3. Mini-Project Idea

* Title: Optimizer Benchmark Suite
* Goal: Systematically compare optimizers and schedules across multiple tasks
* Dataset keywords: simple vision, tabular, synthetic tasks
* Output keywords: training curves, final metrics, comparative plots

4. Q&A Seeds (Questions Only)

* Q1: Why might Adam converge faster than SGD?
* Q2: How does batch size interact with learning rate?
* Q3: How to diagnose need for gradient clipping?
* Q4: What if training loss plateaus early?
* Q5: Tradeoffs between Adam-style and pure SGD optimization?

---

Day 46: Regularization in Deep Networks

1. Theory Keywords

* L2 weight decay recap
* Dropout mechanism and intuition
* Batch normalization basics
* Data augmentation as regularization
* Early stopping in deep learning
* Label smoothing high-level idea
* Implicit regularization by SGD

2. Hands-on Labs (short specs)

* Lab 1 — Add dropout and batch norm to MLP and compare performance
* Lab 2 — Study impact of data augmentation on training and test metrics

3. Mini-Project Idea

* Title: Regularization Cookbook Experiment
* Goal: Evaluate combination of regularization techniques on small image dataset
* Dataset keywords: CIFAR-10 subset, Fashion-MNIST
* Output keywords: experiments matrix, metrics, recommendations

4. Q&A Seeds (Questions Only)

* Q1: How does dropout reduce co-adaptation?
* Q2: Why can batch norm accelerate training?
* Q3: How does augmentation help generalization?
* Q4: What if regularization is too strong?
* Q5: Tradeoffs between explicit and implicit regularization methods?

---

Day 47: Convolutional Neural Networks Basics

1. Theory Keywords

* Convolution operation on images
* Filters, kernels, strides, padding
* Feature maps and channels
* Pooling (max, average)
* Receptive field concept
* CNN vs MLP parameter sharing
* Typical CNN architectures overview

2. Hands-on Labs (short specs)

* Lab 1 — Implement 2D convolution manually for small images
* Lab 2 — Build simple CNN for MNIST-like digits

3. Mini-Project Idea

* Title: First CNN Image Classifier
* Goal: Train small CNN on simple image dataset and compare with MLP
* Dataset keywords: MNIST, Fashion-MNIST, CIFAR subset
* Output keywords: training curves, accuracy, feature map visualizations

4. Q&A Seeds (Questions Only)

* Q1: Why do convolutions reduce parameter count?
* Q2: How do strides and padding affect output size?
* Q3: Why is pooling commonly used?
* Q4: What if receptive field is too small?
* Q5: Tradeoffs between CNN depth and computation cost?

---

Day 48: Deeper CNNs and Modern Blocks

1. Theory Keywords

* Stacking convolutional layers
* Residual connections overview
* Batch norm in CNNs
* Downsampling strategies
* Bottleneck blocks intuition
* Common architectures (ResNet, VGG)
* Vanishing gradients in deep CNNs

2. Hands-on Labs (short specs)

* Lab 1 — Implement small residual CNN block
* Lab 2 — Compare plain CNN vs residual CNN on same task

3. Mini-Project Idea

* Title: Mini-ResNet Implementation
* Goal: Build simplified ResNet-like architecture and train on CIFAR subset
* Dataset keywords: CIFAR-10 subset, small image dataset
* Output keywords: architecture code, training results, comparison plots

4. Q&A Seeds (Questions Only)

* Q1: How do residual connections help training?
* Q2: How does depth affect representation power in CNNs?
* Q3: When to use bottleneck blocks?
* Q4: What if memory usage becomes too high?
* Q5: Tradeoffs between model size and accuracy for CNNs?

---

Day 49: Sequence Modeling and RNNs

1. Theory Keywords

* Sequential data characteristics
* Recurrent neural networks structure
* Hidden state updates
* Backpropagation through time
* Exploding and vanishing gradients
* Truncated BPTT idea
* RNN limitations

2. Hands-on Labs (short specs)

* Lab 1 — Implement simple RNN manually in PyTorch
* Lab 2 — Train RNN on toy sequence prediction task

3. Mini-Project Idea

* Title: Character-Level Sequence Predictor
* Goal: Train RNN to predict next character in simple text corpus
* Dataset keywords: small text corpus, character sequences
* Output keywords: training curves, sample generations, loss plots

4. Q&A Seeds (Questions Only)

* Q1: How do RNNs process sequences step by step?
* Q2: Why do gradients vanish or explode in RNNs?
* Q3: How does truncated BPTT trade accuracy vs computation?
* Q4: What if sequences have long-range dependencies?
* Q5: Tradeoffs between RNN simplicity and modeling capacity?

---

Day 50: LSTMs, GRUs, and Better RNNs

1. Theory Keywords

* LSTM cell structure
* Forget, input, output gates
* GRU high-level structure
* Handling long-term dependencies
* Parameter efficiency considerations
* Sequence-to-sequence modeling basics
* Bidirectional RNNs overview

2. Hands-on Labs (short specs)

* Lab 1 — Use built-in LSTM and GRU layers for sequence classification
* Lab 2 — Compare LSTM vs GRU performance on same task

3. Mini-Project Idea

* Title: Sequence Sentiment Classifier
* Goal: Build LSTM-based sentiment classifier for short texts
* Dataset keywords: IMDB subset, tweets, short reviews
* Output keywords: accuracy metrics, confusion matrix, training logs

4. Q&A Seeds (Questions Only)

* Q1: How do LSTMs mitigate vanishing gradients?
* Q2: When choose GRU over LSTM?
* Q3: How do bidirectional RNNs improve context usage?
* Q4: What if sequence length varies widely?
* Q5: Tradeoffs between recurrent depth and training time?

---

Day 51: Introduction to Attention

1. Theory Keywords

* Limitations of pure recurrence
* Attention mechanism intuition
* Query, key, value concepts
* Attention weights and softmax
* Context vector computation
* Alignment in sequence tasks
* Self-attention high-level overview

2. Hands-on Labs (short specs)

* Lab 1 — Implement simple additive attention for sequence-to-one task
* Lab 2 — Visualize attention weights for sample sequences

3. Mini-Project Idea

* Title: Attention-Based Sequence Classifier
* Goal: Build RNN with attention for text classification and inspect weights
* Dataset keywords: short text classification dataset
* Output keywords: attention maps, metrics, qualitative examples

4. Q&A Seeds (Questions Only)

* Q1: Why is attention useful for long sequences?
* Q2: How are attention weights interpreted?
* Q3: How does attention compare to simple pooling?
* Q4: What if attention focuses on irrelevant positions?
* Q5: Tradeoffs between attention complexity and performance gains?

---

Day 52: Transformers Basics

1. Theory Keywords

* Self-attention mechanism
* Multi-head attention concept
* Positional encodings
* Transformer encoder block structure
* Layer normalization usage
* Residual connections in transformers
* Scaling limitations and complexity

2. Hands-on Labs (short specs)

* Lab 1 — Implement small transformer encoder block in PyTorch
* Lab 2 — Apply transformer encoder to toy sequence classification task

3. Mini-Project Idea

* Title: Mini-Transformer for Toy Data
* Goal: Build tiny transformer model for synthetic sequence classification
* Dataset keywords: synthetic pattern sequences, simple tokens
* Output keywords: training curves, accuracy metrics, attention visualizations

4. Q&A Seeds (Questions Only)

* Q1: How does self-attention compute contextual representations?
* Q2: Why multi-head attention instead of single head?
* Q3: How do positional encodings inject order information?
* Q4: What if sequence length is very large for transformer?
* Q5: Tradeoffs between transformers and recurrent architectures?

---

Day 53: Training and Debugging Deep Models

1. Theory Keywords

* Loss explosion and NaNs
* Gradient norms monitoring
* Overfitting detection in deep networks
* Learning rate warmup and decay
* Debugging shape mismatches
* Unit testing model components
* Using hooks and summaries

2. Hands-on Labs (short specs)

* Lab 1 — Intentionally create training issues and debug them
* Lab 2 — Implement gradient norm logging and visualization

3. Mini-Project Idea

* Title: Training Debugger Utilities
* Goal: Develop reusable utilities to monitor and debug deep learning training runs
* Dataset keywords: small image or text datasets
* Output keywords: logs, diagnostic plots, debugging checklist

4. Q&A Seeds (Questions Only)

* Q1: How to quickly detect exploding gradients?
* Q2: How to distinguish data bugs from model bugs?
* Q3: What minimal metrics and signals to log during training?
* Q4: What if validation performance diverges from training behavior?
* Q5: Tradeoffs between extensive logging and training overhead?

---

Day 54: DataLoaders, Augmentations, and Efficient Input Pipelines

1. Theory Keywords

* Custom Dataset and DataLoader
* Batch size and throughput
* Shuffling and sampling strategies
* Data augmentation for images
* On-the-fly vs offline augmentation
* Worker processes and I/O bottlenecks
* Reproducibility with workers

2. Hands-on Labs (short specs)

* Lab 1 — Implement custom Dataset and DataLoader for images
* Lab 2 — Add augmentations and profile throughput vs accuracy

3. Mini-Project Idea

* Title: Efficient Image Input Pipeline
* Goal: Build high-throughput input pipeline with augmentations for CNN training
* Dataset keywords: CIFAR-10, custom folder images
* Output keywords: pipeline code, throughput measurements, accuracy results

4. Q&A Seeds (Questions Only)

* Q1: How do DataLoaders affect training speed?
* Q2: How can augmentations act as regularization?
* Q3: How many workers is reasonable for given hardware?
* Q4: What if augmentations introduce label noise?
* Q5: Tradeoffs between complex augmentations and training stability?

---

Day 55: Model Checkpointing and Experiment Tracking

1. Theory Keywords

* state_dict vs full model saving
* Checkpointing frequency
* Resuming from checkpoints
* Basic experiment tracking concepts
* Logging libraries overview
* Reproducible experiment naming and metadata
* Best practices for artifacts organization

2. Hands-on Labs (short specs)

* Lab 1 — Implement robust checkpoint saving and loading
* Lab 2 — Integrate simple experiment tracking into training loop

3. Mini-Project Idea

* Title: Mini Experiment Manager
* Goal: Build lightweight system for managing deep learning experiments
* Dataset keywords: small CNN or MLP tasks
* Output keywords: checkpoints, logs, directory structure, usage guide

4. Q&A Seeds (Questions Only)

* Q1: Which objects should be saved in checkpoints?
* Q2: How to handle backward-incompatible model changes?
* Q3: What minimal information must experiments record?
* Q4: What if runs become too many and messy?
* Q5: Tradeoffs between homemade tracking and external tools?

---

Day 56: Intro to Computer Vision Tasks

1. Theory Keywords

* Image classification task formulation
* Object detection, segmentation overview
* Image preprocessing basics
* Color spaces and normalization
* Data augmentation for vision recap
* Transfer learning high-level idea
* Evaluation metrics for classification

2. Hands-on Labs (short specs)

* Lab 1 — Train CNN for simple image classification from scratch
* Lab 2 — Explore different preprocessing and normalization techniques

3. Mini-Project Idea

* Title: Image Classification Starter
* Goal: Build baseline image classifier with simple CNN and basic augmentations
* Dataset keywords: CIFAR-10, Fashion-MNIST, small custom dataset
* Output keywords: training logs, confusion matrix, misclassification analysis

4. Q&A Seeds (Questions Only)

* Q1: Why are CNNs suited for images?
* Q2: How does normalization affect training stability?
* Q3: Which augmentations are safe for given vision tasks?
* Q4: What if dataset is very small for training from scratch?
* Q5: Tradeoffs between task-specific architectures and generic CNNs?

---

Day 57: Transfer Learning for Vision

1. Theory Keywords

* Pretrained CNN backbones
* Feature extraction vs fine-tuning
* Freezing and unfreezing layers
* Learning rates for different layers
* Domain shift in vision
* Data augmentation with transfer learning
* Evaluation for small datasets

2. Hands-on Labs (short specs)

* Lab 1 — Use pretrained CNN as fixed feature extractor
* Lab 2 — Fine-tune pretrained model on custom dataset

3. Mini-Project Idea

* Title: Small Data Image Classifier with Transfer Learning
* Goal: Build high-performing classifier on small image dataset using pretrained network
* Dataset keywords: custom images, small labeled sets
* Output keywords: fine-tuned model, metrics, comparison to training from scratch

4. Q&A Seeds (Questions Only)

* Q1: When is transfer learning especially beneficial?
* Q2: How to decide which layers to freeze?
* Q3: How does learning rate differ across layers?
* Q4: What if source and target domains differ greatly?
* Q5: Tradeoffs between full fine-tuning and frozen feature extraction?

---

Day 58: Model Explainability in Vision

1. Theory Keywords

* Saliency maps basics
* Grad-CAM concept
* Occlusion sensitivity methods
* Visualization of feature maps
* Limitations of visual explanations
* Spurious correlation detection
* Human evaluation of explanations

2. Hands-on Labs (short specs)

* Lab 1 — Generate saliency maps for CNN predictions
* Lab 2 — Implement Grad-CAM visualizations for selected images

3. Mini-Project Idea

* Title: Vision Model Explainability Toolkit
* Goal: Build notebook to inspect and interpret CNN decisions on image dataset
* Dataset keywords: natural images, classification dataset
* Output keywords: explanation maps, qualitative analysis, report

4. Q&A Seeds (Questions Only)

* Q1: How reliable are saliency-based explanations?
* Q2: How can explanations reveal dataset biases?
* Q3: How to validate that explanations make sense to humans?
* Q4: What if explanations highlight irrelevant regions consistently?
* Q5: Tradeoffs between interpretability and model complexity in vision?

---

Day 59: NLP Foundations and Text Preprocessing

1. Theory Keywords

* Tokenization basics
* Word-level vs subword tokenization
* Vocabulary, OOV handling
* Text normalization steps
* Bag-of-words representation
* TF-IDF weighting concept
* Sequence length handling

2. Hands-on Labs (short specs)

* Lab 1 — Implement preprocessing pipeline from raw text to tokens
* Lab 2 — Build bag-of-words and TF-IDF features for corpus

3. Mini-Project Idea

* Title: Text Feature Extraction Toolkit
* Goal: Build generic text preprocessing and feature extraction module
* Dataset keywords: IMDB reviews, tweets, news articles
* Output keywords: tokenized datasets, feature matrices, vocabulary stats

4. Q&A Seeds (Questions Only)

* Q1: When to use word-level vs character-level tokens?
* Q2: How does TF-IDF emphasize informative words?
* Q3: How to handle very long or very short texts?
* Q4: What if vocabulary grows too large?
* Q5: Tradeoffs between simple bag-of-words and dense embeddings?

---

Day 60: Word Embeddings and Simple NLP Models

1. Theory Keywords

* Distributed representations concept
* Word2vec skip-gram high-level idea
* GloVe embeddings basics
* Cosine similarity for word embeddings
* Using pretrained embeddings
* Simple text classification architectures
* Embedding layers in PyTorch

2. Hands-on Labs (short specs)

* Lab 1 — Load pretrained embeddings and compute word similarities
* Lab 2 — Build simple embedding + averaging classifier for text

3. Mini-Project Idea

* Title: Embedding-Based Text Classifier
* Goal: Use pretrained embeddings for sentiment or topic classification
* Dataset keywords: IMDB subset, news categorization
* Output keywords: metrics, similarity examples, embedding analysis plots

4. Q&A Seeds (Questions Only)

* Q1: How do embeddings capture semantic similarity?
* Q2: How to handle unknown words with pretrained embeddings?
* Q3: When is simple averaging of embeddings sufficient?
* Q4: What if embeddings encode biases from training data?
* Q5: Tradeoffs between training embeddings from scratch and reusing pretrained ones?

---

Day 61: Sequence Models for NLP (RNN/LSTM)

1. Theory Keywords

* Sequence representation in NLP
* RNN/LSTM-based text classifiers
* Padding and masking sequences
* Packed sequences in PyTorch
* Bidirectional LSTMs for text
* Handling variable-length sequences
* Evaluation metrics for NLP tasks

2. Hands-on Labs (short specs)

* Lab 1 — Implement LSTM-based sentiment classifier
* Lab 2 — Compare uni- vs bidirectional LSTM performance

3. Mini-Project Idea

* Title: LSTM Sentiment Analyzer
* Goal: Build and evaluate LSTM model for movie reviews
* Dataset keywords: IMDB, Amazon reviews
* Output keywords: accuracy, confusion matrix, qualitative error examples

4. Q&A Seeds (Questions Only)

* Q1: Why is masking important for padded sequences?
* Q2: How do bidirectional LSTMs improve context modeling?
* Q3: How to choose hidden size and number of layers?
* Q4: What if sequences exceed maximum allowed length?
* Q5: Tradeoffs between CNN-based and RNN-based text models?

---

Day 62: Intro to Transformer-Based NLP

1. Theory Keywords

* Subword tokenization (BPE, WordPiece)
* Transformer encoders for text
* Positional encodings in NLP
* Pretrained language models concept
* Fine-tuning vs feature extraction
* Common architectures (BERT-style)
* Attention visualization for text

2. Hands-on Labs (short specs)

* Lab 1 — Use pretrained transformer encoder for text classification
* Lab 2 — Inspect attention patterns for sample sentences

3. Mini-Project Idea

* Title: BERT-Style Text Classifier
* Goal: Fine-tune small transformer model on simple classification task
* Dataset keywords: sentiment dataset, topic classification
* Output keywords: metrics, training logs, attention visualizations

4. Q&A Seeds (Questions Only)

* Q1: Why subword tokenization helps with rare words?
* Q2: How does fine-tuning differ from using fixed embeddings?
* Q3: How to choose maximum sequence length?
* Q4: What if GPU memory is insufficient for larger models?
* Q5: Tradeoffs between classical NLP models and transformer-based models?

---

Day 63: Recommender Systems Fundamentals

1. Theory Keywords

* Explicit vs implicit feedback
* User-item interaction matrix
* Popularity-based baselines
* User-based and item-based k-NN
* Evaluation metrics for recommenders
* Cold start problem
* Sparsity issues in interaction data

2. Hands-on Labs (short specs)

* Lab 1 — Implement simple popularity and k-NN recommenders
* Lab 2 — Evaluate recommenders using appropriate ranking metrics

3. Mini-Project Idea

* Title: Simple Movie Recommender Baselines
* Goal: Build basic recommenders for movie ratings dataset
* Dataset keywords: MovieLens ratings, user-item interactions
* Output keywords: top-N recommendations, metric reports, comparison table

4. Q&A Seeds (Questions Only)

* Q1: Why are recommenders naturally dealing with sparse data?
* Q2: How do user-based and item-based k-NN differ?
* Q3: Which metrics are appropriate for top-N recommendation?
* Q4: What if users or items have no history?
* Q5: Tradeoffs between simple heuristics and model-based recommenders?

---

Day 64: Matrix Factorization for Recommenders

1. Theory Keywords

* Low-rank factorization of interaction matrix
* Latent factors for users and items
* Optimization objective (MSE, regularization)
* Handling implicit feedback high-level
* Cold start limitations for MF
* Bias terms (global, user, item)
* Evaluation on held-out interactions

2. Hands-on Labs (short specs)

* Lab 1 — Implement basic matrix factorization model with SGD
* Lab 2 — Analyze learned latent factors and biases

3. Mini-Project Idea

* Title: Latent Factor Movie Recommender
* Goal: Build and evaluate matrix factorization recommender on MovieLens
* Dataset keywords: MovieLens, user-item ratings
* Output keywords: recommendation quality, factor visualizations, error analysis

4. Q&A Seeds (Questions Only)

* Q1: How does MF capture user preferences?
* Q2: Why is regularization crucial in MF?
* Q3: How to choose latent dimension size?
* Q4: What if interaction matrix is extremely sparse?
* Q5: Tradeoffs between MF simplicity and more complex neural recommenders?

---

Day 65: Neural Recommenders and Embeddings

1. Theory Keywords

* User and item embeddings in neural networks
* Dot-product vs MLP interaction
* Handling side information features
* Negative sampling for implicit feedback
* Ranking losses (BPR, hinge)
* Co-training with other tasks
* Serving recommendations efficiently

2. Hands-on Labs (short specs)

* Lab 1 — Implement simple neural recommender with user/item embeddings
* Lab 2 — Compare rating prediction vs ranking objectives

3. Mini-Project Idea

* Title: Neural Movie Recommender Prototype
* Goal: Build basic neural collaborative filtering model and evaluate rankings
* Dataset keywords: MovieLens, implicit feedback variant
* Output keywords: ranking metrics, learned embeddings, qualitative recommendations

4. Q&A Seeds (Questions Only)

* Q1: How do learned embeddings differ from MF factors?
* Q2: When to use dot-product vs richer interaction functions?
* Q3: How to incorporate contextual features into recommenders?
* Q4: What if negative sampling strategy is poor?
* Q5: Tradeoffs between recommendation quality and serving latency?

---

Day 66: Time Series Basics

1. Theory Keywords

* Time series vs i.i.d. data
* Trends, seasonality, residuals
* Stationarity concept
* Train-test split over time
* Simple baselines (naive, moving average)
* Lagged features construction
* Evaluation metrics for forecasting

2. Hands-on Labs (short specs)

* Lab 1 — Explore and decompose time series dataset
* Lab 2 — Implement simple baseline forecasters and evaluate

3. Mini-Project Idea

* Title: Time Series Baseline Forecaster
* Goal: Build baseline forecasting models for univariate time series
* Dataset keywords: energy consumption, traffic, sales
* Output keywords: forecast plots, error metrics, baseline comparisons

4. Q&A Seeds (Questions Only)

* Q1: Why must time order be preserved in splitting?
* Q2: How do trend and seasonality affect modeling choices?
* Q3: When are simple baselines surprisingly strong?
* Q4: What if series exhibit structural breaks?
* Q5: Tradeoffs between global models and per-series models?

---

Day 67: Deep Learning for Time Series

1. Theory Keywords

* Sliding window approach
* Sequence-to-one and sequence-to-sequence forecasting
* LSTM-based forecasters
* Temporal convolutions overview
* Multi-step forecasting strategies
* Handling multiple correlated series
* Evaluation over forecasting horizon

2. Hands-on Labs (short specs)

* Lab 1 — Implement LSTM-based forecaster for univariate series
* Lab 2 — Evaluate multi-step forecasts vs baselines

3. Mini-Project Idea

* Title: LSTM Forecaster for Real Time Series
* Goal: Build LSTM forecasting model for real dataset and compare to baselines
* Dataset keywords: traffic data, energy, financial time series
* Output keywords: forecast curves, horizon-wise metrics, comparison plots

4. Q&A Seeds (Questions Only)

* Q1: How to construct input/output windows for time series models?
* Q2: How does multi-step forecasting differ from one-step-ahead?
* Q3: How to model multiple related series jointly?
* Q4: What if series length is short for deep models?
* Q5: Tradeoffs between classical and deep forecasting methods?

---

Day 68: Audio and Speech Basics

1. Theory Keywords

* Audio waveform representation
* Sampling rate and Nyquist theorem
* Short-time Fourier transform
* Spectrograms and log-mel spectrograms
* Basic speech preprocessing
* Windowing and framing
* Simple audio augmentation

2. Hands-on Labs (short specs)

* Lab 1 — Load audio, compute and visualize spectrograms
* Lab 2 — Apply simple augmentations (noise, time-shift) and compare features

3. Mini-Project Idea

* Title: Audio Feature Extraction Toolkit
* Goal: Build pipeline converting audio files into model-ready spectrogram features
* Dataset keywords: speech commands, environmental sounds
* Output keywords: spectrogram datasets, visualization plots, summary stats

4. Q&A Seeds (Questions Only)

* Q1: Why use spectrograms instead of raw waveforms sometimes?
* Q2: How does sampling rate affect representational fidelity?
* Q3: Why are log-mel spectrograms popular for speech?
* Q4: What if audio recordings have very different loudness levels?
* Q5: Tradeoffs between richer audio features and model complexity?

---

Day 69: Audio Classification with CNNs

1. Theory Keywords

* Treating spectrograms as images
* CNN architectures for audio
* Time-frequency invariances
* Pooling strategies in time and frequency
* Data augmentation for audio
* Evaluation metrics for audio classification
* Handling variable-length audio clips

2. Hands-on Labs (short specs)

* Lab 1 — Train CNN on spectrograms for simple audio classification
* Lab 2 — Explore different window sizes and hop lengths

3. Mini-Project Idea

* Title: Simple Speech Command Classifier
* Goal: Build CNN-based classifier for spoken word commands
* Dataset keywords: speech commands dataset, small audio corpus
* Output keywords: accuracy metrics, confusion matrix, spectrogram visualizations

4. Q&A Seeds (Questions Only)

* Q1: How does treating spectrograms as images help modeling?
* Q2: How do time and frequency pooling differ in effect?
* Q3: Which augmentations are sensible for audio tasks?
* Q4: What if audio length varies significantly across samples?
* Q5: Tradeoffs between raw waveform models and spectrogram-based models?

---

Day 70: Multi-Task and Multi-Modal Basics

1. Theory Keywords

* Multi-task learning principle
* Shared vs task-specific layers
* Loss weighting across tasks
* Multi-modal data concept
* Fusion strategies (early, late)
* Regularization benefits of sharing
* Negative transfer risks

2. Hands-on Labs (short specs)

* Lab 1 — Implement simple multi-task network with shared backbone
* Lab 2 — Combine image and tabular features in basic multi-modal model

3. Mini-Project Idea

* Title: Multi-Task Classifier Prototype
* Goal: Build network predicting multiple related labels jointly
* Dataset keywords: image plus attributes, tabular with multiple targets
* Output keywords: task-specific metrics, sharing analysis, ablation results

4. Q&A Seeds (Questions Only)

* Q1: How can multi-task learning improve generalization?
* Q2: How to design shared versus task-specific components?
* Q3: When do multi-modal inputs provide clear benefit?
* Q4: What if tasks conflict and hurt performance?
* Q5: Tradeoffs between single-task simplicity and multi-task efficiency?

---

Day 71: End-to-End Vision Project Planning

1. Theory Keywords

* Problem scoping in vision
* Data collection and labeling considerations
* Baselines vs ambitious models
* Train/validation/test protocol for images
* Augmentation and preprocessing choices
* Metrics and business goals alignment
* Error analysis plan for vision

2. Hands-on Labs (short specs)

* Lab 1 — Design and document full plan for vision project
* Lab 2 — Prepare dataset splits and basic exploratory analysis

3. Mini-Project Idea

* Title: Vision Project Blueprint
* Goal: Plan end-to-end image classification or detection project with detailed steps
* Dataset keywords: chosen real image dataset
* Output keywords: project document, dataset splits, baseline design

4. Q&A Seeds (Questions Only)

* Q1: How to choose appropriate scope for initial vision project?
* Q2: Which dataset issues can derail vision performance?
* Q3: How to link evaluation metrics to real-world costs?
* Q4: What if labeling quality is inconsistent?
* Q5: Tradeoffs between perfect data and faster iteration?

---

Day 72: End-to-End NLP Project Planning

1. Theory Keywords

* NLP problem types (classification, tagging, QA)
* Data collection and annotation for text
* Handling sensitive content in NLP
* Tokenization and vocabulary decisions
* Handling multilingual data
* Metrics for NLP tasks
* Error analysis planning for text

2. Hands-on Labs (short specs)

* Lab 1 — Design end-to-end plan for NLP task on chosen dataset
* Lab 2 — Preprocess raw text and create baseline features

3. Mini-Project Idea

* Title: NLP Project Blueprint
* Goal: Prepare detailed plan for sentiment, topic, or intent classification system
* Dataset keywords: reviews, support tickets, chat logs
* Output keywords: project spec, preprocessing pipeline, baseline model plan

4. Q&A Seeds (Questions Only)

* Q1: How to choose appropriate NLP formulation for problem?
* Q2: Which preprocessing steps are high-risk for information loss?
* Q3: How to address label noise in text datasets?
* Q4: What if domain vocabulary shifts over time?
* Q5: Tradeoffs between custom models and pretrained language models?

---

Day 73: End-to-End Recommender Project Planning

1. Theory Keywords

* Problem framing for recommendation
* Logging and feedback loops
* Offline vs online evaluation
* Candidate generation vs ranking stages
* Feature engineering for recommenders
* Cold start mitigation strategies
* Ethical considerations (filter bubbles)

2. Hands-on Labs (short specs)

* Lab 1 — Design architecture diagram for recommender system
* Lab 2 — Create offline evaluation setup for existing interaction dataset

3. Mini-Project Idea

* Title: Recommender System Blueprint
* Goal: Plan multi-stage recommender pipeline for movie or product domain
* Dataset keywords: MovieLens, e-commerce interactions
* Output keywords: system design document, evaluation plan, baseline models

4. Q&A Seeds (Questions Only)

* Q1: How to combine candidate generation with ranking effectively?
* Q2: How to design robust offline evaluation for recommenders?
* Q3: How to mitigate cold start for new users?
* Q4: What if recommender reinforces undesirable biases?
* Q5: Tradeoffs between model complexity and serving infrastructure simplicity?

---

Day 74: Vision Project Implementation I (Baseline)

1. Theory Keywords

* Baseline CNN architecture selection
* Preprocessing pipeline finalization
* Logging metrics and artifacts
* Overfitting monitoring
* Training time vs model size
* Early stopping criteria
* Checkpointing strategy

2. Hands-on Labs (short specs)

* Lab 1 — Implement full baseline CNN training for chosen vision dataset
* Lab 2 — Log metrics, checkpoints, and basic error examples

3. Mini-Project Idea

* Title: Vision Baseline Implementation
* Goal: Deliver working baseline classifier per blueprint for vision project
* Dataset keywords: chosen vision dataset from planning
* Output keywords: trained baseline model, logs, initial metrics, sample predictions

4. Q&A Seeds (Questions Only)

* Q1: How does baseline performance compare to naive expectations?
* Q2: Which bottlenecks appeared most prominently during training?
* Q3: How to prioritize next architecture improvements?
* Q4: What if validation accuracy stagnates early?
* Q5: Tradeoffs between more data cleaning and more model tuning?

---

Day 75: NLP Project Implementation I (Baseline)

1. Theory Keywords

* Baseline model selection for NLP
* Feature representation choice
* Handling OOV and rare tokens
* Class imbalance in text datasets
* Logging qualitative text errors
* Training runtime considerations
* Simple interpretability for NLP

2. Hands-on Labs (short specs)

* Lab 1 — Implement baseline NLP classifier as per blueprint
* Lab 2 — Collect representative error examples by category

3. Mini-Project Idea

* Title: NLP Baseline Implementation
* Goal: Deliver working baseline NLP classifier with full pipeline
* Dataset keywords: chosen NLP dataset from planning
* Output keywords: metrics, confusion matrix, error examples, trained model

4. Q&A Seeds (Questions Only)

* Q1: How good is baseline compared to simple heuristics?
* Q2: Which failure modes are most common in predictions?
* Q3: How to refine preprocessing to address observed errors?
* Q4: What if certain classes almost never predicted?
* Q5: Tradeoffs between bag-of-words baselines and deep NLP models?

---

Day 76: Recommender Project Implementation I (Baseline)

1. Theory Keywords

* Baseline recommender selection
* Data splitting over time
* Evaluation metrics for chosen task
* Handling extremely sparse interactions
* Logging recommendation lists and feedback
* Popularity and random baselines
* Coverage and diversity metrics

2. Hands-on Labs (short specs)

* Lab 1 — Implement popularity and k-NN baselines offline
* Lab 2 — Evaluate baselines with ranking metrics and coverage

3. Mini-Project Idea

* Title: Recommender Baseline Implementation
* Goal: Build baseline recommender pipeline per blueprint
* Dataset keywords: chosen recommender dataset from planning
* Output keywords: top-N recommendation lists, metric reports, baseline code

4. Q&A Seeds (Questions Only)

* Q1: How do baselines perform relative to naive random recommendations?
* Q2: How does time-aware splitting affect evaluation results?
* Q3: How to interpret coverage and diversity metrics?
* Q4: What if popular items dominate recommendations excessively?
* Q5: Tradeoffs between accuracy and diversity in recommender design?

---

Day 77: Vision Project Implementation II (Improvements)

1. Theory Keywords

* Transfer learning in project context
* Augmentation tuning
* Learning rate schedules revisited
* Architectural tweaks exploration
* Regularization adjustments
* Model ensembling basics
* Error-driven model refinement

2. Hands-on Labs (short specs)

* Lab 1 — Add transfer learning model and compare performance
* Lab 2 — Tune augmentations and regularization based on error analysis

3. Mini-Project Idea

* Title: Vision Project Improvement Round
* Goal: Iterate on baseline vision model using systematic experiments
* Dataset keywords: same vision dataset
* Output keywords: improved metrics, experiment logs, ablation results

4. Q&A Seeds (Questions Only)

* Q1: Which changes gave largest performance gains?
* Q2: How to avoid overfitting during aggressive improvement?
* Q3: How to structure ablation studies effectively?
* Q4: What if more complex models only marginally help?
* Q5: Tradeoffs between squeezing extra accuracy and engineering cost?

---

Day 78: NLP Project Implementation II (Improvements)

1. Theory Keywords

* Transformer-based fine-tuning in project
* Improved tokenization decisions
* Class-weighted loss for NLP
* Curriculum learning idea
* Domain adaptation basics
* Interpretability for advanced NLP models
* Robustness to shifted language distributions

2. Hands-on Labs (short specs)

* Lab 1 — Fine-tune small transformer model on project dataset
* Lab 2 — Compare transformer performance with baseline model

3. Mini-Project Idea

* Title: NLP Project Improvement Round
* Goal: Upgrade baseline NLP system using modern architectures
* Dataset keywords: same NLP dataset
* Output keywords: new metrics, error analysis, ablation experiments

4. Q&A Seeds (Questions Only)

* Q1: How much uplift from transformer vs baseline?
* Q2: Did error types change with new model?
* Q3: How to handle out-of-domain test examples?
* Q4: What if model memorizes training data phrases?
* Q5: Tradeoffs between model size and latency for NLP deployment?

---

Day 79: Recommender Project Implementation II (Improvements)

1. Theory Keywords

* Matrix factorization in project context
* Neural recommender integration
* Negative sampling strategy refinement
* Contextual features incorporation
* Evaluation by user segments
* Long-tail recommendation strategies
* A/B testing high-level idea

2. Hands-on Labs (short specs)

* Lab 1 — Add MF and simple neural recommender to pipeline
* Lab 2 — Evaluate performance by user and item groups

3. Mini-Project Idea

* Title: Recommender Project Improvement Round
* Goal: Enhance recommender using MF and neural approaches
* Dataset keywords: same recommender dataset
* Output keywords: improved metrics, segment-wise analysis, ablation studies

4. Q&A Seeds (Questions Only)

* Q1: How do new models change recommendation diversity?
* Q2: Which user groups benefit most from improvements?
* Q3: How to mitigate popularity bias after improvements?
* Q4: What if offline metrics improve but business proxies might not?
* Q5: Tradeoffs between complexity of recommender and maintainability?

---

Day 80: Integrated Time Series Project

1. Theory Keywords

* Forecasting problem setup
* Baseline and deep models comparison
* Feature engineering for time series
* Evaluation protocol over horizons
* Robustness to missing timestamps
* Seasonal adjustment approaches
* Uncertainty estimation basics

2. Hands-on Labs (short specs)

* Lab 1 — Implement baseline and LSTM forecasters for chosen series
* Lab 2 — Evaluate performance across multiple forecast horizons

3. Mini-Project Idea

* Title: End-to-End Forecasting System
* Goal: Build complete forecasting pipeline with classical and deep models
* Dataset keywords: energy usage, traffic, sales time series
* Output keywords: forecast plots, metrics, comparison report, codebase

4. Q&A Seeds (Questions Only)

* Q1: How do baselines compare with deep models in this dataset?
* Q2: Which feature engineering choices mattered most?
* Q3: How to communicate forecast uncertainty effectively?
* Q4: What if sudden regime change occurs in series?
* Q5: Tradeoffs between short-horizon and long-horizon forecast accuracy?

---

Day 81: Evaluation and Robustness Across Domains

1. Theory Keywords

* Distribution shift and domain generalization
* Cross-domain evaluation strategies
* Data augmentation for robustness
* Adversarial examples high-level overview
* Stress testing models
* Out-of-distribution detection basics
* Fairness and bias considerations

2. Hands-on Labs (short specs)

* Lab 1 — Evaluate existing models on shifted or perturbed data
* Lab 2 — Implement simple OOD detection heuristic

3. Mini-Project Idea

* Title: Robustness Analysis for One Project
* Goal: Perform robustness and fairness analyses on one chosen domain project
* Dataset keywords: one of previous projects’ datasets
* Output keywords: robustness metrics, stress-test results, recommendations

4. Q&A Seeds (Questions Only)

* Q1: How to simulate realistic distribution shifts?
* Q2: How can augmentation help but also harm?
* Q3: How to define fairness metrics for given project?
* Q4: What if model performance collapses under small perturbations?
* Q5: Tradeoffs between robustness, accuracy, and computational cost?

---

Day 82: Model Compression and Deployment Prep

1. Theory Keywords

* Parameter pruning basics
* Quantization overview
* Knowledge distillation concept
* Latency vs accuracy tradeoffs
* Memory footprint considerations
* Batch size and throughput in serving
* Exporting models for deployment

2. Hands-on Labs (short specs)

* Lab 1 — Apply simple pruning or quantization to trained model
* Lab 2 — Measure speed and accuracy before and after compression

3. Mini-Project Idea

* Title: Lightweight Model Variant
* Goal: Create compressed version of one project model suitable for constrained environments
* Dataset keywords: chosen project dataset
* Output keywords: compressed model, latency metrics, accuracy comparison, deployment notes

4. Q&A Seeds (Questions Only)

* Q1: Which components are good candidates for pruning?
* Q2: How does quantization affect numerical behavior?
* Q3: How to design student networks for distillation?
* Q4: What if compression severely hurts minority-class performance?
* Q5: Tradeoffs between model compactness and future maintainability?

---

Day 83: Experiment Design and Ablations

1. Theory Keywords

* Controlled experiments in ML
* Ablation study design
* One-factor-at-a-time vs factorial designs
* Reproducibility in experiments
* Reporting uncertainty in results
* Avoiding p-hacking in ML context
* Communicating findings effectively

2. Hands-on Labs (short specs)

* Lab 1 — Design and run ablation study for one project model
* Lab 2 — Summarize experiment results in clear tables and plots

3. Mini-Project Idea

* Title: Comprehensive Ablation Report
* Goal: Produce thorough ablation study for selected project component
* Dataset keywords: any project dataset
* Output keywords: experiment grid, metrics tables, concise report, plots

4. Q&A Seeds (Questions Only)

* Q1: How to choose factors for ablation?
* Q2: How many runs per configuration to trust results?
* Q3: How to separate signal from noise in experiment outcomes?
* Q4: What if ablations reveal minimal effect for complex component?
* Q5: Tradeoffs between breadth and depth of experimental exploration?

---

Day 84: Putting It Together: Mini AI Portfolio Design

1. Theory Keywords

* Project selection for portfolio
* Balancing breadth and depth
* Storytelling with projects
* Code quality and documentation
* Reproducibility for external reviewers
* Visualization and reporting quality
* Ethical and privacy considerations

2. Hands-on Labs (short specs)

* Lab 1 — Select and prioritize 3–4 strongest projects
* Lab 2 — Plan improvements and documentation for portfolio readiness

3. Mini-Project Idea

* Title: AI Foundations Portfolio Plan
* Goal: Design coherent portfolio from completed projects with clear narratives
* Dataset keywords: previously used datasets
* Output keywords: portfolio outline, improvement checklist, documentation plan

4. Q&A Seeds (Questions Only)

* Q1: Which projects best showcase progression from math to applications?
* Q2: How to present failures and learnings constructively?
* Q3: How to ensure projects are reproducible by others?
* Q4: What if some projects feel redundant thematically?
* Q5: Tradeoffs between polishing existing work and adding new projects?

---

Day 85: Portfolio Project Polish I (Vision or NLP)

1. Theory Keywords

* Refactoring project code
* Improving experiment organization
* Enhancing data visualizations
* Writing clearer README and docs
* Adding tests for critical components
* Packaging project for reuse
* Licensing and attribution basics

2. Hands-on Labs (short specs)

* Lab 1 — Refactor and clean code for chosen flagship project
* Lab 2 — Improve documentation and visuals for same project

3. Mini-Project Idea

* Title: Flagship Project Polished Release
* Goal: Turn best vision or NLP project into polished, shareable repository
* Dataset keywords: chosen flagship project dataset
* Output keywords: clean codebase, docs, plots, installation instructions

4. Q&A Seeds (Questions Only)

* Q1: Which parts of code most needed refactoring?
* Q2: How can visuals better communicate model performance?
* Q3: How to structure repository for easy navigation?
* Q4: What if project lacks automated tests?
* Q5: Tradeoffs between adding features and ensuring stability?

---

Day 86: Portfolio Project Polish II (Recommender or Time Series)

1. Theory Keywords

* Clarity of evaluation methodology
* Parameter documentation
* Result reproducibility scripts
* Visualizing longitudinal metrics
* Business-oriented interpretation of results
* Limitations and future work sections
* Comparison against baselines clarity

2. Hands-on Labs (short specs)

* Lab 1 — Clean and document recommender or forecasting project
* Lab 2 — Add reproducible evaluation scripts and summary notebook

3. Mini-Project Idea

* Title: Applied ML Project Polished Release
* Goal: Produce polished version of recommender or time series project
* Dataset keywords: chosen applied project dataset
* Output keywords: clear metrics, comparison tables, business-focused discussion

4. Q&A Seeds (Questions Only)

* Q1: How to explain project value to non-technical stakeholders?
* Q2: How to highlight robustness and limitations honestly?
* Q3: Which baselines need clearer justification and comparison?
* Q4: What if reproduction yields slightly different results?
* Q5: Tradeoffs between technical detail and accessibility in documentation?

---

Day 87: Capstone Integration Project Design

1. Theory Keywords

* Multi-domain integration opportunities
* Combining models in pipelines
* Data flow and orchestration
* Monitoring and logging across components
* Failure modes in integrated systems
* Scalability considerations high-level
* Iterative deployment roadmap

2. Hands-on Labs (short specs)

* Lab 1 — Sketch architecture of capstone system using existing components
* Lab 2 — Define evaluation and monitoring plan for integrated system

3. Mini-Project Idea

* Title: Capstone AI System Blueprint
* Goal: Design integrated system combining at least two core domains
* Dataset keywords: combination of image, text, or interaction data
* Output keywords: architecture diagrams, design document, evaluation plan

4. Q&A Seeds (Questions Only)

* Q1: Which existing components integrate naturally?
* Q2: What are primary data flow bottlenecks?
* Q3: How to monitor health of system in production-like setting?
* Q4: What if components fail or deliver inconsistent predictions?
* Q5: Tradeoffs between monolithic and modular system designs?

---

Day 88: Capstone Mini Implementation

1. Theory Keywords

* Minimal viable integration
* Interfaces between components
* Data format agreements
* Error handling across modules
* Logging and tracing across pipeline
* Performance measurement end-to-end
* Manual vs automated orchestration

2. Hands-on Labs (short specs)

* Lab 1 — Implement minimal integrated pipeline for capstone blueprint
* Lab 2 — Run small-scale end-to-end tests and collect metrics

3. Mini-Project Idea

* Title: Capstone System Prototype
* Goal: Build working minimal version of integrated AI system
* Dataset keywords: reduced-scale combination of selected datasets
* Output keywords: pipeline code, logs, basic metrics, example outputs

4. Q&A Seeds (Questions Only)

* Q1: Which integration issues were hardest technically?
* Q2: How does end-to-end performance compare to component-level metrics?
* Q3: How to prioritize improvements to integrated system?
* Q4: What if one component becomes bottleneck for entire pipeline?
* Q5: Tradeoffs between building more features vs hardening existing pipeline?

---

Day 89: Capstone Refinement and Evaluation

1. Theory Keywords

* Holistic evaluation of system
* User-centric metrics and UX considerations
* Robustness under edge cases
* Logging for future debugging
* Documentation of overall system behavior
* Risk assessment and mitigation
* Roadmap for future extensions

2. Hands-on Labs (short specs)

* Lab 1 — Conduct detailed evaluation of capstone prototype
* Lab 2 — Document system behavior, limitations, and future work

3. Mini-Project Idea

* Title: Capstone Evaluation Report
* Goal: Produce comprehensive evaluation and discussion of capstone system
* Dataset keywords: same as capstone prototype
* Output keywords: metrics summary, error analysis, limitations, roadmap document

4. Q&A Seeds (Questions Only)

* Q1: Does capstone system meet original design goals?
* Q2: Which failure cases are most concerning in practice?
* Q3: How could system be scaled or extended realistically?
* Q4: What if stakeholders prioritize different metrics than currently optimized?
* Q5: Tradeoffs between pursuing new capabilities and shoring up existing weaknesses?

---

Day 90: Reflection and Next-Phase Planning

1. Theory Keywords

* Review of math foundations
* Review of classical ML skills
* Review of deep learning abilities
* Strengths and weaknesses assessment
* Gaps in domains (vision, NLP, recsys, time series, audio)
* Learning strategies reflection
* Planning for Phase 2

2. Hands-on Labs (short specs)

* Lab 1 — Summarize key learnings and artifacts from 90 days
* Lab 2 — Define concrete goals and topics for next learning phase

3. Mini-Project Idea

* Title: AI Foundations Retrospective
* Goal: Consolidate all work into clear narrative and roadmap for progression
* Dataset keywords: all prior project artifacts
* Output keywords: retrospective document, prioritized learning plan, portfolio snapshot

4. Q&A Seeds (Questions Only)

* Q1: Which areas feel most solid theoretically and practically?
* Q2: Which parts of pipeline building still feel uncomfortable?
* Q3: How to deepen expertise in favorite domain next?
* Q4: What recurring mistakes appeared across projects?
* Q5: Tradeoffs between specializing deeply now vs broadening further first?
