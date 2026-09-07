# Neural Net Experiments

A chronological set of Python experiments documenting a hands-on introduction to machine learning and neural networks. The project starts with NumPy fundamentals and gradually builds toward regression, classification, backpropagation, multi-layer networks, validation, mini-batch training, and optimization methods.

The emphasis is on learning the mechanics: most models are implemented directly with NumPy instead of relying on a deep-learning framework. The scripts retain explanatory comments and intermediate output, so they are intended to be read, modified, and run as individual experiments—not imported as a production package.

## What is in this repository?

Each `d<day>-<date>.py` file is a self-contained daily exercise. The files are ordered by learning progression, from array manipulation through a general-purpose, batched neural-network trainer.

| Stage | Files | Topics |
| --- | --- | --- |
| NumPy foundations | Days 1–9 | arrays, indexing and slicing, broadcasting, statistics, matrices, randomness, simulation, vectors, dot products, cosine similarity, and matrix multiplication |
| Classical ML foundations | Days 10–17 | k-nearest neighbors, linear regression, numerical and analytical gradients, loss functions, and vectorized regression |
| Classification | Days 18–20 | logistic regression, sigmoid, binary cross-entropy, softmax, categorical cross-entropy, and multiclass prediction |
| Neural networks and optimization | Days 21–30 | ReLU networks, backpropagation, decision-region graphs, validation and hyperparameter search, multiple hidden layers, He initialization, mini-batches, learning-rate decay, momentum, and Adam |
| Concept review | Day 31 | A written review of network shapes, forward propagation, backpropagation, and Adam |

## Learning path

| Day | File | Focus |
| ---: | --- | --- |
| 1 | `d1-7-25.py` | Creating and reshaping NumPy arrays, ranges, grids, and random matrices. |
| 2 | `d2-7-26.py` | Indexing, slicing, copying, masking, and in-place array updates. |
| 3 | `d3-7-27.py` | Broadcasting and feature normalization. |
| 4 | `d4-7-27.py` | Descriptive statistics, percentiles, and boolean counting. |
| 5 | `d5-7-28.py` | Matrix multiplication, transposes, identity matrices, and dimensions. |
| 6 | `d6-7-28.py` | Random sampling, frequency analysis, and probability simulations. |
| 7 | `d7-7-29.py` | A NumPy-based roulette simulation. |
| 8 | `d8-7-29.py` | Vector operations, a manual dot product, and cosine similarity. |
| 9 | `d9-7-29.py` | Additional matrix-multiplication practice. |
| 10 | `d10-7-29.py` | A vectorized k-nearest-neighbors classifier. |
| 11 | `d11-7-30.py` | A first linear-regression model and comparison with `numpy.polyfit`. |
| 12 | `d12-7-31.py` | An initial loss-optimization experiment. |
| 13 | `d13-7-31.py` | Gradient descent for fitting a linear model. |
| 14 | `d14-8-1.py` | Mean squared error (MSE) and mean absolute error (MAE). |
| 15 | `d15-8-1.py` | A multivariable linear-regression project using synthetic student data. |
| 16 | `d16-8-3.py` | Analytical gradients and iterative optimization. |
| 17 | `d17-8-3.py` | Vectorized linear regression and a scikit-learn comparison. |
| 18 | `d18-8-4.py` | Logistic regression, sigmoid activation, and binary cross-entropy. |
| 19 | `d19-8-6.py` | Multiclass classification with softmax and categorical cross-entropy. |
| 20 | `d20-8-7.py` | Improvements to the multiclass classifier. |
| 21 | `d21-8-9.py` | A first neural network: ReLU hidden layer, softmax outputs, and backpropagation. |
| 22 | `d22-8-10.py` | Neural-network visualization and decision regions. |
| 23 | `d23-8-13.py` | XOR experiments and further neural-network exploration. |
| 24 | `d24-8-13.py` | Train/test/validation splits and a small hyperparameter search. |
| 25 | `d25-8-14-15.py` | Generalizing a network to support one or multiple hidden layers. |
| 26 | `d26-8-16.py` | Multi-layer training improvements and He initialization. |
| 27 | `d27-8-16.py` | Mini-batch training, shuffled batches, loss tracking, and testing. |
| 28 | `d28-8-18.py` | Learning-rate reduction when loss improvement stalls. |
| 29 | `d29-8-19.py` | Momentum-based gradient updates and loss visualization. |
| 30 | `d30-8-24-26.py` | An Adam optimizer experiment with first and second moments plus bias correction. |
| 31 | `d31-8-30-9-6.MD` | Written review of network foundations, shapes, forward passes, backpropagation, and Adam. |

`numpy-testing.py` contains smaller scratch experiments and helper-function practice.

## Featured experiments: neural-network training

The later exercises train fully connected classifiers on synthetic student data. The input features are hours studied and hours slept; the model predicts one of five grade bands: `F`, `D`, `C`, `B`, or `A`.

The Day 27 `batched_nn` function accepts a layer layout such as `[2, 8, 8, 5]`, then:

1. initializes every dense layer (with optional He initialization);
2. runs a forward pass using ReLU hidden activations and a softmax output;
3. calculates sparse categorical cross-entropy;
4. propagates gradients backward through each layer; and
5. updates weights and biases one shuffled mini-batch at a time.

It reports training progress, evaluates a held-out test set with a confusion matrix, and plots loss by epoch. The following scripts continue that same from-scratch model:

- Day 28 reduces the learning rate when loss improvement plateaus.
- Day 29 adds velocity-based momentum to weight and bias updates.
- Day 30 explores Adam-style updates using exponentially weighted first and second gradient moments with bias correction.
- Day 31 records a written check-in on how the network’s shapes, forward pass, backpropagation, and optimizer fit together.

## Requirements

- Python 3.10 or newer is recommended.
- [NumPy](https://numpy.org/) is required by every main experiment.
- [Matplotlib](https://matplotlib.org/) is required for the plotting exercises (Days 22, 23, 27, 29, and 30).
- [scikit-learn](https://scikit-learn.org/) is used only in Day 17 for comparison with `LinearRegression`.

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy matplotlib scikit-learn
```

## Running an experiment

Scripts are independent and should be run from the repository root:

```bash
python3 d1-7-25.py
python3 d17-8-3.py
python3 d30-8-24-26.py
```

Some scripts generate different results on each run because they create synthetic data with NumPy randomness. Days 22, 23, 27, 29, and 30 open Matplotlib windows; close the window to let the script finish. Day 18 prompts for desired study and sleep hours after training.

For a good guided tour, run the files in day order. If you want to start with the most advanced optimizer experiment, begin with `d30-8-24-26.py` and then trace backward through Days 29–21. Day 31 is a Markdown review note and is meant to be read rather than executed.

## Project conventions and scope

- Source filenames encode the learning day and the date of the exercise; Day 31 is a Markdown review note rather than an executable script.
- Most scripts run immediately when executed and print their observations, diagnostics, predictions, accuracy, or loss values to the terminal.
- Datasets are generated in code. There are no external data files or downloads.
- Code intentionally favors explicit, commented steps over library abstractions, making the underlying linear algebra and gradient calculations visible.
- This is an educational notebook-style repository, not a reusable ML package. There is no test suite, command-line interface, model persistence, or pinned dependency file yet.
