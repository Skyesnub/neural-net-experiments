# Neural Net Experiments

A chronological set of Python experiments documenting a hands-on introduction to machine learning and neural networks. The project starts with NumPy fundamentals and progresses through regression, classification, backpropagation, regularization, optimization methods, and real-world datasets.

The emphasis is on learning the mechanics: most models are implemented directly with NumPy instead of relying on a deep-learning framework. The scripts retain explanatory comments and intermediate output, so they are intended to be read, modified, and run as individual experiments—not imported as a production package.

## What is in this repository?

Each `d<day>-<date>.py` file is a self-contained daily exercise. Files are grouped into folders by learning stage, from array manipulation through neural-network regularization and experiments on local wine datasets.

| Stage | Files | Topics |
| --- | --- | --- |
| NumPy foundations | Days 1–9 | arrays, indexing and slicing, broadcasting, statistics, matrices, randomness, simulation, vectors, dot products, cosine similarity, and matrix multiplication |
| Classical ML foundations | Days 10–17 | k-nearest neighbors, linear regression, numerical and analytical gradients, loss functions, and vectorized regression |
| Classification | Days 18–20 | logistic regression, sigmoid, binary cross-entropy, softmax, categorical cross-entropy, and multiclass prediction |
| Basic neural networks | Days 21–33 | ReLU networks, backpropagation, decision-region graphs, validation and hyperparameter search, multiple hidden layers, He initialization, mini-batches, learning-rate decay, momentum, Adam, overfitting, dropout, and L2 regularization |
| Concept review | Day 31 | A written review of network shapes, forward propagation, backpropagation, and Adam |
| Real datasets | Days 34–35 | Loading, normalizing, and classifying local wine datasets with the from-scratch network |

## Repository layout

```text
.
├── d001-d009-np_foundations/       # NumPy and linear-algebra fundamentals
├── d010-d017-ml_foundations/       # Classical machine-learning foundations
├── d018-d020-classification/       # Binary and multiclass classification
├── d021-d033-basic_nns/            # From-scratch neural-network experiments
├── d034-d0---real_datasets/        # Wine-dataset experiments and local CSV data
│   └── datasets/
├── numpy-testing.py                 # Scratch NumPy experiments
└── README.MD
```

## Learning path

| Day | File | Focus |
| ---: | --- | --- |
| 1–9 | `d001-d009-np_foundations/` | Arrays, slicing, broadcasting, statistics, matrices, randomness, simulation, vectors, and cosine similarity. |
| 10–17 | `d010-d017-ml_foundations/` | k-nearest neighbors, regression, loss functions, gradients, and vectorized regression. |
| 18–20 | `d018-d020-classification/` | Logistic regression, sigmoid, softmax, cross-entropy, and multiclass classification. |
| 21–30 | `d021-d033-basic_nns/` | Neural networks, backpropagation, visualization, validation, mini-batches, learning-rate scheduling, momentum, and Adam. |
| 31 | `d021-d033-basic_nns/d31-8-30-9-6.MD` | Written review of network shapes, forward passes, backpropagation, and Adam. |
| 32 | `d021-d033-basic_nns/d32-9-6-7.py` | Train/validation/test splits and experiments with larger networks. |
| 33 | `d021-d033-basic_nns/d33-9-8.py` | Deliberate overfitting experiments, dropout, L2 regularization, and hyperparameter comparison. |
| 34 | `d034-d0---real_datasets/d34-9-23.py` | Classification using the UCI Wine dataset stored locally in `datasets/wine.data`. |
| 35 | `d034-d0---real_datasets/d35-9-23.py` | A harder multiclass experiment using the white-wine quality dataset. |

`numpy-testing.py` contains smaller scratch experiments and helper-function practice.

## Featured experiments: regularization and real data

The later exercises train fully connected classifiers on synthetic student data. The input features are hours studied and hours slept; the model predicts one of five grade bands: `F`, `D`, `C`, `B`, or `A`.

The Day 27 `batched_nn` function accepts a layer layout such as `[2, 8, 8, 5]`, then:

1. initializes every dense layer (with optional He initialization);
2. runs a forward pass using ReLU hidden activations and a softmax output;
3. calculates sparse categorical cross-entropy;
4. propagates gradients backward through each layer; and
5. updates weights and biases one shuffled mini-batch at a time.

It reports training progress, evaluates a held-out test set with a confusion matrix, and plots loss by epoch. Later scripts continue that same from-scratch model:

- Day 28 reduces the learning rate when loss improvement plateaus.
- Day 29 adds velocity-based momentum to weight and bias updates.
- Day 30 explores Adam-style updates using exponentially weighted first and second gradient moments with bias correction.
- Day 31 records a written check-in on how the network’s shapes, forward pass, backpropagation, and optimizer fit together.
- Day 32 introduces separate training, validation, and test partitions for larger-network experiments.
- Day 33 adds inverted dropout and L2 weight regularization while comparing training and validation behavior.
- Days 34–35 apply the network to local wine datasets, replacing synthetic student data with tabular real-world data.

## Requirements

- Python 3.10 or newer is recommended.
- [NumPy](https://numpy.org/) is required by every main experiment.
- [pandas](https://pandas.pydata.org/) is required for Days 34–35 to load CSV-style datasets.
- [Matplotlib](https://matplotlib.org/) is required for the plotting exercises (Days 22, 23, 27, 29, and 30).
- [scikit-learn](https://scikit-learn.org/) is used only in Day 17 for comparison with `LinearRegression`.

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib scikit-learn
```

## Running an experiment

Scripts are independent and should be run from the repository root:

```bash
python3 d001-d009-np_foundations/d1-7-25.py
python3 d010-d017-ml_foundations/d17-8-3.py
python3 d021-d033-basic_nns/d33-9-8.py
python3 d034-d0---real_datasets/d34-9-23.py
```

Some scripts generate different results on each run because they create synthetic data with NumPy randomness. Days 22, 23, 27, 29, and 30 open Matplotlib windows; close the window to let the script finish. Day 18 prompts for desired study and sleep hours after training. Days 34 and 35 set their working directory to their own folder, so their bundled `datasets/` files load whether the scripts are run from the repository root or that folder.

For a good guided tour, run the files in day order. To start with the latest neural-network regularization experiment, begin with `d021-d033-basic_nns/d33-9-8.py`; to see the same approach applied to real data, use `d034-d0---real_datasets/d34-9-23.py`. Day 31 is a Markdown review note and is meant to be read rather than executed.

## Project conventions and scope

- Folder names encode the day range and topic; source filenames encode the individual learning day and date. Day 31 is a Markdown review note rather than an executable script.
- Most scripts run immediately when executed and print their observations, diagnostics, predictions, accuracy, or loss values to the terminal.
- Days 1–33 use datasets generated in code. Days 34–35 use the local wine datasets in `d034-d0---real_datasets/datasets/`; no dataset download is needed to run them.
- Code intentionally favors explicit, commented steps over library abstractions, making the underlying linear algebra and gradient calculations visible.
- This is an educational notebook-style repository, not a reusable ML package. There is no test suite, command-line interface, model persistence, or pinned dependency file yet.
