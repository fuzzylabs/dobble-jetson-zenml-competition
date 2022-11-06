# The Dobblegängers Month of MLOps Submission

This repository contains The Dobblegängers (a.k.a., Fuzzy Labs) submission to [ZenML Month of MLOps competition](https://zenml.notion.site/ZenML-s-Month-of-MLOps-Competition-Announcement-3c59f628447c48f1944035de85ff1a5f)

## Contents

- [The Dobblegängers](#the-dobblegängers)
- [What have we done?](#what-have-we-done)
- [Code & Repository Structure](#code--repository-structure)
- [Project Overview](#project-overview)
- [Setup](#setup)
- [Running the Pipelines](#running-the-pipelines)
- [Blog Posts & Demo](#blog-posts--demo)
 

## The Dobblegängers

1. [Misha Iakovlev](https://github.com/d-lowl)
2. [Shubham Gandhi](https://github.com/dudeperf3ct)
3. [Oscar Wong](https://github.com/osw282)
4. [Christopher Norman](https://github.com/Christopher-Norman)
5. [Jon Carlton](https://github.com/JonoCX)

## What have we done?

At [Fuzzy Labs](https://www.fuzzylabs.ai/) we're trying to become Dobble world champions. So, we came up with a plan - we've trained an ML model to recognise the common symbol between two cards, and what better way to make it than with a ZenML pipeline.

If you're reading this and wondering: what on earth is [Dobble](https://www.dobblegame.com/en/games/)? Let us explain. It's a game of speed and observation where the aim is to be the quickest to identify the common symbol between two cards. If you're the first to find it and name it, then you win the card. Simple, right? It essence, it's a more sophisticated version of snap.

Now that you're all caught up, let's go into a little more detail about what we've done. Obviously as we're wanting to win the world championships, we need a concealable device. So, to also provide an extra challenge, we decided to deploy our model to a [NVIDIA Jetson Nano](https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-nano/education-projects/).

## Code & Repository Structure

This repository contains all the code and resources to set up and run a data pipeline, training pipeline, and inference on the Jetson. It's structured as follows:

```bash
.
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt                        # dependencies required for the project
├── docs                                    # detailed documentation for the project
├── pipelines                               # all pipelines inside this folder
│   └── training_pipeline
        └── training_pipeline.py
        └── config_training_pipeline.yaml   # each pipeline will have one config file containing information regarding step and other configuration
├── run.py                                  # main file where all pipelines can be run
└── steps                                   # all steps inside this folder
    └── data_preprocess                     # each step is in its own folder (as per ZenML best practises)
        └── data_preprocess_step.py
    └── src                                 # extra utilities that are required by steps added in this folder
└── zenml_stack_recipes                     # contains the modified aws-minimal stack recipe

```

As we've also used some cloud resources to store data and host experiment tracking, we used one of the ZenML stack recipes. There's more information on this [here](docs/stack_recipe_readme.md).

## Project Overview

To give an overview of our solution (see [here](docs/pipelines_overview.md) for an in-depth description), we've broken this challenge down into three stages, with two pipelines:

### [Data Pipeline](docs/pipelines_overview.md#data-pipeline)

This downloads the labelled data, processes it into the correct format for training, and uploads to an S3 bucket.

### [Training Pipeline](docs/pipelines_overview.md#training-pipeline)

This pipeline downloads the data, validates the data, trains and evaluates a model, and exports to the correct format for deployment.

### Deployment Stage

Here, the trained model is loaded onto the device and inference is performed in real-time

## Setup

The first step is creating a virtual environment and install the project requirements, we've used `conda` but feel free to use whatever you prefer (as long as you can install a set of requirements):

```bash
conda create -n dobble_venv python=3.8 -y
conda activate dobble_venv
pip install -r requirements.txt
```

The next step is to setup ZenML, with the first step being to install the required integrations:

```bash
zenml integrations install -y pytorch mlflow
```

Initialise the ZenML repository

```bash
zenml init
```

Start the ZenServer

```bash
zenml up
```

> **Note**
> Visit  ZenML dashboard is available at 'http://127.0.0.1:8237'. You can connect to it using the 'default' username and an empty password.
> If there's a TCP error about port not being available. Run `fuser -k port_no/tcp` to close an open port and run `zenml up` command again, for MacOS, run `kill $(lsof -t -i:8237)`.

By default, ZenML comes with a stack which runs locally. Next, we add MLflow as an experiment tracker to this local stack, which is we'll run the pipelines:

```bash
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml stack register fuzzy_stack \
    -a default \
    -o default \
    -e mlflow_tracker \
    --set
```

You're now in a position where you can run the pipelines locally.

## Running the Pipelines

We have a couple of options for running the pipelines, specified by flags:

```bash
python run.py -dp       # run the data pipeline only
python run.py -tp       # run the training pipeline only
python run.py -dp -tp   # run both the data and training pipelines
```

## Setup using the Stack Recipe

Please see [here](docs/stack_recipe_readme.md) for a detailed guide on what we've modified in the `aws-minimal` stack recipe and how to run it

## Blog Posts & Demo

As part of our submission, we've written a series of blogs on our website. Each of the blogs has an accompanying video.

### Introduction

https://www.youtube.com/watch?v=j9TAVpM5NRQ

### About the Edge

https://www.youtube.com/watch?v=djliB4QnuoQ

### The Data Science

Video: https://www.youtube.com/watch?v=gCAzpyE0Zr8
Blog: https://www.fuzzylabs.ai/blog-post/zenmls-month-of-mlops-data-science-edition

### Pipelines on the Edge

Blog: https://www.fuzzylabs.ai/blog-post/mlops-pipeline-on-the-edge 
