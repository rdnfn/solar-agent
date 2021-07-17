# Solar Agent

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

This project provides Gym environments to train RL agents to operate batteries in solar-plus-battery installations. The setup of how an agent is able to interact with the environment is shown in the Figure below.

![environment setup illustration](https://github.com/rdnfn/solar-agent/blob/main/docs/img/img_003_environment_setup_v2.png)


## Installation

To get started, first clone the repo:

```
git clone https://github.com/rdnfn/solar-agent.git
```

Once the repo is cloned, an environment with all the relevant requirements can be created using:

```
make env
```

Alternatively, you can install the Solar Agent package (`solara`) in an existing environment using the following command in the project directory:

```
pip install -e .
```

## Reproducing Report Experiments

This implementation was created as part of an AI4ER MRes Project. To reproduce the plots in the corresponding report follow the instructions above to locally install the package. Then the plots can be created by running the code in all notebooks in the `notebooks/` subdirectory sequentially (excluding the additional exploratory notebooks in `notebooks/exploratory/`). Note that depending on the available compute running all experiments may take from a few hours to possibly weeks.

---


This project was originally initialised using the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter) template.
