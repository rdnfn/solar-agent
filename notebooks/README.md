# Notebooks
## Structure
Exploratory notebooks for initial explorations go into the `notebooks/exploratory` folder.
Polished work for demonstration purposes goes directly into this `notebooks/` folder.

## Naming convention
We use the following naming convention for notebooks (inspired by [cookiecutter-datascience](https://drivendata.github.io/cookiecutter-data-science/#notebooks-are-for-exploration-and-communication))
```<initials>_<step>_<description>.ipynb```

For example, for `Tom Baker Adams` a valid name would be `tba_1_data-analysis,ipynb`.

## Useful initialization cell
To avoid having to reload the notebook when you change code from underlying imports, we recommend the following handy initialization cell for jupyter notebooks:
```
%load_ext autoreload             # loads the autoreload package into ipython kernel
%autoreload 2                    # sets autoreload mode to automatically reload modules when they change
%config IPCompleter.greedy=True  # enables tab completion
```