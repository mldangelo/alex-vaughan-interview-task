# Environment Setup [Optional]

### Pipenv

If you don't have Pipenv installed you can install it on Mac with:

```bash
brew install pipenv
```

If you're not familiar with Pipenv, you can learn about it [here](https://pipenv.readthedocs.io/en/latest/).

To start, in your project folder run:

```bash
pipenv install
```

This will create a Pipfile and Pipfile.lock. Please add any python packages you use to this Pipfile. Note that you may wish to specify a python version with the `--python` argument (ex. `pipenv --python 3.6.5`). We recommend using [pyenv](https://github.com/pyenv/pyenv) for python version management.

(Note if you get an error on install due to pip version 18.1 incompatibility with Pipenv, you can pin your pip version to 18.0 with `pip install pip==18.0`)

To add packages to the Pipfile run:

```bash
pipenv install <PACKAGE-NAME>
```

Next run:

```bash
pipenv shell
```

This will bring up a terminal in your virtualenv like this:

```bash
(my-virtualenv-name) bash-4.4$
```

### Jupyter Notebook

Run:

```bash
pipenv install jupyter
```

To create a Jupyter Notebook kernel for this virtual env, in the shell run:

```bash
python -m ipykernel install --user --name=`basename $VIRTUAL_ENV` --display-name "My Virtualenv Name"
```
("My Virtualenv Name" can be anything you want)

Launch jupyter notebook within the shell:

```bash
jupyter notebook
```

When running an existing notebook, make sure the kernel "My Virtualenv Name" is selected (if not, you should be able to select it as an option in Kernel -> Change Kernel).

When creating a new notebook, make sure to select "My Virtualenv Name" as the default kernel.

In the future when opening the project, you should only have to run:

```bash
pipenv shell
jupyter notebook
```
