# Contributing to `reddit-memes`

This document contains instructions and general guidelines for contributing to `reddit-memes`. The goal is to make the development process as smooth as possible, while maintaining a high level of code quality, so use your best judgment when deciding the level of detail to include in your contributions.

# Table of Contents

1. [Pre-requisites](#1-pre-requisites)
   1.1. [Installing `uv`](#11-installing-uv)
   1.2. [Installing `make` and `awk`](#12-installing-make-and-awk)
2. [Setup Local Environment](#2-setup-local-environment)
3. [Quality Features](#3-quality-features)
   3.1. [Makefile](#31-makefile)
   3.2. [UV](#32-uv)
   3.3. [Pre-commit](#33-pre-commit)
   3.4. [Ruff/MyPy/Pytest](#34-ruffmypypytest)
4. [General Guidelines](#4-general-guidelines)
   4.1. [Code Style](#41-code-style)
   4.2. [Documentation](#42-documentation)
   4.3. [Branching & Pull Requests](#43-branching--pull-requests)

---

# 1. Pre-requisites

One of the main libraries used for model training in the Computer Vision will be `TensorFlow`. For those who want to train models with GPU support and are on a Windows system, you will need to install WSL2 (Windows Subsystem for Linux) and install `CUDA 12.3` and `cuDNN 8.9.7`. You can follow [this step-by-step guide.](https://pradeepl.com/blog/installing-nvidia-cuda-tensorflow-on-windows-wsl2/)

It is also assumed that you have Git installed and ready to go on your system. If not, you can download it from [Git's official website.](https://git-scm.com/downloads)

## 1.1. Installing `uv`

`uv` is a command-line tool that helps you manage your project's dependencies and environment. It is being used in this project due to its all-in-one solution for managing dependencies, environments and Python versions, as well as for its great speed improvements over traditional package managers like `pip`.

To install `uv`, you can run the following command:

- On Windows (run with elevated privileges):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- On MacOS/WSL/Linux (may need to prefix with `sudo`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Afterwards, you can check if `uv` was installed correctly by running:

```bash
uv --version
```

You can use `uv` outside of the project, as it is a very useful tool for managing python environments and dependencies. Popular commands:

- `uv sync` - Installs all dependencies in the dependencies file.
- `uv run` - Runs a command in the project's environment. For example, `uv run python -m my_module` will run the `my_module` module in the project's environment.
- `uv add <package>` - Adds a package to the dependencies file and installs it.
- `uv remove <package>` - Removes a package from the dependencies file and uninstalls it.
- `uv pip <regular pip syntax>` - Runs pip commands through uv
- `uv lock` - Locks the dependencies file to the current versions of the installed packages.
- `uv upgrade` - Upgrades all packages in the dependencies file to the latest version.
- `uv python install <version>` - Installs a specific version of python.

## 1.2. Installing `make` and `awk`

This step is only necessary for Windows users that are not using WSL2, and Unix-based systems have these tools installed by default. `make` and `awk` are used to run Makefile commands, which are used to automate things like installing the environment and syncing dependencies, testing and formatting code, generating documentation automatically, and more.

On Windows, you can install these programs with a package manager like `chocolatey`. To install `chocolatey`, run on Powershell with elevated privileges:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

And confirm the installation by running (you may need to restart your terminal for the program to be recognized):

```powershell
choco --version
```

Then run also on terminal with elevated privileges:

```powershell
choco install make awk
```

and confirm the installation by running:

```powershell
make --version
awk --version
```

Some MacOS users may need to install `make`, as newer versions do not include it by default. You can install it with `brew`:

```bash
brew install make
```

or with `xcode-select` in the terminal:

```bash
xcode-select --install
```

---

# 2. Setup Local Environment

Now that you have all the necessary tools installed, you can set up the local environment for the project. This will install all the necessary dependencies and set up the environment for development.

- Navigate to the desired directory and clone the repository:

```bash
cd <directory_in_which_repo_should_be_created>
git clone https://github.com/martimesteves1/reddit-memes/invitations
```

- Navigate to the project directory and install the environment and pre-commit hooks:

```bash
cd reddit-memes
make install
```

- If for some reason the previous step didn't work, you can alternatively run:

```bash
uv sync
uv run pre-commit install
uv run python -m ipykernel install --user --name=reddit-memes --display-name="Reddit Memes Kernel"
```

- To activate the environment, run:

```bash
# Windows
.venv\Scripts\activate

# MacOS/WSL/Linux
source .venv/bin/activate
```

Now you are ready to start developing! Develop your code inside the `reddit-memes\reddit_memes` directory, and put any necessary tests in the `reddit-memes\tests` directory. Don't forget to add any extra dependencies with `uv add <package>` and to lock the dependencies file with `uv lock` afterwards.

If you are working with VS Code, I also recommend selecting the newly created environment as the default interpreter. You can do this by pressing `Ctrl+Shift+P` and typing `Python: Select Interpreter`, then selecting the `reddit-memes` environment in the .venv directory. This way, everytime you open the integrated terminal, it will automatically activate the environment.

---

# 3. Quality Features

## 3.1. Makefile

A Makefile was created to automate the most common tasks in the project. You can run the following commands:

```bash
make install - Installs the virtual environment and pre-commit hooks.
make check - Runs code quality tools (Ruff, Mypy, Pytest and Prettier).
make test - Runs the tests with pytest.
make docs-test - Tests if the documentation can be built without warnings or errors.
make docs - Builds and serves the documentation (use it to check if docstrings are correct).
make help - Shows all available commands.
```

Keep in mind that in order to run these commands, you **must be in the project's root directory**, where the Makefile is located.

The `make check` command often marks files for merging just by virtue of running `Ruff`, even if you didn't make any changes to the files. For those cases, you can either commit everything that was marked, or:
1 - Run `git checkout -- <file>` to discard the changes.
2- Add only the files you want to commit with `git add <file>`, commit them and then stash the rest with `git stash`. Then, just remove the stash with `git stash drop`.

## 3.2. UV

For this project `uv` automatically syncs the dependencies file everytime a commit is done, so make sure that you don't install packages with `uv pip install <package>` or `pip install <package>`. Instead, use `uv add <package>` and `uv remove <package>` to manage dependencies. After either adding or removing, update the dependencies file with `uv lock`.

With uv, you can also run commands in the project's environment with `uv run <command>`, which can be useful if you want to run a specific check individually, e.g. `uv run ruff check <file>`.

## 3.3. Pre-commit

Pre-commit is a tool that automatically checks your code for issues before you commit it. It is configured to run Ruff, Mypy, Prettier and Pytest before every commit, although it is recommended that you run `make check` and `make test` before committing to avoid any issues.

One of the features of pre-commit is that it blocks the commit if it finds any issues, which can be useful if you forgot to run the checks before committing, but can also be annoying if there are certain fixes that are being forced that are unnecessary. If you think that your code is correct, you can skip the pre-commit checks by adding the `--no-verify` flag to the commit command.

```bash
git commit -m "Your message" --no-verify
```

### 3.4. Ruff/MyPy/Pytest

Note that you can run these programs with `make check` and `make test`, but for each individual tool the `uv run` command is indicated.

#### Ruff

Ruff is a code formatter that enforces a consistent code style. Here are the most common commands:

```bash
uv run ruff check <file> - Checks the code for formatting issues.
uv run ruff format <file> - Formats the code according to the style guide.
uv run ruff rule <rule / --all> - Shows the documentation for a specific rule or all rules.
```

Check the [docs](https://docs.astral.sh/ruff/rules/#legend) for more readable list of rules.

#### Mypy

Mypy checks for type hints in the code. Type hints are helpful for understanding the code and for catching bugs early. To run mypy, use the following command:

```bash
uv run mypy <file/directory>
```

Examples of functions with type hints:

```python
# Basic Example
def add(a: int, b: int) -> int:
   return a + b

# When a function can accept None or a specific type
from typing import Optional
def greet(name: Optional[str] = None) -> str:
   return(f"Hello, {name or 'World'}")

# Type hinting with 3rd party libraries
import pandas as pd
import numpy as np
from numpy.typing import NDArray
def process_array(arr: NDArray[np.float64]) -> pd.DataFrame:
   return pd.DataFrame(arr)

# When a variable can be multiple types
from typing import Union
def square_or_cube(num: Union[int, float]) -> Union[int, float]:
   return num ** 2 if isinstance(num, int) else num ** 3

# Nested type hints
def process_data(data: dict[str, list[tuple[int, float]]]) -> int:
   return len(data)
```

#### Pytest

I recommend using pytest only when you think your code is going to be either altered or is commonly used in the project.

For example, a module that downloads images based on a link column in a pandas dataframe should have tests for things like identifying valid and invalid links, downloading the images in the correct format, saving them to the correct directory, etc.;

while a module used to train a specific computer vision model is not going to be used in other parts of the project, so it is not necessary to write tests for it.

To run pytest, use the following command after placing your tests in the `tests` directory:

```bash
make test
```

or alternatively:

```bash
uv run pytest
```

If you include doctests in your code, you can run them with the following command:

```bash
uv run pytest --doctest-modules
```

Example of a doctest in the docstring:

```python
def add(a: int, b: int) -> int:
   """
   Adds two numbers together.

   >>> add(1, 2)
   3
   >>> add(0, 0)
   0
   """
   return a + b
```

---

# 4. General Guidelines

## 4.1. Code Style

For the code style, apply the same judgment as you would with the other code quality tools of the project. It is nice to have readable code, but don't be too worried about if it slows you down. The most important thing is to have correct code, and the code style can be fixed later.

Some quick tips for code style, based on what Ruff and Mypy check for, as well as other common practices:

- Use type hints whenever possible (check the [mypy section](#mypy) for examples).
  <br>
- Follow [PEP8](https://peps.python.org/pep-0008/) guidelines:
  - Use 4 spaces for indentation.
  - Use `snake_case` for variable, function and method names, `CamelCase` for class names and `UPPER_CASE` for constants.
  - Use `"""triple quotes"""` for docstrings.
  - Limit lines to 79 characters, and 72 characters for docstrings.
  - Use spaces around operators and after commas (e.g. `a = 1 + 2`, `func(a, b)`).
    <br>
- Limit function complexity - if the task is too complex, break it down to smaller functions.
  <br>
- Use descriptive variable/function/class names - the name of the variable should describe what it is used for.
  <br>
- Include docstrings for functions, classes, methods and modules. More on this on the [documentation section](#42-documentation).
  <br>
- Order imports in the following way:
  1.  Standard library imports.
  2.  Related third party imports.
  3.  Local application/library specific imports.
  - Imports from the same package should be grouped together, e.g.:
    ```python
    import os
    import numpy as np
    from reddit_memes import utils
    from reddit_memes.utils import download_image
    ```
  - Separate each group with a blank line.
  - Avoid using `from module import *`.
    <br>
- Use `f-strings` for string formatting, as they are more readable and faster than other methods.

```python
name = "Alice"
age = 25
print(f"Hello, my name is {name} and I am {age} years old.")
```

- Use list comprehensions and generator expressions instead of `for` loops when possible.

```python
squares = [x ** 2 for x in range(10)]
```

- Use exceptions when possible instead of returning error codes.

```python
def read_data(filepath: str) -> pd.DataFrame:
   try:
      return pd.read_csv(filepath)
   except FileNotFoundError:
      raise FileNotFoundError(f"File {filepath} not found.")
```

## 4.2. Documentation

For writing documentation, you can use [Google](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), [NumPy](https://numpydoc.readthedocs.io/en/latest/example.html) or [Sphinx](https://documentation-style-guide-sphinx.readthedocs.io/en/latest/style-guide.html) style docstrings, but my recommendation is using [NumPy](https://numpydoc.readthedocs.io/en/latest/example.html) style docstrings.

NumPy docstring sections order (not all components need to be present in every context):

1. [Short Summary](https://numpydoc.readthedocs.io/en/latest/format.html#short-summary)
2. [Deprecation Warning](https://numpydoc.readthedocs.io/en/latest/format.html#deprecation-warning)
3. [Extended Summary](https://numpydoc.readthedocs.io/en/latest/format.html#extended-summary)
4. [Parameters](https://numpydoc.readthedocs.io/en/latest/format.html#parameters)
5. [Returns](https://numpydoc.readthedocs.io/en/latest/format.html#returns)
6. [Yields](https://numpydoc.readthedocs.io/en/latest/format.html#yields)
7. [Receives](https://numpydoc.readthedocs.io/en/latest/format.html#receives)
8. [Other Parameters](https://numpydoc.readthedocs.io/en/latest/format.html#other-parameters)
9. [Raises](https://numpydoc.readthedocs.io/en/latest/format.html#raises)
10. [Warns](https://numpydoc.readthedocs.io/en/latest/format.html#warns)
11. [Warnings](https://numpydoc.readthedocs.io/en/latest/format.html#warnings)
12. [See Also](https://numpydoc.readthedocs.io/en/latest/format.html#see-also)
13. [Notes](https://numpydoc.readthedocs.io/en/latest/format.html#notes)
14. [References](https://numpydoc.readthedocs.io/en/latest/format.html#references)
15. [Examples](https://numpydoc.readthedocs.io/en/latest/format.html#examples)

Examples:

- **For Functions**:

```python
def load_data(filepath: str) -> list:
    """
    Loads Reddit meme data from a file.

    This function reads a JSON file containing meme data and returns
    the corresponding objects for further analysis.

    Parameters
    ----------
    filepath : str
        The path to the file containing meme data.

    Returns
    -------
    list of Meme
        A list of Meme objects extracted from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains invalid data.

   Examples
   --------
   >>> load_data("data/memes.json")
   [Meme(...), Meme(...), ...]
    """
    with open(filepath, 'r') as file:
        data = json.load(file)

    memes = [Meme(meme_data) for meme_data in data]
    return memes
```

- **For Classes and Methods**:

```python
class MemeNetwork:
    """
    A class used to represent a network of Reddit memes.

    This class allows you to load, manipulate, and analyze the
    relationships between memes on Reddit, such as their interactions
    and similarities.

    Attributes
    ----------
    memes : list of Meme
        A list of Meme objects representing the nodes in the network.
    edges : list of tuple
        A list of tuples representing connections between memes.

    Methods
    -------
    add_meme(meme):
        Adds a meme node to the network.
    add_edge(meme1, meme2):
        Adds a connection (edge) between two memes.
    analyze():
        Analyzes the network and returns basic network statistics.

   Examples
    --------
    Create a new MemeNetwork, add some memes, and analyze the network:

    >>> network = MemeNetwork()
    >>> meme1 = Meme("Meme1")
    >>> meme2 = Meme("Meme2")
    >>> network.add_meme(meme1)
    >>> network.add_meme(meme2)
    >>> network.add_edge(meme1, meme2)
    >>> stats = network.analyze()
    >>> print(stats)
    {'num_memes': 2, 'num_edges': 1, 'density': 1.0}
    """
    def __init__(self, memes=None):
        """
        Parameters
        ----------
        memes : list of Meme, optional
            A list of Meme objects to initialize the network with (default is None).
        """
        self.memes = memes if memes is not None else []
        self.edges = []

    def add_meme(self, meme):
        """
        Adds a meme node to the network.

        Parameters
        ----------
        meme : Meme
            A Meme object representing the meme to be added.
        """
        self.memes.append(meme)

    def add_edge(self, meme1, meme2):
        """
        Adds a connection (edge) between two memes in the network.

        Parameters
        ----------
        meme1 : Meme
            The first meme object.
        meme2 : Meme
            The second meme object.
        """
        self.edges.append((meme1, meme2))

    def analyze(self):
        """
        Analyzes the network and returns basic network statistics.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - 'num_memes' : int
                The number of meme nodes in the network.
            - 'num_edges' : int
                The number of edges (connections) between memes.
            - 'density' : float
                The density of the network (ratio of actual connections to possible connections).
        """
        num_memes = len(self.memes)
        num_edges = len(self.edges)
        density = num_edges / (num_memes * (num_memes - 1) / 2) if num_memes > 1 else 0
        return {"num_memes": num_memes, "num_edges": num_edges, "density": density}
```

- **For Modules**:

```python
"""
reddit_memes.network_analysis

This module provides functionality for analyzing the network structure
of memes from Reddit. It includes classes and functions to extract,
analyze, and visualize meme relationships.

Functions:
----------
    load_data: Loads Reddit meme data from a file.
    analyze_network: Analyzes the meme network and returns metrics.

Classes:
--------
    MemeNetwork: Represents a network of memes.
"""
```

You can also add an examples section if the functions/classes/methods don't have examples in their docstrings.

```python
"""
Examples
--------
Load a Reddit meme dataset, create a meme network, and analyze it:

>>> from reddit_memes.network_analysis import load_data, MemeNetwork
>>> memes = load_data("memes_data.json")
>>> network = MemeNetwork(memes)
>>> network.analyze()
{'num_memes': 10, 'num_edges': 15, 'density': 0.33}
"""
```

## 4.3. Branching & Pull Requests

In order to organize the development process, it is recommended that you create a branch for local development every time you want to start on a specific feature or fix. You should name branches based on the work you are doing, e.g. `data/download_memes`, `data/jq-processing`, etc.

```bash
git checkout -b data/download_memes
```

Then once the work is done, you run `make check` and `make test` to check if everything is correct. If everything is correct, you can commit your changes and push them to the repository. You can also make multiple smaller commits for a complicated feature, as it can be better for reviewing, debugging or reverting changes if needed.

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin data/download_memes
```

After that, you can create a pull request on the GitHub website.
