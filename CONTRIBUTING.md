# Contributing

To submit a pull request, please do the following:

1. Fork the [repository](https://github.com/JohnGiorgi/DeCLUTR) by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/DeCLUTR.git
   $ cd DeCLUTR
   $ git remote add upstream https://github.com/JohnGiorgi/DeCLUTR.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   __do not__ work on the `master` branch.

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   $ pip install -e ".[dev]"
   ```

   (If the repository was already installed in the virtual environment, remove it with `pip uninstall` before reinstalling it in editable mode with the `-e` flag.)

5. Develop the features on your branch.

   This repository relies on `black` to format its source code
   consistently. After you make changes, format them with:

   ```bash
   $ black declutr
   ```

   This repository also uses `flake8` to check for coding mistakes. To run the checks locally:

   ```bash
   $ flake8 declutr
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied, go to the webpage of your fork on GitHub.
   Click on 'Pull request' to send your changes to the project maintainers for review.

> This is a work in progress. Inspiration for these guidelines were drawn from [here](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md) and [here](https://github.com/nayafia/contributing-template).