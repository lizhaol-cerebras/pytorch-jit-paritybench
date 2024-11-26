A test suite to run PyTorch tests on many `nn.Module`s crawled from popular GitHub projects.

Each test case is self contained and does not require relative dependencies.


###  Running ParityBench

- [Install conda] with python>=3.9
and create/activate a [conda environment]

- Install requirements:
```
conda install pip
pip install -r requirements.txt
```

A file `errors.csv` is generated containing the top error messages and example
`generated/*` files to reproduce those errors.

[Install conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
[conda environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


### Regenerate ParityBench

#### Download available repos. By default, downloads to `./paritybench_download` folder.
```
python main.py --download [--download-dir download-dir] [--limit max-repos]
```
#### Regenerate PyTorch module test cases. By default, loads from `./paritybench_download` and outputs to `./generated` folder.
```
python main.py --generate-all [--download-dir download-dir] [--tests-dir outputs-dir] [--limit max-repos] [--jobs num-worker-threads]
```
