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

*WARNING*: this will download 10+ gigabytes of code from crawling github and
take days to complete.  It is likely not necessary for you to do this.
#### Download available repos into `./paritybench_download` folder
```
python main.py --download [--limit max-repos]
```
#### Regenerate PyTorch module testing cases in `./generated` folder
```
python main.py --generate-all [--limit max-repos] [--jobs num-worker-threads]
```
