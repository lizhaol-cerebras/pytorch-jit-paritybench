A test suite to measure TorchScript parity with PyTorch on many `nn.Module`s
crawled from popular GitHub projects.


###  Running ParityBench

- [Install conda] with python>=3.9
and create/activate a [conda environment]

- Install requirements:
```
conda install pip
pip install -r requirements.txt
conda install pytorch torchvision cpuonly -c pytorch-nightly
```

```
A file `errors.csv` is generated containing the top error messages and example
`generated/*` files to reproduce those errors.

[Install conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
[conda environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


### Regenerate ParityBench

*WARNING*: this will download 10+ gigabytes of code from crawling github and
take days to complete.  It is likely not necessary for you to do this.
```
python main.py --download
python main.py --generate-all
```

### Download, generate, evaluate 
You can limit number of github projects to download for testing and running on a smaller set of github repos
```
python main.py --download [--download-dir <folder path>] [--limit 10]
```
Generate multiple tests from downloaded projects
```
python main.py --generate-all [--download-dir <folder path>] [--limit 10]
```
You can generate tests for one project folder `-g`. This will extract nn modules from that project and generate a test script `--tests-dir`
```
python main.py -g <folder path> --tests-dir <folder path>
```
You can evaluate one generated test script `-e` and try export the module to onnx `--onnxdir` 
```
python main.py -e <test.py file> --onnxdir <folder path>
```
You can evaluate using different compile mode, e.g, `dynamo`(default) or `torchscript`.
```
python main.py -e <test.py file> --compile_mode dynamo
```
You can evaluate using different dynamo backends provided in `torch._dynamo`, please refer `torch._dynamo.list_backends()`.
```
python main.py -e <test.py file> --backend eager
```
You can evaluate using `cuda`(default) or `cpu`.
```
python main.py -e <test.py file> --device cuda
```




