#!/usr/bin/env python3
import ast
import inspect
import logging
import os
import re
import subprocess
import sys
import tempfile
import types
import zipfile
from functools import partial
from typing import List, Optional, TextIO

import astor
import torch
from torch.nn.parallel import DistributedDataParallel

from .deduce_parameters import DeduceParameters, DeduceParameter
from .reporting import Stats, ErrorAggregatorDict
from .static_analysis import ASTCleanup
from .static_analysis import CONFIG_NAMES
from .static_analysis import CheckCallableMembers
from .static_analysis import ExtractConfigUsage
from .static_analysis import ExtractReadsWrites
from .static_analysis import IMPORT_WHITELIST
from .static_analysis import split_import
from .utils import call_with_timeout

log = logging.getLogger(__name__)

NN_MODULE_RE = re.compile(r"(\btorch\b)|(\bnn[.]Module\b)", re.MULTILINE)
PREFIX = f"""
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import {', '.join(sorted(IMPORT_WHITELIST))}
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor

__version__ = "1.0.0"
xrange = range
wraps = functools.wraps
"""
SUFFIX = """
import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace

"""


def to_source(node):
    return astor.to_source(
        node,
        pretty_source="".join,
        pretty_string=partial(astor.string_repr.pretty_string, max_line=8192),
    )


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(
        self,
        tempdir: str,
        errors: ErrorAggregatorDict,
        stats: Stats,
        output_py: TextIO,
        args,
    ):
        super(PyTorchModuleExtractor, self).__init__()
        self.errors = errors
        self.stats = stats

        self.output = IncrementalModule(tempdir, output_py)

        self.imports = dict()
        self.constants = []
        # list of class names to output to the generated file
        self.output_class_names = []

        self.available_symbols = dict()
        self.global_config = None

        self.testcases = []
        self.args = args

    def search_file(self, filename: str, open_fn=open):
        """get module from filename .py file"""
        if not filename.endswith(".py") or ".#" in filename:
            return

        with open_fn(filename, "r") as fp:
            source = fp.read()
            if isinstance(source, bytes):
                source = source.decode("utf-8")

        has_match = bool(NN_MODULE_RE.search(source))  # there is torch in .py

        try:
            tree = self.ast_parse(source, filename)
        except Exception as e:
            return self.errors.record("parse", e)

        self.search_ast(tree, has_match)

    @staticmethod
    def ast_parse(source, filename):
        try:
            return ast.parse(source, filename)  # get ast nodes from code
        except SyntaxError:
            # perhaps python2?
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".py") as tmp:
                tmp.write(
                    re.sub(r"\basync *=", "non_blocking=", source)
                    .replace("\t", "    ")
                    .encode("utf-8")
                )
                tmp.flush()
                with open("/dev/null", "w") as null:
                    subprocess.check_call(
                        ["2to3", "-w", tmp.name], stderr=null, stdout=null
                    )
                return ast.parse(open(tmp.name).read(), filename)

    def search_ast(self, tree: ast.AST, overwrite: bool):
        """get torch classes, import and functions"""
        scope = types.ModuleType("_scope")
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self.add_available_symbol(node, overwrite)

                if overwrite and self.should_output_class(scope, node):
                    self.output_class_names.append(node.name)
                    continue

                bases = [to_source(base).strip() for base in node.bases]
                if overwrite and any(
                    self.should_output_class(scope, node, base=base) for base in bases
                ):
                    self.output_class_names.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if overwrite:
                    for module_name, import_node in split_import(node):
                        if module_name == "torch":
                            # Run torch imports so we can run issubclass(.., torch.nn.Module)
                            try:
                                exec(
                                    compile(
                                        ast.Module([import_node], []),
                                        "<string>",
                                        "exec",
                                    ),
                                    scope.__dict__,
                                    scope.__dict__,
                                )
                            except Exception:
                                log.exception("Bad torch import")
                                continue
                        if module_name in IMPORT_WHITELIST:
                            self.imports[to_source(import_node)] = import_node

            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                ),
            ):
                self.add_available_symbol(node, overwrite)

    def should_output_class(
        self, scope: types.ModuleType, node: ast.ClassDef, base: Optional[str] = None
    ):
        """Check if a class should be kept in the output (nn.Modules and dataclasses)"""
        # We want to keep dataclasses
        for decorator in node.decorator_list:
            # Check for @dataclass
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                return True
            # Check for @dataclasses.dataclass
            elif isinstance(decorator, ast.Attribute) and decorator.attr == "dataclass":
                if (
                    isinstance(decorator.value, ast.Name)
                    and decorator.value.id == "dataclasses"
                ):
                    return True

        if not base:
            return False

        # Checks for torch.nn.Module
        if base in ("torch.nn.Module", "nn.Module", "Module"):
            return True
        if base.split(".")[-1] in self.output_class_names:
            return True
        try:
            for part in base.split("."):
                scope = getattr(scope, part, object)
            return issubclass(scope, torch.nn.Module)
        except Exception:
            log.exception("Error in should_keep_class()")

    def search_directory(self, filename: str):
        for root, _, files in os.walk(filename, topdown=False):
            for name in files:
                self.search_file(os.path.join(root, name))

    def search_zipfile(self, filename: str):
        with zipfile.ZipFile(filename) as archive:
            for name in sorted(archive.namelist()):
                self.search_file(name, archive.open)

    def add_available_symbol(self, node, overwrite=False):
        node = ast.fix_missing_locations(ASTCleanup().visit(node))  # clean ast
        try:
            if overwrite:
                self.available_symbols[node.name] = node
            else:
                self.available_symbols.setdefault(node.name, node)
        except AttributeError:  # node.name is missing
            reads, writes = ExtractReadsWrites.run(node)
            for name in writes:
                if overwrite:
                    self.available_symbols[name] = node
                else:
                    self.available_symbols.setdefault(name, node)

    def construct_module(self):
        self.output.run_statement(
            self.ast_parse(PREFIX, "<string>"), source_required=True
        )
        self.name_to_ast = dict()

        for statement in self.imports.values():
            try:
                self.output.run_statement(statement)
            except Exception as e:
                self.errors.record("import", e, "")
        for statement in self.constants:
            try:
                self.output.run_statement(statement)
            except Exception as e:
                self.errors.record("constant", e, getattr(statement, "name", ""))
        for name in self.output_class_names:
            statement = self.available_symbols.get(name)
            if statement:
                self.add_requirements(statement)  # add what is needed for module
                try:
                    self.run_statement(statement)
                    self.available_symbols.pop(name)
                except Exception as e:
                    self.errors.record("define", e, getattr(statement, "name", ""))

    def run_statement(self, statement):
        self.output.run_statement(statement, source_required=True)
        name = getattr(statement, "name", None)
        if name:
            self.name_to_ast[name] = statement

    def add_requirements(self, statement):
        """
        Recursively add symbols to the output module needed by statement.

        :param statement: ast.Node to add to the module
        """
        reads, writes = ExtractReadsWrites.run(statement)
        needs = {sym for sym in reads - writes if sym not in self.output}
        log.debug(
            f"add_requirements: {getattr(statement, 'name', '')} "
            f"available {needs & set(self.available_symbols.keys())} "
            f"unavailable {needs - set(self.available_symbols.keys())}"
        )

        need_config = False
        for name in sorted(needs):
            if name in self.available_symbols and name not in self.output:
                requirement = self.available_symbols.get(name)
                self.add_requirements(requirement)
                try:
                    self.run_statement(requirement)
                    self.available_symbols.pop(name)
                except:
                    log.warning("Error adding requirement", exc_info=True)
            elif name in CONFIG_NAMES:
                need_config = True

    def test_modules(self):
        for name, value in list(sorted(self.output.items())):
            if self.should_test_cls(value):
                self.test_nn_module(name, value)

    def should_test_cls(self, cls):
        if not isinstance(cls, type):  # check if is class
            return False
        if not issubclass(cls, torch.nn.Module):  # check if is torch module
            return False
        if issubclass(cls, DistributedDataParallel):
            return False
        return self.output.same_module(cls)

    def test_nn_module(self, name: str, nn_cls: type):
        if self.args.filter and self.args.filter not in name:
            return

        self.stats["total"] += 1
        checker = CheckCallableMembers.run(
            self.name_to_ast.get(name)
        )  # get modules inside module

        try:
            stats, errors, testcases = call_with_timeout(
                extract_nn_module,
                args=(name, nn_cls, checker, self.errors.context),
                timeout=300,
            )
            self.errors.update(errors)
            self.stats.update(stats)
            self.testcases.extend(testcases)
        except OSError as os:
            log.exception("test_nn_module OS error: {}".format(os))
            self.stats["module_crash"] += 1
        except TimeoutError:
            self.stats["module_timeout"] += 1

    def main(self, filename: str):
        basename = re.sub(r"[.]zip$", "", os.path.basename(filename))

        if os.path.isdir(filename):
            self.search_directory(filename)  # find nn modules, imports, symbols
        else:
            self.search_zipfile(filename)

        self.construct_module()  # run and write ast nodes
        if self.args.args_deducer == "rule-based":
            self.test_modules()
            self.write_testcases(basename)

        log.info(f"{basename}: {self.stats}")

    def write_testcases(self, basename):
        if not self.testcases:
            return
        self.output.write(SUFFIX.format(basename=basename))
        self.output.write("\nTESTCASES = [\n")
        self.output.write("    # (nn.Module, init_args, forward_args)\n")
        for name, init_args, forward_args, compiles in self.testcases:
            self.output.write(f"    ({name},\n")
            self.output.write(f"     lambda: ({init_args[0]}, {init_args[1]}),\n")
            self.output.write(
                f"     lambda: ({forward_args[0]}, {forward_args[1]})),\n"
            )
        self.output.write("]\n\n")


def extract_nn_module(name: str, nn_cls: type, checker, context):
    errors = ErrorAggregatorDict(context)
    stats = Stats()
    testcases = []
    extract_nn_module_inner(name, nn_cls, checker, stats, errors, testcases)
    return stats, errors, testcases


def extract_nn_module_inner(name: str, nn_cls: type, checker, stats, errors, testcases):
    """
    name: name of the module
    nn_cls: module class type
    checker: modules inside nn_cls module
    mode: what to test module with: ts, onnx, etc
    """
    init_signature = inspect.signature(nn_cls)  # get args for init of module
    try:
        init_deducer = DeduceParameters(
            nn_cls,
            *DeduceParameters.initial_args_init(init_signature),
            checker=checker.check,
        )
        init_deducer.search()
        nn_module = init_deducer.last_result
    except Exception as e:
        return errors.record("init", e, nn_cls)

    try:
        nn_module.eval()
    except:
        pass

    stats["init_ok"] += 1

    forward_signature = inspect.signature(
        nn_module.forward
    )  # get args for forward of module
    try:
        forward_deducer = DeduceParameters(
            nn_module, *DeduceParameters.initial_args_forward(forward_signature)
        )
        forward_deducer.search()
    except Exception as e:
        return errors.record("deduce", e, nn_cls)

    stats["deduced_args_ok"] += 1

    try:
        torch.jit.script(nn_module)

    except Exception as e:
        testcases.append(
            (name, init_deducer.testcase_args(), forward_deducer.testcase_args(), False)
        )

        return errors.record("compile", e, nn_cls)

    stats["jit_compiles"] += 1

    testcases.append(
        (name, init_deducer.testcase_args(), forward_deducer.testcase_args(), True)
    )


class IncrementalModule(object):
    """
    Construct a python module statement by statement, recording the result
    to a generated python file.
    """

    def __init__(self, tempdir: str, output_py: TextIO):
        super().__init__()
        self.tempdir = tempdir
        module_name = f"{__name__}.output"
        self.output_module = types.ModuleType(module_name)
        sys.modules[module_name] = self.output_module
        self.output_py = output_py

    def __contains__(self, name):
        """
        :param name: symbol to check for
        :return: True if output module contains name (and it is not an alias)
        """
        return (
            getattr(self.output_module, name, self.output_module)
            is not self.output_module
        )

    def __del__(self):
        sys.modules.pop(self.output_module.__name__, None)

    def items(self):
        return self.output_module.__dict__.items()

    def same_module(self, obj):
        """
        :param obj: a python object
        :return: True if obj is defined in this module
        """
        return obj.__module__ == self.output_module.__name__

    def write(self, data: str):
        self.output_py.write(data)

    def writelines(self, data: List[str]):
        self.output_py.writelines(data)

    def run_statement(self, statement, source_required=False):
        """
        Runs a ast statement node and writes code into output_py
        """
        source = to_source(statement)
        if not source_required:
            code = compile(ast.Module([statement], []), "<string>", "exec")
        else:
            # TorchScript requires source code to exist on disk
            assert self.tempdir
            fn, filename = tempfile.mkstemp(suffix=".py", dir=self.tempdir, prefix="pb")
            with os.fdopen(fn, "w") as fd:
                fd.write(source)
                fd.flush()
            code = compile(source, filename, "exec")
        exec(code, self.output_module.__dict__, self.output_module.__dict__)
        self.output_py.writelines(["\n", source, "\n"])
