import logging
import gc
from typing import Any, Dict, List, TextIO, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

SEP = "|"
DEFAULT_PROMPT_TEMPLATE = """You are a PyTorch testing expert. I will provide you with Python code containing PyTorch nn.Module definitions. Your task is to analyze these modules and generate comprehensive test cases in a specific format.

For each nn.Module class in the code, create a test case where each line contains:
1. The module class name
2. kwargs for module's __init__ method
3. kwargs for module's forward method

The output must only in the following this exact format:
RMSNorm{sep}{{"dim": 64, "elementwise_affine": True, "eps": 1e-6}}{sep}{{"x": torch.rand([4, 64])}}

Rules for generating test cases:
- Must use correct Python syntax and keywords
- Include all PyTorch nn.Module sub-classes defined in the code
- Deduce appropriate tensor shapes based on the module's architecture
- Input tensor shapes must be compatible with the layers in the module
- Use standard batch sizes (e.g., 4) and reasonable input dimensions
- Each field is separted by "{sep}"
- Include all arguments from __init__ method
- Include all arguments from forward method
- For loss functions, ensure input and target tensors have compatible shapes
- Follow PyTorch's dimension conventions (batch_size, channels, height, width)
- Use torch.rand() for creating input tensors
- For device placement, use "cpu"
- Keep tensor dimensions reasonable (e.g., 4, 8, 16, 32, 64)
- Brackets and quotes must be properly closed

Here's the code to analyze:
{code}

Generate only the test cases for all nn.Module classes found in the code. Do not include any explanations or other text."""

TESTCASE_TEMPLATE = """

# Auto-generated test cases
import torch.nn as nn

TESTCASES = [
    # (nn.Module, init_args, forward_args)
{test_cases}
]

"""


# Mlp{sep}{{"in_features": 64, "hidden_features": 128, "out_features": 64}}{sep}{{"x": [4, 64]}}
class TestCaseParsingError(Exception):
    """Exception raised for errors during test case parsing."""

    pass


class PromptTemplate:
    """A class to manage the prompt template for test case generation."""

    def __init__(self, template: str = DEFAULT_PROMPT_TEMPLATE):
        """
        Initialize the prompt template.

        Args:
            template (str): The template string with {code} placeholder
        """
        self._validate_template(template)
        self.template = template

        # Pre-create messages for reuse
        self.base_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes PyTorch modules.",
            }
        ]

    def _validate_template(self, template: str):
        """
        Validate that the template contains the required placeholder.

        Args:
            template (str): Template to validate

        Raises:
            ValueError: If template doesn't contain {code} placeholder
        """
        if "{code}" not in template:
            raise ValueError("Template must contain {code} placeholder")

    def create_messages(self, code: str) -> List[Dict[str, str]]:
        """
        Create the complete message list for the model.

        Args:
            code (str): The code to analyze

        Returns:
            List[Dict[str, str]]: List of message dictionaries
        """
        return self.base_messages + [
            {"role": "user", "content": self.template.format(sep=SEP, code=code)}
        ]

    @classmethod
    def default(cls) -> "PromptTemplate":
        """Create a PromptTemplate with the default template."""
        return cls(DEFAULT_PROMPT_TEMPLATE)


class PyTorchTestGenerator:
    """A class to generate test cases for PyTorch modules using LLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M",
        prompt_template: Union[str, PromptTemplate] = None,
        generation_config: Dict[str, Any] = None,
    ):
        """
        Initialize the test generator.

        Args:
            model_name (str): Name/path of the model to use for generation
            prompt_template: Either a template string or PromptTemplate instance
            generation_config (Dict[str, Any]): Configuration for text generation
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        # Ensure model is loaded
        self.load_model()

        # Set up prompt template
        if prompt_template is None:
            self.prompt_template = PromptTemplate.default()
        elif isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(prompt_template)
        elif isinstance(prompt_template, PromptTemplate):
            self.prompt_template = prompt_template
        else:
            raise ValueError(
                f"Expected str or PromptTemplate, got {type(prompt_template)}"
            )

        # Set up generation config with defaults
        self.generation_config = {
            "max_new_tokens": 12288,
            "temperature": 0.001,  # Very low temperature for near-deterministic output
            "top_p": 0.95,  # Nucleus sampling threshold
            "top_k": 40,  # Limit vocabulary choices
            # "repetition_penalty": 1.1,  # Prevent redundant test cases
            # "length_penalty": 1.0,  # Balance between length and quality
            **(generation_config or {}),
        }

    def load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            # Store the device for reuse
            self.device = next(self.model.parameters()).device

    def read_input(self, input_source: Union[str, TextIO]) -> str:
        """
        Read content from either a string or a file-like object.

        Args:
            input_source: Either a string containing the code or a file-like object

        Returns:
            str: The code content

        Raises:
            ValueError: If the input_source is neither a string nor a TextIO object
        """
        if isinstance(input_source, str):
            return input_source
        elif isinstance(input_source, TextIO):
            input_source.seek(0)
            return input_source.read()
        else:
            raise ValueError(
                f"Expected string or TextIO object, got {type(input_source)}"
            )

    def parse_output(
        self, output: str
    ) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """
        Parse the model output into a list of test case tuples.

        Args:
            output (str): Raw model output

        Returns:
            List[Tuple[str, Dict, Dict]]: List of (class_name, init_args, forward_args) tuples

        Raises:
            TestCaseParsingError: If the output format is invalid
        """
        test_cases = []
        lines = [line.strip() for line in output.strip().split("\n") if line.strip()]

        try:
            for i, line in enumerate(lines):
                parts = line.split(SEP)
                if len(parts) != 3:
                    log.warning(f"Incomplete test case at line {i}: {line}")
                    continue

                # Extract components
                class_name, init_args, forward_args = parts
                test_cases.append(
                    (class_name.strip(), init_args.strip(), forward_args.strip())
                )

        except Exception as e:
            raise TestCaseParsingError(f"Error parsing output: {str(e)}")

        return test_cases

    @torch.no_grad()  # Disable gradient computation for inference
    def generate_testcases(
        self, input_source: Union[str, TextIO]
    ) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """
        Generate test cases for the provided PyTorch code.

        Args:
            input_source: Either a string containing the code or a file-like object

        Returns:
            List[Tuple[str, Dict, Dict]]: List of (class_name, init_args, forward_args) tuples

        Raises:
            ValueError: If the input_source is invalid
            TestCaseParsingError: If the output format is invalid
        """
        # Read the input
        code = self.read_input(input_source)

        # Create messages using the template
        messages = self.prompt_template.create_messages(code)

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Prepare inputs and move to device
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate output with optimized settings
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_config,
            use_cache=True,  # Enable KV-cache for faster generation
        )

        # Process output efficiently
        start_idx = len(inputs.input_ids[0])
        generated_text = self.tokenizer.decode(
            generated_ids[0][start_idx:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Parse and return formatted test cases
        return self.parse_output(generated_text.strip())

    def write_testcases_to_file(self, filepath: str, test_cases: List[tuple]):
        """
        Append test cases to the end of the source file.

        Args:
            filepath: Path to the source file
            test_cases: List of (class_name, init_args, forward_args) tuples
        """
        if not test_cases:
            return

        # Convert test cases to formatted strings
        formatted_cases = []
        for name, init_args, forward_args in test_cases:
            case_str = f"    ({name},\n"
            case_str += f"     lambda: ([], {init_args}),\n"
            case_str += f"     lambda: ([], {forward_args})),"
            formatted_cases.append(case_str)

        # Join all test cases with newlines
        all_cases = "\n".join(formatted_cases)

        # Create the complete test case section
        test_section = TESTCASE_TEMPLATE.format(test_cases=all_cases)

        # Append to file
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(test_section)

    @classmethod
    def process_source_files(
        cls,
        src_paths: List[str],
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        batch_size: int = 1,
        cleanup_cuda: bool = True,
    ):
        """
        Process multiple source files and generate test cases with proper resource cleanup.

        Args:
            src_paths: List of file paths to process
            model_name: Name of the model to use
            batch_size: Number of files to process before cleanup
            cleanup_cuda: Whether to perform CUDA cleanup after each batch
        """
        log = logging.getLogger(__name__)

        # Convert all paths to Path objects for consistency
        paths = src_paths

        for i in range(0, len(paths), batch_size):
            # Create generator instance for this batch
            generator = cls(model_name=model_name)
            batch_paths = paths[i : i + batch_size]

            try:
                # Process each file in the batch
                for src_path in batch_paths:
                    log.info(f"Generating test cases for {src_path}...")

                    try:
                        with open(src_path, "r", encoding="utf-8") as src_file:
                            src = src_file.read()
                            test_cases = generator.generate_testcases(src)
                        generator.write_testcases_to_file(src_path, test_cases)
                        log.info(
                            f"Successfully wrote {len(test_cases)} test cases to {src_path}"
                        )

                    except Exception as e:
                        log.warning(f"Error processing {src_path}: {str(e)}")
                        continue

            finally:
                # Cleanup after batch processing
                del generator
                gc.collect()

                if cleanup_cuda and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


# Example usage:
if __name__ == "__main__":
    # Create generator
    generator = PyTorchTestGenerator()

    # Test code
    code_str = """
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 128)
        
        def forward(self, x):
            return self.linear(x)
    """

    try:
        # Generate test cases
        test_cases = generator.generate_testcases(code_str)

        # Print formatted results
        print("\nGenerated test cases:")
        for class_name, init_args, forward_args in test_cases:
            print(f"\nClass: {class_name}")
            print(f"Init args: {init_args}")
            print(f"Forward args: {forward_args}")

    except TestCaseParsingError as e:
        print(f"Error parsing test cases: {e}")
    except Exception as e:
        print(f"Error: {e}")
