from typing import TextIO, Union, List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

DEFAULT_PROMPT_TEMPLATE = """You are a PyTorch testing expert. I will provide you with Python code containing PyTorch nn.Module definitions. Your task is to analyze these modules and generate comprehensive test cases in a specific format.

For each nn.Module class in the code, create a test case where each line contains:
1. The module class name
2. kwargs for module's constructor method
3. kwargs for module's forward method

The output must only in the following this exact format:
Mlp
{{"in_features": 64, "hidden_features": 128, "out_features": 64}}
{{"x": [4, 64]}}
RMSNorm
{{"dim": 64, "elementwise_affine": True, "eps": 1e-6}}
{{"x": [4, 64]}}

Rules for generating test cases:
- Include all nn.Module definitions
- Deduce appropriate tensor shapes based on the module's architecture
- Input tensor shapes must be compatible with the layers in the module
- Use standard batch sizes (e.g., 4) and reasonable input dimensions
- Include all arguments from __init__ method
- Include all arguments from forward method
- Include all PyTorch nn.Module classes
- For loss functions, ensure input and target tensors have compatible shapes
- Follow PyTorch's dimension conventions (batch_size, channels, height, width)
- Use torch.rand() for creating input tensors
- For device placement, use "cpu"
- Keep tensor dimensions reasonable (e.g., 4, 8, 16, 32, 64)
- All parenthesis/brackets/curly braces must be properly closed
- Maintain proper indentation in the output

Here's the code to analyze:
{code}

Generate only the test cases for all nn.Module classes found in the code. Do not include any explanations or other text."""


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
            {"role": "user", "content": self.template.format(code=code)}
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
            "max_new_tokens": 8192,
            "temperature": 0.01,
            "top_p": 0.8,
            "top_k": 30,
            **(generation_config or {}),
        }

    def load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                max_memory={"cpu": "16GiB"},  # Limits CPU memory
                low_cpu_mem_usage=True,  # Reduces CPU memory during loading
                # offload_folder="offload_folder",  # Optional: directory for offloading
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
            # Process lines in groups of 3
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    raise TestCaseParsingError(f"Incomplete test case at line {i}")

                # Extract components
                class_name = lines[i]
                init_args = lines[i + 1]
                forward_args = lines[i + 2]

                # init_args = json.loads(lines[i + 1])
                # forward_args = json.loads(lines[i + 2])
                # # Validate components
                # if not isinstance(init_args, dict):
                #     raise TestCaseParsingError(
                #         f"Invalid init args format for {class_name}"
                #     )
                # if not isinstance(forward_args, dict):
                #     raise TestCaseParsingError(
                #         f"Invalid forward args format for {class_name}"
                #     )
                # test_cases.append((class_name, repr(init_args), repr(forward_args)))

                test_cases.append((class_name, init_args, forward_args))

        except json.JSONDecodeError as e:
            raise TestCaseParsingError(f"Invalid JSON format: {str(e)}")
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
        inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Generate output with optimized settings
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=False,  # Enable KV-cache for faster generation
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
            print(f"Init args: {json.dumps(init_args, indent=2)}")
            print(f"Forward args: {json.dumps(forward_args, indent=2)}")

    except TestCaseParsingError as e:
        print(f"Error parsing test cases: {e}")
    except Exception as e:
        print(f"Error: {e}")
