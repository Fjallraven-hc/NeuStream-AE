from contextlib import contextmanager
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload

from tqdm import tqdm
import time
import multiprocessing as mp
from PIL import Image
import logging
import os
from transformers import LlavaProcessor
import zmq
import pickle

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages)
from vllm.inputs import PromptInputs, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    GuidedDecodingRequest, get_local_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_kwargs, is_list_of
from vllm.core.predictor import LlamaPredictor

from vllm.logger import init_logger, add_consolehandler, add_filehandler
# if "MatPlotAgent" in os.environ and os.environ["MatPlotAgent"] is not None:

import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.parent.parent.parent.resolve() 
path = root_path / "MatPlotAgent"
# sys.path.append(os.environ['MatPlotAgent'])
sys.path.append(str(path))
import neustream.prompt_llava as prompt_llava
import neustream.prompt_codellama as prompt_codellama
from agents.utils import fill_in_placeholders

data_dict = {0: 979, 2: 729, 7: 525, 9: 803, 6: 1000, 12: 798, 16: 693, 18: 1000, 13: 1000, 14: 517, 19: 843, 30: 540, 22: 625, 25: 655, 29: 418, 32: 1000, 37: 587, 35: 886, 38: 645, 44: 625, 36: 576, 33: 1000, 43: 839, 47: 817, 56: 506, 45: 1000, 57: 303, 52: 843, 50: 535, 58: 1000, 64: 1000, 59: 1000, 72: 827, 62: 1000, 69: 393, 74: 1000, 71: 965, 75: 831, 80: 1000, 88: 653, 85: 459, 91: 620, 94: 1000, 97: 592, 100: 979, 90: 1000, 101: 1000, 95: 681, 99: 607, 110: 651, 105: 978, 104: 843, 116: 693, 117: 856, 111: 1000, 115: 1000, 121: 479, 123: 774, 130: 540, 126: 911, 129: 418, 134: 768, 137: 587, 128: 622, 138: 645, 141: 666, 136: 564, 144: 624, 143: 839, 153: 243, 146: 1000, 149: 1000, 152: 843, 157: 303, 150: 535, 155: 1000, 163: 1000, 159: 1000, 162: 740, 172: 810, 173: 611, 169: 393, 167: 1000, 181: 725, 177: 838, 187: 397, 183: 946, 180: 1000, 179: 1000, 186: 753, 188: 1000, 189: 847, 194: 1000, 190: 1000, 193: 1000, 198: 658, 200: 978, 203: 772, 210: 752, 206: 1000, 204: 843, 199: 607, 213: 1000, 217: 856, 218: 1000, 215: 1000, 221: 479, 223: 774, 230: 540, 225: 655, 224: 1000, 229: 418, 232: 1000, 235: 886, 227: 1000, 236: 576, 240: 932, 243: 839, 253: 243, 248: 549, 257: 303, 246: 1000, 252: 844, 254: 569, 250: 535, 258: 949, 260: 1000, 289: 855, 295: 998, 298: 1000, 291: 1000, 293: 1000, 1: 939, 3: 1000, 5: 978, 10: 651, 4: 843, 8: 852, 17: 856, 11: 1000, 20: 1000, 15: 1000, 23: 774, 21: 479, 26: 911, 31: 711, 34: 768, 24: 1000, 27: 1000, 28: 622, 39: 1000, 41: 666, 40: 970, 53: 243, 42: 1000, 48: 816, 49: 1000, 46: 1000, 54: 609, 55: 988, 51: 1000, 63: 810, 65: 1000, 60: 1000, 68: 1000, 67: 1000, 70: 951, 66: 1000, 87: 397, 79: 1000, 78: 1000, 86: 753, 96: 585, 89: 847, 92: 724, 98: 1000, 102: 730, 103: 772, 93: 1000, 107: 525, 106: 1000, 109: 803, 112: 798, 113: 1000, 108: 852, 118: 1000, 114: 517, 120: 1000, 119: 843, 122: 1000, 131: 711, 125: 655, 132: 1000, 124: 1000, 127: 1000, 135: 886, 139: 1000, 133: 1000, 140: 970, 142: 1000, 147: 825, 148: 816, 145: 1000, 156: 506, 154: 609, 161: 546, 151: 1000, 158: 1000, 164: 1000, 165: 784, 160: 1000, 168: 1000, 170: 951, 166: 1000, 171: 965, 182: 555, 174: 1000, 175: 1000, 176: 1000, 184: 815, 178: 1000, 185: 466, 191: 620, 196: 585, 192: 724, 197: 592, 207: 525, 209: 803, 195: 681, 202: 657, 201: 1000, 205: 979, 212: 906, 208: 856, 216: 693, 211: 882, 214: 517, 220: 1000, 219: 843, 222: 1000, 226: 911, 231: 847, 234: 768, 237: 587, 238: 645, 239: 1000, 228: 622, 241: 666, 233: 1000, 242: 1000, 247: 817, 245: 1000, 256: 885, 249: 1000, 261: 666, 255: 1000, 251: 1000, 259: 1000, 296: 566, 294: 1000, 297: 789, 292: 216, 290: 1000, 299: 607}

code_data_dict = {0: 617, 1: 732, 2: 607, 3: 573, 4: 681, 5: 617, 6: 563, 7: 343, 8: 859, 9: 350, 10: 405, 11: 932, 12: 502, 13: 461, 14: 1000, 15: 820, 16: 353, 17: 417, 18: 382, 19: 681, 20: 654, 21: 719, 22: 1000, 23: 388, 24: 996, 25: 742, 26: 624, 27: 1000, 28: 1000, 29: 666, 30: 179, 31: 375, 32: 506, 33: 1000, 34: 380, 35: 611, 36: 736, 37: 499, 38: 625, 39: 550, 40: 994, 41: 533, 42: 749, 43: 687, 44: 426, 45: 1000, 46: 934, 47: 723, 48: 709, 49: 818, 50: 1000, 51: 1000, 52: 909, 53: 279, 54: 728, 55: 803, 56: 433, 57: 428, 58: 808, 59: 1000, 60: 1000, 61: 270, 62: 960, 63: 619, 64: 664, 65: 602, 66: 954, 67: 880, 68: 717, 69: 829, 70: 776, 71: 809, 72: 357, 73: 355, 74: 559, 75: 665, 76: 727, 77: 500, 78: 901, 79: 804, 80: 763, 81: 383, 82: 429, 83: 407, 84: 532, 85: 695, 86: 565, 87: 282, 88: 354, 89: 611, 90: 1000, 91: 768, 92: 755, 93: 970, 94: 538, 95: 434, 96: 190, 97: 665, 98: 631, 99: 1000, 100: 617, 101: 644, 102: 607, 103: 557, 104: 761, 105: 617, 106: 555, 107: 343, 108: 859, 109: 348, 110: 405, 111: 969, 112: 502, 113: 530, 114: 1000, 115: 1000, 116: 353, 117: 417, 118: 382, 119: 761, 120: 654, 121: 718, 122: 1000, 123: 388, 124: 776, 125: 742, 126: 624, 127: 1000, 128: 992, 129: 606, 130: 179, 131: 375, 132: 506, 133: 1000, 134: 378, 135: 611, 136: 736, 137: 646, 138: 604, 139: 550, 140: 607, 141: 513, 142: 753, 143: 626, 144: 426, 145: 1000, 146: 1000, 147: 651, 148: 883, 149: 817, 150: 1000, 151: 1000, 152: 752, 153: 279, 154: 654, 155: 800, 156: 409, 157: 428, 158: 798, 159: 1000, 160: 1000, 161: 224, 162: 942, 163: 752, 164: 658, 165: 609, 166: 982, 167: 864, 168: 668, 169: 829, 170: 579, 171: 773, 172: 384, 173: 364, 174: 570, 175: 537, 176: 785, 177: 500, 178: 942, 179: 802, 180: 762, 181: 376, 182: 429, 183: 506, 184: 532, 185: 702, 186: 565, 187: 206, 188: 399, 189: 611, 190: 1000, 191: 1000, 192: 876, 193: 969, 194: 556, 195: 1000, 196: 190, 197: 686, 198: 649, 199: 1000, 200: 617, 201: 637, 202: 607, 203: 596, 204: 681, 205: 617, 206: 736, 207: 343, 208: 860, 209: 348, 210: 318, 211: 932, 212: 444, 213: 521, 214: 1000, 215: 820, 216: 353, 217: 347, 218: 382, 219: 761, 220: 666, 221: 729, 222: 757, 223: 388, 224: 996, 225: 793, 226: 624, 227: 1000, 228: 1000, 229: 606, 230: 179, 231: 375, 232: 680, 233: 1000, 234: 378, 235: 611, 236: 734, 237: 451, 238: 566, 239: 778, 240: 957, 241: 533, 242: 753, 243: 677, 244: 377, 245: 1000, 246: 934, 247: 651, 248: 709, 249: 812, 250: 1000, 251: 1000, 252: 763, 253: 276, 254: 730, 255: 803, 256: 423, 257: 428, 258: 802, 259: 1000, 260: 1000, 261: 270, 262: 558, 263: 619, 264: 681, 265: 575, 266: 1000, 267: 724, 268: 717, 269: 829, 270: 579, 271: 809, 272: 386, 273: 356, 274: 540, 275: 731, 276: 727, 277: 500, 278: 900, 279: 802, 280: 763, 281: 376, 282: 429, 283: 410, 284: 532, 285: 690, 286: 565, 287: 282, 288: 471, 289: 599, 290: 1000, 291: 505, 292: 757, 293: 1000, 294: 560, 295: 1000, 296: 190, 297: 735, 298: 649, 299: 1000}
logger = init_logger(__name__)


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use `max_seq_len_to_capture` instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
        **kwargs: Arguments for :class:`~vllm.EngineArgs`. (See
            :ref:`engine_args`)

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the :class:`~vllm.AsyncLLMEngine` class instead.
    """

    DEPRECATE_LEGACY: ClassVar[bool] = False
    """A flag to toggle whether to deprecate the legacy generate/encode API."""

    @classmethod
    @contextmanager
    def deprecate_legacy_api(cls):
        cls.DEPRECATE_LEGACY = True

        yield

        cls.DEPRECATE_LEGACY = False

    def __init__(
        self,
        model: str,
        name: str = "llm",
        workspace: str = str(path),
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False for decoder-only models and True
        for encoder/decoder models, since encoder/decoder models
        do not currently support CUDAGraph.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = (
            "image_token_id",
            "image_feature_size",
            "image_input_shape",
            "image_input_type",
        )
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError(
                "There is no need to pass vision-related arguments anymore.")
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS, name=name)
        self.request_counter = Counter()
        self.step_time_ls: List[float] = []
        self.name = name
        self.workspace = workspace
        self.is_initial = "codellama_1" == self.name
        if "llava" in self.name:
            # self.predictor = LlavaPredictor(0.95)
            self.processor = LlavaProcessor.from_pretrained(model)
            test_img = Image.open(f'{str(path)}/test.jpg')
            for _ in range(20):
                self.processor.image_processor(test_img, return_tensors="pt")["pixel_values"]
        # p_time, p_para, d_para = read_param(self.name.split('_')[0])
        # self.predictor = LlamaPredictor(0.95,p_time, p_para, d_para)
        
        self.log_file = f"{workspace}/log/{name}.log"
        self.class_name = "LLM"
        self.logger = init_logger(__file__)
        self.logger.setLevel(logging.DEBUG)
        # add_consolehandler(self.logger)
        add_filehandler(self.logger, os.path.abspath(self.log_file))
        self.logger.debug("-----"*30)
        self.logger.debug("llm.py imported.")       

    def get_tokenizer(self) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.llm_engine.get_tokenizer_group(TokenizerGroup)

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

    @overload  # LEGACY: single (prompt + optional token ids)
    def generate(
        self,
        prompts: str,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    def generate(
        self,
        prompts: Optional[str] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    def generate(
        self,
        prompts: None,
        sampling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload
    def generate(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        /,  # We may enable `inputs` keyword after removing the old API
        *,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @deprecate_kwargs(
        "prompts",
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'inputs' parameter instead.",
    )
    def generate(
        self,
        prompts: Union[Union[PromptInputs, Sequence[PromptInputs]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            inputs: A list of inputs to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``RequestOutput`` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).")

        if prompt_token_ids is not None:
            inputs = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            inputs = cast(Union[PromptInputs, Sequence[PromptInputs]], prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, RequestOutput)

    def chat(
        self,
        messages: List[ChatCompletionMessageParam],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A single conversation represented as a list of messages.
                Each message is a dictionary with 'role' and 'content' keys.
            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
              If not provided, the model's default chat template will be used.
            add_generation_prompt: If True, adds a generation template
                to each message.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
        """

        tokenizer = self.get_tokenizer()
        model_config = self.llm_engine.get_model_config()

        conversation, mm_data = parse_chat_messages(messages, model_config,
                                                    tokenizer)

        prompt: Union[str, List[int]]
        if isinstance(tokenizer, MistralTokenizer):
            prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            prompt = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
            )

        inputs: PromptInputs
        if is_list_of(prompt, int):
            inputs = TokensPrompt(prompt_token_ids=prompt)
        else:
            inputs = TextPrompt(prompt=prompt)

        if mm_data is not None:
            inputs["multi_modal_data"] = mm_data

        return self.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    @overload  # LEGACY: single (prompt + optional token ids)
    def encode(
        self,
        prompts: str,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    def encode(
        self,
        prompts: List[str],
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    def encode(
        self,
        prompts: Optional[str] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    def encode(
        self,
        prompts: Optional[List[str]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    def encode(
        self,
        prompts: None,
        pooling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload
    def encode(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        /,  # We may enable `inputs` keyword after removing the old API
        *,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @deprecate_kwargs(
        "prompts",
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'inputs' parameter instead.",
    )
    def encode(
        self,
        prompts: Union[Union[PromptInputs, Sequence[PromptInputs]],
                       Optional[Union[str, List[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[EmbeddingRequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            inputs: The inputs to the LLM. You may pass a sequence of inputs for
                batch inference. See :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            generated embeddings in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if not self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.encode() is only supported for embedding models (XModel)."
            )

        if prompt_token_ids is not None:
            inputs = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            inputs = cast(Union[PromptInputs, Sequence[PromptInputs]], prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=pooling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, EmbeddingRequestOutput)

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    # LEGACY
    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, List[str]]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
    ):
        # skip_tokenizer_init is now checked in engine

        if prompts is not None:
            prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]
        if prompt_token_ids is not None:
            prompt_token_ids = [
                p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
            ]

        num_requests = None
        if prompts is not None:
            num_requests = len(prompts)
        if prompt_token_ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompt_token_ids)):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

            num_requests = len(prompt_token_ids)
        if num_requests is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        inputs: List[PromptInputs] = []
        for i in range(num_requests):
            item: PromptInputs

            if prompts is not None:
                item = TextPrompt(prompt=prompts[i])
            elif prompt_token_ids is not None:
                item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
            else:
                raise AssertionError

            inputs.append(item)

        return inputs

    def _validate_and_add_requests(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
    ) -> None:
        if isinstance(inputs, (str, dict)):
            # Convert a single prompt to a list.
            inputs = [inputs]

        num_requests = len(inputs)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        if isinstance(params, list):
            params = [
                self._add_guided_processor(param, guided_options)
                if isinstance(param, SamplingParams) else param
                for param in params
            ]
        elif isinstance(params, SamplingParams):
            params = self._add_guided_processor(params, guided_options)

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
            )

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        arrival_time: Optional[float] = None,
        pslo: float = 1.5,
        dslo: float = 1.5,
        budget: float = 0,
        req_id: int = 0,        
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            inputs,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            arrival_time=arrival_time,
            pslo=pslo,
            dslo=dslo,
            budget=budget,
            req_id=req_id,
        )

    def _add_guided_processor(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options:
            if guided_options.guided_decoding_backend is None:
                decoding_config = self.llm_engine.get_decoding_config()
                guided_options.guided_decoding_backend = (
                    decoding_config.guided_decoding_backend)
            guided_logits_processor = get_local_guided_decoding_logits_processor(  #noqa
                guided_options.guided_decoding_backend, guided_options,
                self.get_tokenizer())
            if guided_logits_processor:
                if params.logits_processors is None:
                    params.logits_processors = []
                params.logits_processors.append(guided_logits_processor)
        return params

    def _run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # In the loop below, only finished outputs are used
        self.llm_engine.step_return_finished_only = True

        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            start = time.time()
            step_outputs, _ = self.llm_engine.step()
            self.step_time_ls.append(time.time() - start)
        return None
        #     for output in step_outputs:
        #         if output.finished:
        #             outputs.append(output)
        #             if use_tqdm:
        #                 if isinstance(output, RequestOutput):
        #                     # Calculate tokens only for RequestOutput
        #                     total_in_toks += len(output.prompt_token_ids)
        #                     in_spd = total_in_toks / pbar.format_dict["elapsed"]
        #                     total_out_toks += sum(
        #                         len(stp.token_ids) for stp in output.outputs)
        #                     out_spd = (total_out_toks /
        #                                pbar.format_dict["elapsed"])
        #                     pbar.postfix = (
        #                         f"est. speed input: {in_spd:.2f} toks/s, "
        #                         f"output: {out_spd:.2f} toks/s")
        #                 pbar.update(1)

        # # Restore original behavior
        # self.llm_engine.step_return_finished_only = False

        # if use_tqdm:
        #     pbar.close()
        # # Sort the outputs by request ID.
        # # This is necessary because some requests may be finished earlier than
        # # its previous requests.
        # return sorted(outputs, key=lambda x: int(x.request_id))

    def _is_encoder_decoder_model(self):
        return self.llm_engine.is_encoder_decoder_model()

    def _is_embedding_model(self):
        return self.llm_engine.is_embedding_model()

    def generate_prompt(self, query: str, type: str = "", original: str=""):
        if "llava" in self.name:
            information = {
                "query": original,
                "file_name": "novice_final.png",
                "code": query
            }
            prompt = ""
            prompt += "SYSTEM: " + fill_in_placeholders(prompt_llava.SYSTEM_PROMPT, information) + "\n"
            prompt += "USER: " + "<image>" * 576 + "\n" + fill_in_placeholders(prompt_llava.USER_PROMPT,information) + "\n"
            prompt += "ASSISTANT: "
            return prompt
        else:
            if type == "initial":
                information = {
                    "query": query,
                    "file_name": "novice.png"
                }
                prompt = ""
                prompt += "<s>[INST] <<SYS>>\n" + fill_in_placeholders(prompt_codellama.INITIAL_SYSTEM_PROMPT, information) + "\n<</SYS>>\n\n"
                prompt += fill_in_placeholders(prompt_codellama.INITIAL_USER_PROMPT, information) + " [/INST]"
                return prompt
            else:
                information ={
                    "query": "\n\n" + query,
                    "file_name": "novice_final.png"
                }
                prompt = ""
                prompt += "<s>[INST] <<SYS>>\n" + fill_in_placeholders(prompt_codellama.VIS_SYSTEM_PROMPT, information) + "\n<</SYS>>\n\n"
                prompt += fill_in_placeholders(prompt_codellama.VIS_USER_PROMPT, information) + " [/INST]"
                return prompt


    def serve(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        pslo: float,
        dslo: float,
        sampling_params: Optional[SamplingParams] = None,
    ):
        use_zmq = False
        if isinstance(input_queue, zmq.Socket):
            use_zmq = True
        self.reset_counter()
        outputs: List[RequestOutput] = []
        pred_time_list = []
        time_dict = {}
        if "codellama_2" in self.name:
            none_count = 0
        else:
            none_count = 1
        original_info = {}
        org_time_dict = {}
        real_time_dict = {}
        while none_count < 2:
            prompts = []
            if use_zmq:
                try:
                    data = input_queue.recv(flags=zmq.NOBLOCK)
                    prompts.append(pickle.loads(data))
                except zmq.Again as e:
                    ...
            else:
                qsize = input_queue.qsize()
                for _ in range(qsize):
                    # if use_zmq:
                    #     data = input_queue.recv()
                    #     item = pickle.loads(data)
                    # else:
                    item = input_queue.get()
                    prompts.append(item)
            for prompt in prompts:
                if prompt is None:
                    none_count += 1
                    continue
                req_id = prompt["req_id"]
                time_dict[req_id] = time.time()
                org_time_dict[req_id] = prompt['org_arrival_time']
                real_time_dict[req_id] = prompt['real_time']
                self.logger.debug(f"Request id: {req_id}, real_arrive_time: {time_dict[req_id]}, queue_put_time: {prompt['arrival_time']}, comm_cost: {time_dict[req_id]-prompt['arrival_time']}")
                if "llava" in self.name:
                    prompt_ = self.generate_prompt(prompt["query"],"",prompt["original"])
                    img_path = f"{self.workspace}/workspace3/example_{req_id}/novice.png"
                    img = Image.open(img_path)

                    pixel_values = self.processor.image_processor(img, return_tensors="pt")["pixel_values"].to("cuda")
                    img_features = self.llm_engine.model_executor.generate_embeds(pixel_values)#.cpu()
                    inp = TextPrompt(prompt=prompt_, multi_modal_data={"clip": img_features,})
                    real_arrival_time = prompt["arrival_time"]
                    elapsed = time.time() - real_arrival_time
                    self.logger.info(f"Process image take: {elapsed};")
                    if req_id in data_dict:
                        sampling_params.max_tokens = data_dict[req_id]
                    else:
                        sampling_params.max_tokens = 1000
                    self._add_request(inp, sampling_params, arrival_time=real_arrival_time, pslo=pslo, dslo=dslo,budget=prompt["budget"],req_id=req_id,)
                else:
                    if self.is_initial:
                        prompt_ = self.generate_prompt(prompt["query"],"initial")
                        original_info[req_id] = prompt["original"]
                        if req_id in code_data_dict:
                            sampling_params.max_tokens = code_data_dict[req_id]
                            sampling_params.ignore_eos = True
                        else:
                            sampling_params.max_tokens = 1000
                            sampling_params.ignore_eos = False
                    else:
                        prompt_ = self.generate_prompt(prompt["query"])
                    inp = {"prompt": prompt_}
                    self.logger.debug(f"Fill prompt: Request id: {req_id}, time: {time.time()-prompt['arrival_time']}s.")
                    self._add_request(inp, sampling_params,  arrival_time=prompt["arrival_time"], pslo=pslo, dslo=dslo,budget=prompt["budget"],req_id=req_id)
            if self.llm_engine.has_unfinished_requests():
                start = time.time()
                step_outputs, pred_time = self.llm_engine.step()
                now = time.time()
                pred_time_list.append((pred_time, now-start))
                # self.logger.debug(f"step time: {(now-start)*1e3} ms")
                for output in step_outputs:
                    if output.finished:
                        output.finished_time = now
                        # 这里需要判断是否投入下一个queue中
                        is_work, _, _, _, bu, _ = self.llm_engine.scheduler[0].predictor.check_seq_step_slo(output)
                        ## 如果没有抛弃且budget >= 0
                        text = output.outputs[0].text
                        if is_work and bu >= 0: ## Need add new field
                            req_id = output.req_id
                            self.logger.debug(f"Req: {req_id}, arrive_time: {time_dict[req_id]}, finish_time: {output.finished_time}")
                            out = {
                                "req_id": req_id,
                                "query": text,
                                "budget": bu,
                                "arrival_time": time.time(),
                                "org_arrival_time": org_time_dict[req_id],
                                "real_time": real_time_dict[req_id] + output.finished_time - time_dict[req_id]
                            }
                            if self.is_initial:
                                out["original"] = original_info.pop(req_id)
                            if use_zmq:
                                output_queue.send(pickle.dumps(out))
                            else:
                                output_queue.put(out)
                        self.logger.debug(f"Request id: {req_id} Generate text:\n{text}\n budget: {bu}")
                        # self.logger.debug(f"Req: {req_id}, ")
                        outputs.append(output)

        while self.llm_engine.has_unfinished_requests():
            start = time.time()
            step_outputs, pred_time = self.llm_engine.step()
            now = time.time()
            pred_time_list.append((pred_time, now-start))
            # self.logger.debug(f"step time: {now-start}s")
            for output in step_outputs:
                if output.finished:
                    output.finished_time = now
                    # 这里需要判断是否投入下一个queue中
                    is_work, _, _, _, bu, _ = self.llm_engine.scheduler[0].predictor.check_seq_step_slo(output)
                    ## 如果没有抛弃且budget >= 0
                    if is_work and bu >= 0: ## Need add new field
                        text = output.outputs[0].text
                        req_id = output.req_id
                        # self.logger.debug(f"Req: {req_id}, arrive_time: {time_dict[req_id]}, finish_time: {output.finished_time}")
                        out = {
                            "req_id": req_id,
                            "query": text,
                            "budget": bu,
                            "arrival_time": time.time(),
                            'org_arrival_time': org_time_dict[req_id],
                            "real_time": real_time_dict[req_id] + output.finished_time - time_dict[req_id]
                        }
                        if self.is_initial:
                            out["original"] = original_info.pop(req_id)
                        self.logger.debug(f"Request id: {req_id} Generate text: \n{text}\n budget: {bu}")
                        if use_zmq:
                                output_queue.send(pickle.dumps(out))
                        else:
                                output_queue.put(out)
                    outputs.append(output)
        if use_zmq:
            output_queue.send(pickle.dumps(None))
        else:
            output_queue.put(None)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        self.logger.debug(f"pred_time_list: {pred_time_list}")
        return outputs


    def reset_counter(self):
        self.request_counter = Counter()