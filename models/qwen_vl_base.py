import os
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union

class QwenVL:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        trust_remote_code: bool = False,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Initialize QwenVL model
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism, valid values are 1, 2, 4, 8
            trust_remote_code: Whether to trust remote code
            gpu_memory_utilization: GPU memory utilization
        """
        # Validate tensor_parallel_size parameter
        valid_sizes = [1, 2, 4, 8]
        if tensor_parallel_size not in valid_sizes:
            print(f"Warning: tensor_parallel_size={tensor_parallel_size} is not valid, using default value 2")
            tensor_parallel_size = 2
            
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
        # Default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=4096,
            stop_token_ids=[],
        )

    def prepare_prompt(self, image: Union[List[Image.Image], Image.Image], prompt_template: str) -> Any:
        if not isinstance(image, List):
            image = [image]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    } for img in image
                    ] + [{"type": "text", "text": prompt_template}],
            },
        ]


        return messages
    
    def forward(self, final_prompt, custom_sampling_params: Optional[SamplingParams] = None):
        sampling_params = custom_sampling_params or self.default_sampling_params

        prompt = self.processor.apply_chat_template(
            final_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(final_prompt)
        resize_size = image_inputs[0].size if image_inputs is not None else None
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        # if resize_size is not None:
        #     return generated_text, resize_size
        
        return generated_text

    def generate(
        self,
        image: Union[List[Image.Image], Image.Image],
        prompt_template: str,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """
        Model forward inference
        
        Args:
            image: PIL Image object or list of images
            prompt_template: Prompt template
            custom_sampling_params: Custom sampling parameters
            
        Returns:
            Tuple[str, tuple]: (Generated text, image size)
        """
        if not isinstance(image, List):
            image = [image]

        sampling_params = custom_sampling_params or self.default_sampling_params
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    } for img in image
                    ] + [{"type": "text", "text": prompt_template}],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        resize_size = image_inputs[0].size if image_inputs is not None else None
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        if resize_size is not None:
            return generated_text, resize_size
        
        return generated_text
    


if __name__ == "__main__":
    qwen = QwenVL(model_path="pretrain/Qwen2.5-VL-72B-Instruct", tensor_parallel_size=2)
    image = Image.open("test.jpg")
    prompt_template = "Please describe the content of the image in detail."
    generated_text, resize_size = qwen.forward(image, prompt_template)
    print(generated_text)
