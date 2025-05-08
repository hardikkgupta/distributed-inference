from typing import List, Dict, Any, Optional
import ray
from ray.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import vllm
from vllm import LLM, SamplingParams

class BatchInferencePipeline:
    def __init__(
        self,
        model_name: str,
        num_gpus: int = 8,
        batch_size: int = 32,
        max_tokens: int = 2048,
        tensor_parallel_size: int = 8,
    ):
        """Initialize the batch inference pipeline.
        
        Args:
            model_name: Name of the model to load
            num_gpus: Number of GPUs to use
            batch_size: Batch size for processing
            max_tokens: Maximum number of tokens to generate
            tensor_parallel_size: Size of tensor parallelism
        """
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        # Initialize vLLM model
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=4096,
        )
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        
    def process_batch(self, input_data: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of inputs using Ray Data.
        
        Args:
            input_data: List of input prompts
            
        Returns:
            List of generated outputs with metadata
        """
        # Create Ray Dataset
        ds = ray.data.from_items(input_data)
        
        # Process in batches
        results = ds.map_batches(
            self._process_batch,
            batch_size=self.batch_size,
            num_gpus=self.num_gpus,
            compute=ray.data.ActorPoolStrategy(size=self.num_gpus),
        )
        
        return results.take_all()
    
    def _process_batch(self, batch: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        """Process a single batch of inputs.
        
        Args:
            batch: Dictionary containing input prompts
            
        Returns:
            Dictionary containing generated outputs
        """
        prompts = batch["item"]
        
        # Generate outputs using vLLM
        outputs = self.model.generate(
            prompts,
            sampling_params=self.sampling_params,
        )
        
        # Process and format results
        results = []
        for output in outputs:
            results.append({
                "generated_text": output.outputs[0].text,
                "num_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            })
            
        return {"results": results}

class OnlineInferenceServer:
    def __init__(
        self,
        model_name: str,
        num_gpus: int = 8,
        tensor_parallel_size: int = 8,
    ):
        """Initialize the online inference server.
        
        Args:
            model_name: Name of the model to load
            num_gpus: Number of GPUs to use
            tensor_parallel_size: Size of tensor parallelism
        """
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize vLLM model
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=4096,
        )
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """Generate text for a single prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing generated output and metadata
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        outputs = self.model.generate(
            [prompt],
            sampling_params=sampling_params,
        )
        
        output = outputs[0].outputs[0]
        return {
            "generated_text": output.text,
            "num_tokens": len(output.token_ids),
            "finish_reason": output.finish_reason,
        } 