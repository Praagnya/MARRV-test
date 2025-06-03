from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

class MetaThinkingAgent:
    """Agent responsible for strategic planning and task decomposition"""
    
    def __init__(self, config, base_model=None, tokenizer=None):
        self.config = config
        
        # Load model and tokenizer if not provided
        if base_model is None or tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                device_map=config.model.device_map,
                torch_dtype=torch.float16
            )
            
            # Load LoRA adapter if specified
            if hasattr(config, 'meta_agent_adapter_path') and config.meta_agent_adapter_path:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    config.meta_agent_adapter_path
                )
        else:
            self.model = base_model
            self.tokenizer = tokenizer
            
        self.model.eval()  # Set to evaluation mode
        
    def generate_plan(self, problem, memory_context=None):
        """Generate a strategic plan for solving the problem"""
        # Construct prompt with memory context if available
        if memory_context:
            prompt = f"""You are a strategic planning agent. Your task is to create a step-by-step plan to solve this math problem.

Previous similar problems and strategies:
{memory_context}

Current Problem: {problem}

Strategic Plan:"""
        else:
            prompt = f"""You are a strategic planning agent. Your task is to create a step-by-step plan to solve this math problem.

Problem: {problem}

Strategic Plan:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                do_sample=self.config.generation.do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode and extract plan
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        plan = generated_text[len(prompt):]
        
        return plan.strip()