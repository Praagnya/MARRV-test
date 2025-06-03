from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    name: str = "microsoft/DialoGPT-medium"  # Default model
    tokenizer_name: str = "microsoft/DialoGPT-medium" 
    device_map: str = "auto"
    max_input_length: int = 2048

    @classmethod
    def get_better_math_model(cls):
        """Get a model better suited for mathematical reasoning"""
        # Try these models in order of preference
        models_to_try = [
    ("meta-math/MetaMath-7B", "meta-math/MetaMath-7B"),
    ("WizardLM/WizardMath-7B-V1.1", "WizardLM/WizardMath-7B-V1.1"),
    ("microsoft/phi-2", "microsoft/phi-2"),
      ]
        
        # For now, return GPT-2 which is better for text generation
        return cls(
            name="gpt2-medium",
            tokenizer_name="gpt2-medium",
            device_map="auto",
            max_input_length=1024
        )


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class AgentConfig:
    """Main configuration class for MetaThinkingAgent"""
    model: ModelConfig
    generation: GenerationConfig
    meta_agent_adapter_path: Optional[str] = None
    
    @classmethod
    def default(cls):
        """Create default configuration"""
        return cls(
            model=ModelConfig(),
            generation=GenerationConfig()
        )
    
    @classmethod
    def for_math_reasoning(cls):
        """Configuration optimized for mathematical reasoning"""
        return cls(
            model=ModelConfig.get_better_math_model(),
            generation=GenerationConfig(
                max_new_tokens=400,
                temperature=0.7,  # Balanced creativity and focus
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3  # Higher to avoid repetition
            )
        )

class MetaThinkingAgent:
    """Agent responsible for strategic planning and task decomposition"""
    
    def __init__(self, config: AgentConfig = None, base_model=None, tokenizer=None):
        # Use default config if none provided
        self.config = config if config is not None else AgentConfig.default()
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer if not provided
        if base_model is None or tokenizer is None:
            self._load_model_and_tokenizer()
        else:
            self.model = base_model
            self.tokenizer = tokenizer
            
        self.model.eval()  # Set to evaluation mode
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from config"""
        try:
            self.logger.info(f"Loading tokenizer: {self.config.model.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.tokenizer_name,
                trust_remote_code=True
            )
            
            self.logger.info(f"Loading model: {self.config.model.name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                device_map=self.config.model.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Load LoRA adapter if specified
            if hasattr(self.config, 'meta_agent_adapter_path') and self.config.meta_agent_adapter_path:
                self.logger.info(f"Loading LoRA adapter: {self.config.meta_agent_adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.config.meta_agent_adapter_path
                )
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
    def generate_plan(self, problem: str, memory_context: Optional[str] = None) -> str:
        """Generate a strategic plan for solving the problem"""
        
        # Construct prompt with memory context if available
        prompt = self._construct_prompt(problem, memory_context)
        
        # Generate response
        try:
            plan = self._generate_response(prompt)
            self.logger.info("Successfully generated strategic plan")
            return plan
        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}")
            return f"Error generating plan: {str(e)}"
    
    def _construct_prompt(self, problem: str, memory_context: Optional[str] = None) -> str:
        """Construct the prompt for plan generation"""
        
        base_instruction = """You are a mathematical problem-solving assistant. Create a clear, step-by-step plan to solve this problem.

Requirements:
1. Break down the problem into logical steps
2. Identify the mathematical method to use
3. List the specific calculations needed
4. Mention what the final answer should look like

"""
        
        if memory_context:
            prompt = f"""{base_instruction}
Previous examples:
{memory_context}

Problem to solve: {problem}

Step-by-step solution plan:
1."""
        else:
            prompt = f"""{base_instruction}
Problem to solve: {problem}

Step-by-step solution plan:
1."""
        
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using the model"""
        
        # Tokenize input
        max_input_length = getattr(self.config.model, 'max_input_length', 2048)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        ).to(self.model.device)
        
        # Generation parameters
        generation_params = {
            'max_new_tokens': self.config.generation.max_new_tokens,
            'temperature': self.config.generation.temperature,
            'top_p': self.config.generation.top_p,
            'do_sample': self.config.generation.do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Add repetition penalty if configured
        if hasattr(self.config.generation, 'repetition_penalty'):
            generation_params['repetition_penalty'] = self.config.generation.repetition_penalty
            
        # Generate with proper memory management
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_params
            )
            
        # Decode and extract plan
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        plan = generated_text[len(prompt):].strip()
        
        return self._clean_plan(plan)
    
    def _clean_plan(self, plan: str) -> str:
        """Clean and format the generated plan"""
        
        # Remove any unwanted artifacts and repetitive text
        lines = plan.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('###'):  # Remove markdown headers if any
                # Check for repetitive content
                if line not in seen_lines and len(line.split()) > 2:  # Avoid single words
                    cleaned_lines.append(line)
                    seen_lines.add(line)
                elif len(cleaned_lines) == 0:  # Keep first line even if short
                    cleaned_lines.append(line)
                    
        # Limit to reasonable length
        if len(cleaned_lines) > 10:
            cleaned_lines = cleaned_lines[:10]
            
        return '\n'.join(cleaned_lines)
    
    def decompose_task(self, problem: str, plan: str) -> Dict[str, Any]:
        """Decompose the problem into subtasks based on the plan"""
        
        subtasks = []
        lines = plan.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line.startswith(f'{i+1}.') or line.startswith('-')):
                subtasks.append({
                    'id': i,
                    'description': line,
                    'status': 'pending',
                    'dependencies': []  # Could be enhanced to detect dependencies
                })
        
        return {
            'original_problem': problem,
            'plan': plan,
            'subtasks': subtasks,
            'total_subtasks': len(subtasks)
        }
    
    def evaluate_plan_quality(self, problem: str, plan: str) -> Dict[str, float]:
        """Evaluate the quality of the generated plan"""
        
        # Simple heuristics for plan quality
        lines = [line.strip() for line in plan.split('\n') if line.strip()]
        
        # Check for structured steps
        structured_steps = sum(1 for line in lines if any(line.startswith(f'{i}.') for i in range(1, 11)))
        
        # Check for key mathematical concepts
        math_keywords = ['equation', 'solve', 'calculate', 'substitute', 'simplify', 'factor']
        keyword_coverage = sum(1 for keyword in math_keywords if keyword in plan.lower())
        
        return {
            'completeness': min(structured_steps / 5.0, 1.0),  # Expect ~5 steps
            'mathematical_relevance': keyword_coverage / len(math_keywords),
            'clarity': len(lines) / max(len(plan.split()), 1),  # Words per line ratio
            'overall_score': (structured_steps + keyword_coverage) / 10.0
        }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            # Import torch locally to avoid None reference issues
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass  # Ignore cleanup errors during shutdown


# Example usage
if __name__ == "__main__":
    print("Creating agent with math-optimized config...")
    
    # Use math-optimized configuration with GPT-2
    config = AgentConfig.for_math_reasoning()
    agent = MetaThinkingAgent(config)
    
    # Generate a plan
    problem = "Solve the quadratic equation 2x² + 5x - 3 = 0"
    print(f"\nProblem: {problem}")
    
    plan = agent.generate_plan(problem)
    print("Generated Plan:")
    print(plan)
    
    # Decompose into subtasks
    task_breakdown = agent.decompose_task(problem, plan)
    print(f"\nTask Breakdown ({task_breakdown['total_subtasks']} subtasks):")
    for subtask in task_breakdown['subtasks']:
        print(f"- {subtask['description']}")
    
    # Evaluate plan quality
    quality_metrics = agent.evaluate_plan_quality(problem, plan)
    print(f"\nPlan Quality Metrics:")
    for metric, score in quality_metrics.items():
        print(f"- {metric.replace('_', ' ').title()}: {score:.2f}")
        
    print(f"\nModel used: {config.model.name}")
    print("Note: For better results, consider using instruction-tuned models like:")
    print("- microsoft/DialoGPT-large")
    print("- Any fine-tuned math reasoning model from Hugging Face")