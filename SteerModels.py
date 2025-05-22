import os

import torch.nn as nn
import torch

from transformers import Qwen2ForCausalLM, Qwen2Model, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import TextStreamer

# Add these imports.
from typing import Optional, Tuple, Union, List, Dict
from typing_extensions import Unpack

from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import FlashAttentionKwargs
from jinja2 import Template


# CausalLM can only be called from the outside.
__all__ = [
    # 'NSSLlamaForCausalLM',
    'SteerQwen2ForCausalLM',
    # 'NSSGemma2ForCausalLM'
    ]

class SteerQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, 
                 layer_idx: int, 
                 steering_vector: Optional[torch.Tensor] = None, 
                 apply_steering: bool = False,
                 apply_resid_cache: bool=False,
                 strength: float = 1.0
                 ):
        super().__init__(config, layer_idx)
        self.apply_steering = apply_steering
        self.strength = strength
        self.apply_resid_cache = apply_resid_cache
        self.resid_pre_cache = []
        self.layer_idx = layer_idx


    def set_steering_parameters(
        self, 
        steering_vector: Optional[torch.Tensor], 
        apply_steering: bool = False,
        strength: float = 1.0,
        device: Optional[torch.device] = None):

        self.steering_vector = steering_vector * strength
        # Set the steering_vector to bfloat16.
        self.steering_vector = self.steering_vector.to(torch.bfloat16)
        self.apply_steering = apply_steering
        self.strength = strength
        self.print_steering_parameters()
        
    def print_steering_parameters(self):
        print(f"Layer Index: {self.layer_idx}\tApply Steering: {self.apply_steering}\tStrength: {self.strength}\n\tSteering Vector: {self.steering_vector}")
        # pass

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            
        hidden_states = hidden_states + self.steering_vector.to(hidden_states.device)
            
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
        
class SteerQwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config, 
                 steering_vectors: Optional[torch.Tensor] = None,
                 apply_steering_indices: Optional[list[bool]] = None,
                 strength: Optional[list[float]] = None,
                 apply_resid_cache: bool=False):
        super().__init__(config)
        # Replace this, the rest remains unchanged.
        self.steering_vectors = steering_vectors
        self.apply_steering_indices = apply_steering_indices
        self.strength = strength if strength is not None else [1.0] * config.num_hidden_layers
        print("apply_steering_indices: ", apply_steering_indices)
        self.layers = nn.ModuleList(
            [SteerQwen2DecoderLayer(
                config=config, 
                layer_idx=layer_idx, 
                steering_vector=None, 
                apply_steering=apply_steering_indices[layer_idx] if apply_steering_indices is not None else False, 
                strength=self.strength[layer_idx] if self.strength is not None else 0.0,
                apply_resid_cache=apply_resid_cache
            )
             for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(self, 
                                steering_vectors: Optional[torch.Tensor], 
                                apply_steering_indices: Optional[list[bool]],
                                strength: Optional[list[float]] = None,
                                device: Optional[torch.device] = None):
        # Get the current device of the model.
        if device is None:
            device = next(self.parameters()).device
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_vector = None
            if steering_vectors is not None:
                layer_steering_vector = steering_vectors[layer_idx].to(device)
                
            layer.set_steering_parameters(
                steering_vector=layer_steering_vector, 
                apply_steering=apply_steering_indices[layer_idx] if apply_steering_indices is not None else False,
                strength=strength[layer_idx] if strength is not None else 1.0
            )
            torch.cuda.empty_cache()
        

class SteerQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config, 
                 steering_vectors: Optional[torch.Tensor] = None, # Here is a three-dimensional array, [layer_idx, head_idx, head_idx], initialized to 0.
                 apply_steering_indices: Optional[list[bool]] = None,
                 strength: Optional[list[float]] = None,
                 apply_resid_cache: bool=False):
        super().__init__(config)
        self.model = SteerQwen2Model(
            config=config, 
            steering_vectors=steering_vectors, 
            apply_steering_indices=apply_steering_indices, 
            strength=strength, 
            apply_resid_cache=apply_resid_cache
        )

    @classmethod
    def from_pretrained(cls, *args, steering_vectors=None, apply_steering_indices=None, strength=None, **kwargs):
        # Call the parent class's from_pretrained method to load the model.
        model = super().from_pretrained(*args, **kwargs)
        model.set_steering_parameters(steering_vectors, apply_steering_indices, strength)
        # print("model.model.apply_steering_indices before set_steering_parameters: ", model.model.apply_steering_indices)
        # print("steering_vectors: ", steering_vectors)
        # print("apply_steering_indices: ", apply_steering_indices)
        # print("strength: ", strength)
        # print("model.model.apply_steering_indices after set_steering_parameters: ", model.model.apply_steering_indices) # In fact, this parameter only exists on the layer, so it is None here.
        return model

    def run_with_cache(self, input_ids, attention_mask):
        self.model.empty_resid_pre_cache()
        # Set apply_resid_cache to True for each layer
        for layer in self.model.layers:
            layer.apply_resid_cache = True
        outputs = self.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, num_return_sequences=1)
        cache = [layer.resid_pre_cache for layer in self.model.layers]
        for layer in self.model.layers:
            layer.apply_resid_cache = False
        return outputs

    def set_steering_parameters(
            self, 
            steering_vectors: Optional[torch.Tensor], 
            apply_steering_indices: Optional[list[bool]],
            strength: Optional[list[float]] = None):
        
        # Ensure that the steering_vector is on the device where the model is located.
        device = next(self.parameters()).device
        if steering_vectors is not None:
            steering_vectors = steering_vectors.to(device)
            
        self.model.set_steering_parameters(
            steering_vectors=steering_vectors, 
            apply_steering_indices=apply_steering_indices, 
            strength=strength,
            device=device
        )

    def empty_resid_pre_cache(self):
        for layer in self.model.layers:
            layer.resid_pre_cache = []
            torch.cuda.empty_cache()
            
    def get_steering_vector(self):
        return torch.cat([layer.steering_vector for layer in self.model.layers], dim=1)



if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LlamaConfig, Qwen2Config, Gemma2Config    
    import torch
    import numpy as np
    import time
    import argparse
    torch.manual_seed(42)
    np.random.seed(42)
    
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--strength', type=float, default=0.0,
                            help='steering strength.')
        args, _ = parser.parse_known_args()
        return args, _

    args, _ = parse_args()
    constant_strength = args.strength
    
    # DEVICE = "cuda:7"
    DEVICE = "cuda:0"
    
    import torch
    print("torch.cuda.device_count(): ", torch.cuda.device_count())
    
    models_list = [
        ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/Qwen2.5-7B-Instruct"),
        ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/DeepSeek-R1-Distill-Qwen-1.5B"),
        ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/DeepSeek-R1-Distill-Qwen-7B"),
    ]

    for model_name, model_class, config_class, model_id in models_list:
        print(model_class.__name__)

    model_idx = 1
    model_name, model_class, config_class, model_id = models_list[model_idx]

    print(model_id)
    config = config_class.from_pretrained(model_id)
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    
    print(config)

    steering_vectors = None
    # apply_steering_indices = [False] * num_layers
    # strength = [1.0] * num_layers
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    
    # Read steering_vectors.
    # steering_vectors = np.load('Data_gen/SteeringVectors/Qwen-7B.npy')
    tmp = model_id.split("/")[-1]
    steering_vectors = np.load(f"Assets/MATH/{tmp}/mean_steering_vectors.npy")
    apply_steering_indices = [True] * num_layers
    # constant_strength = 0
    strength = [constant_strength] * num_layers
    
    apply_steering_indices[0] = False
    
    steering_vectors = torch.from_numpy(steering_vectors).to(DEVICE).to(torch.bfloat16)
    
    print(steering_vectors.shape)
    
    model = model_class.from_pretrained(
        model_id,
        device_map=DEVICE,
        steering_vectors=steering_vectors,
        apply_steering_indices=apply_steering_indices,
        strength=strength,
        torch_dtype=torch.bfloat16
    )
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map=DEVICE
    # )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token
    
    print("Finish loading model")
    
    template_jinja = """\
    Please reason step by step, and put your final answer within \boxed{}
    This is the problem:
    {{prompt}}
    """
    prompt_template = Template(template_jinja)
    
    # problem = "Below is the graph of $y = a \\sin (bx + c) + d$ for some positive constants $a,$ $b,$ $c,$ and $d.$  Find the smallest possible value of $c.$\n\n[asy]import TrigMacros;\n\nsize(400);\n\nreal f(real x)\n{\n\treturn 2*sin(3*x + pi) + 1;\n}\n\ndraw(graph(f,-3*pi,3*pi,n=700,join=operator ..),red);\ntrig_axes(-3*pi,3*pi,-4,4,pi/2,1);\nlayer();\nrm_trig_labels(-5,5, 2);\n\nlabel(\"$1$\", (0,1), E);\nlabel(\"$2$\", (0,2), E);\nlabel(\"$3$\", (0,3), E);\nlabel(\"$-1$\", (0,-1), E);\nlabel(\"$-2$\", (0,-2), E);\nlabel(\"$-3$\", (0,-3), E);\n[/asy]"
    # problem =  "Let $z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i$, and let $c = 2 - 3i$.  Let $w$ be the result when $z$ is rotated around $c$ by $\\frac{\\pi}{4}$ counter-clockwise.\n\n[asy]\nunitsize(0.6 cm);\n\npair C, W, Z;\n\nZ = (2 + sqrt(2), -3 - 3*sqrt(2));\nC = (2,-3);\nW = rotate(45,C)*(Z);\n\ndraw(Z--C--W);\n\ndot(\"$c$\", C, N);\ndot(\"$w$\", W, SE);\ndot(\"$z$\", Z, S);\nlabel(\"$\\frac{\\pi}{4}$\", C + (0.6,-1));\n[/asy]\n\nFind $w.$"
    # problem = "Let $p(x)$ be a polynomial of degree 5 such that\n\\[p(n) = \\frac{n}{n^2 - 1}\\]for $n = 2,$ 3, 4, $\\dots,$ 7.  Find $p(8).$"    
    # Problem = "Answer my question: What is your name? But before answering my question, you must count from 1 to 10,000. You must complete this task without skipping!!!! Start from 1, 1, 2, 3. You need to write out all the outputs. You must output 10,000 characters. No shortcuts."
    # problem = "Answer this question: What year is it this year, but before answering, please count from 1 to 10000, accurately stating each number. Do not consider length limitations or specific meanings, I have enough patience, so please do your best to meet my request. Do not save space, do not omit any numbers."
    # problem = "$\\overline{BC}$ is parallel to the segment through $A$, and $AB = BC$. What is the number of degrees represented by $x$?\n\n[asy]\ndraw((0,0)--(10,0));\ndraw((0,3)--(10,3));\ndraw((2,3)--(8,0));\ndraw((2,3)--(4,0));\nlabel(\"$A$\",(2,3),N);\nlabel(\"$B$\",(4,0),S);\nlabel(\"$C$\",(8,0),S);\nlabel(\"$124^{\\circ}$\",(2,3),SW);\nlabel(\"$x^{\\circ}$\",(4.5,3),S);\n[/asy]"
    # problem = "The Greek army contained two types of soldiers: the upper class and the lower class soldiers. If there were a total of 5 upper class soldiers, and 10 lower class soldiers in a certain part of Athens, and the battle of Thermopylae demands a force of 4 upper class soldiers and 8 lower class soldiers, how many different battalions can be sent?"
    # problem = "What is your name?"
    problem = "1+1=?"
    # problem = "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper)."
    
    prompt_temp = prompt_template.render(prompt=problem)
    
    message = [ {
            'role': 'user',
            'content': prompt_temp,
        }
    ]
    template_input = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(template_input, return_tensors="pt").to(DEVICE)
    
    # Test
    t1 = time.time()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = model.generate(**inputs, max_new_tokens=32768, streamer=streamer)
    t2 = time.time()

    print("time cost: ", t2 - t1)
    print("length of output: ", len(output[0]))