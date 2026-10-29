import re
import json
import torch
import logging
from typing import Dict, Any, List, Union
from transformers import MllamaForConditionalGeneration, AutoProcessor

class SatelliteVisionAgent:
    """
    4-bit quantized Vision-Language Model (Llama-3.2-Vision) 
    Optimized for Satellite Image Analysis on a 24GB GPU.
    """
    def __init__(self, model_id: str, system_message: str):
        self.logger = logging.getLogger("SatVLM")
        self.system_message = system_message
        
        self.logger.info(f"Booting {model_id} into VRAM (4-bit mode)...")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map={"": 0} # Specifically locks to the A5000
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def prepare_inputs(self, image_path: str, prompt: str) -> dict:
        """
        Processes image and text into model-ready tensors.
        Fixes Python 3.8 PIL typing crash and Llama-3.2 System Message bug.
        """
        # Late import to prevent typing conflicts on older Python versions
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Llama-3.2-Vision Fix: Merge instructions into 'user' role for multimodal compatibility
        combined_prompt = f"{self.system_message}\n\n[Satellite IQA Context & Task]: {prompt}"
        
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": combined_prompt}
                ]
            }
        ]
        
        formatted_prompt = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            image, 
            formatted_prompt, 
            return_tensors="pt"
        ).to(self.model.device)
        
        return inputs

    @torch.no_grad()
    def generate_json(self, inputs: dict, max_new_tokens: int) -> Dict[str, Any]:
        """
        Generates text, slices off the echoed prompt, and uses a robust 
        string-slicing method to extract and clean the JSON block.
        """
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # ISOLATE THE NEW TEXT: Slice off the prompt using the input length
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        response_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        
        # Print exactly what the model generated to the terminal for debugging
        self.logger.info(f"DEBUG - NEW VLM TEXT ONLY:\n{response_text}")

        # Find the outermost curly braces in the newly generated text
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        
        if start != -1 and end != 0:
            json_str = response_text[start:end]
            
            # Clean common LLM formatting errors
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*}', '}', json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON Parse Error: {e} | Extracted Content: {json_str}")
                return {}
                
        self.logger.warning("VLM failed to return a valid JSON structure.")
        return {}
