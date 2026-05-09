from transformers import AutoProcessor, AutoModelForCausalLM
from langchain.messages import (
    HumanMessage,
    SystemMessage,
)
from torch._C import dtype

#MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
#MODEL_ID = "mistralai/Codestral-22B-v0.1"
MODEL_ID = "frameai/CodeLoxa-4B"
class HuggingChat:
    def __init__(self,
                 model_id=MODEL_ID,
                 dtype='auto',
                 device_map='auto'):
        self.device_map = device_map
        self.model_id = model_id
        self.dtype = dtype


    def get_module_and_processor(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=self.dtype, device_map=self.device_map)
        model.to(dtype=self.dtype)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        return model, processor

    def execute_query(self, sys_msg, user_msg):
        model, processor = self.get_module_and_processor()
        messages = [
            SystemMessage(
                content=sys_msg
            ),
            HumanMessage(
                content=user_msg
            ),
        ]

        # Process input
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = model.generate(**inputs, max_new_tokens=1024)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse output
        ai_msg = processor.parse_response(response)
        return ai_msg