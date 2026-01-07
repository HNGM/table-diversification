from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16", device_map="auto", trust_remote_code=True, load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def send_request(self, request) -> str:
        text = self.tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=request.get("max_new_tokens", 512), do_sample=False, num_beams=1)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response




