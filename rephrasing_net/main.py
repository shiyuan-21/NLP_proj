import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# model_name = "./Llama-2-7b-chat-hf"
# adapters_name = "rephrasing_net/7b_modified/checkpoint-400"

# # model_name = "meta-llama/Llama-2-13b-hf"
# # adapters_name = "results_modified/checkpoint-200"

# print(f"Starting to load the model {model_name} into memory")

# m = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     #load_in_4bit=True,
#     torch_dtype=torch.bfloat16,
#     device_map={"": 0}
# )

# m = PeftModel.from_pretrained(m, adapters_name)
# m = m.merge_and_unload()
# tok = LlamaTokenizer.from_pretrained(model_name)
# tok.bos_token_id = 1

# stop_token_ids = [0]

# print(f"Successfully loaded the model {model_name} into memory")

# # while 1:
# #     # Generate tokens

# #     user_prompt = input()
# #     system_prompt = "You need to rephrase user input so they are more detailed, more formal and fit in a resume. You must not make up anythong on your own. You may only output the rephrased input and nothing else"
# #     prompt = "[INST] <<SYS>>\n" + system_prompt + "<</SYS>>" + user_prompt + "[/INST]"
# #     inputs = tok(prompt, return_tensors="pt")

# #     print(inputs)

# #     output = m.generate(inputs.input_ids.cuda())

# #     print("Generated tokens:")
# #     print(tok.decode(output[0], skip_special_tokens=True))

# system_prompt = {"role": "system", "content": "please ask the user questions regarding their personal information, education, project and publication until you gathered enough information, then compose a resume for the user"}

# chat = [
#   system_prompt,
# #   {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]

# while True:
#     user_prompt = input()
#     chat.append({"role": "user", "content": user_prompt})
#     inputs = tok(tok.apply_chat_template(chat, tokenize=False), return_tensors="pt")
#     # print(inputs)

#     output = m.generate(inputs.input_ids.cuda(), repetition_penalty=1.3)
#     print("generated tokens:")
#     output = tok.decode(output[0], skip_special_tokens=True)
#     output = output.split("[/INST]")[-1]
#     print(output)
#     chat.append({"role": "assistant", "content": output})
#     # chat.append(system_prompt)

class RephrasingNet:
    def __init__(self,model_name = "./Llama-2-7b-chat-hf",adapters_name = "rephrasing_net/7b_modified/checkpoint-400") -> None:
        #model_name = "./Llama-2-7b-chat-hf"
        #adapters_name = "rephrasing_net/7b_modified/checkpoint-400"
        #adapters_name = "rephrasing_net/7b_modified-2/checkpoint-10000"
        # model_name = "meta-llama/Llama-2-13b-hf"
        # adapters_name = "rephrasing_net/results_modified/checkpoint-200"

        # model_name = "meta-llama/Llama-2-13b-hf"
        # adapters_name = "results_modified/checkpoint-200"

        print(f"Starting to load the model {model_name} into memory")

        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            #load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )

        m = PeftModel.from_pretrained(m, adapters_name)
        m = m.merge_and_unload()
        tok = LlamaTokenizer.from_pretrained(model_name)
        tok.bos_token_id = 1

        stop_token_ids = [0]

        print(f"Successfully loaded the model {model_name} into memory")
        self.m = m
        self.tok = tok

    def rephrase(self, user_prompt: str):
        # chat = [{"role": "system", "content": "please rephrase the user's input to be more detailed and better fit into a resume. Output ONLY the rephrased sentence"}]
        # chat.append({"role": "user", "content": user_prompt})
        sys_prompt = "please rephrase the following sentence to be more comprehensive. Output ONLY the rephrased sentence using information given in the sentence. If you are unsure, output nothing, do not converse with the user: "
        chat = sys_prompt + '"' + user_prompt + '"'
        # inputs = self.tok(self.tok.apply_chat_template(chat, tokenize=False), return_tensors="pt")
        inputs = self.tok(chat, return_tensors="pt")
        print(inputs)
        print(self.tok.decode(inputs.input_ids.view(-1)))

        output = self.m.generate(inputs.input_ids.cuda(), repetition_penalty=1.3)
        # print("generated tokens:")
        output = self.tok.decode(output[0], skip_special_tokens=True)
        output = output.split('"')[-1]
        # output = output.split("[/INST]")[-1]
        # print(output)
        return output