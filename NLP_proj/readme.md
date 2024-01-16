This is the final project of our 2023 NLP class
Firstly, please download Llama-2-7b-chat-hf and place the folder inside the "NLP_proj"
To train the model:
`
    cd rephrasing_net
    python train_7b.py
`
To test the model:
`
    python test.py --model_name "./Llama-2-7b-chat-hf" --adapters_name "rephrasing_net/7b_modified/checkpoint-400"
`
You could use your own trained or fine-tuned models