from llama_cpp import Llama
from transformers import AutoTokenizer
from retrieval.retrieval_qdrant import retrieval_qdrant

def rag_ans(rag_func, question):
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Llama(
        model_path='models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf',
        n_ctx=512,
        n_gpu_layers=-1       # Number of model layers to offload to GPU
    )


    if rag_func == True:
        retrieval_result = retrieval_qdrant(question)
        PROMPT = f'''
        당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
        You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.

        문맥에 근거하여 알맞게 답변을 해주세요.
        Please answer appropriately based on the context.

        문맥은 다음과 같습니다.
        The context is as follows.
        1. {retrieval_result[0]}
        2. {retrieval_result[1]}
        3. {retrieval_result[2]}
        '''
    else:
        PROMPT = f'''
        당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
        You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.
        '''

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{question}"}
        ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize = False,
        add_generation_prompt=True
    )

    generation_kwargs = {
        "max_tokens":512,
        "stop":["<|eot_id|>"],
        "top_p":0.9,
        "temperature":0.6,
        "echo":True, # Echo the prompt in the output
    }

    resonse_msg = model(prompt, **generation_kwargs)
    return resonse_msg['choices'][0]['text'][len(prompt):]