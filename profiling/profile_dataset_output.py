from argparse import Namespace
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
sys.path.insert(1, os.path.dirname(os.getcwd()))
from utils import populate_dataset, format_problem_and_options, get_first_user_msg

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", torch_dtype="auto", device_map="auto")

datasets = ["aime", "math", "gpqa", "mmlu", "humaneval"]
for dataset_name in datasets:
    args = Namespace(dataset_name=dataset_name)
    dataset=populate_dataset(args)
    
    problem_ids = [0, 1, 2, 3, 4, 5, 6]
    output_lengths = []
    for problem_id in problem_ids:
        raw_data = format_problem_and_options(args, problem_id)
        # print(f"raw_data:\n{raw_data}\n\n")

        messages = [
            {"role": "user", "content": get_first_user_msg(args, raw_data)},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print(f"text:\n{text}\n\n")

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # print(f"model_inputs:\n{model_inputs}\n\n")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
        # print(f"generated_ids:\n{generated_ids}\n\n")

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(f"output:\n{output}\n\n")
        output_lengths.append(len(generated_ids[0])-len(model_inputs["input_ids"][0]))

    print(f"Dataset: {dataset_name}")
    print(f"Output lengths: {output_lengths}")
    print(f"Average output length: {sum(output_lengths) / len(output_lengths)}")
    print("--------------------------------")