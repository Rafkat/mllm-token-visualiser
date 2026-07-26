from transformers import AutoProcessor

model_path = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hey there!"},
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
        ],
    }
]


def calculate_total_tokens(msgs, prc, add_generation_prompt=False):
    input_data = prc.apply_chat_template(
        msgs,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    return (
        input_data["input_ids"][0].shape[0],
        input_data["input_ids"][0],
        input_data["attention_mask"][0],
    )


def calculate_text_tokens(msgs, prc):
    total_count = 0
    for msg in msgs:
        for content in msg["content"]:
            if content["type"] == "text":
                tokens = prc.tokenizer.encode(content["text"], add_special_tokens=False)
                total_count += len(tokens)
    return total_count


def calculate_image_tokens(input_ids, attention_mask, prc):
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    image_token_id = prc.tokenizer.convert_tokens_to_ids(image_token)

    image_tokens = ((input_ids == image_token_id) & attention_mask.bool()).sum(dim=-1)
    return image_tokens.item()


def calculate_video_tokens(input_ids, attention_mask, prc):
    video_token = getattr(processor, "video_token", "<|video_pad|>")
    video_token_id = prc.tokenizer.convert_tokens_to_ids(video_token)

    video_tokens = (input_ids == video_token_id) & attention_mask.bool()
    return video_tokens.item()


total_tokens, input_ids, attention_mask = calculate_total_tokens(messages, processor)
text_tokens = calculate_text_tokens(messages, processor)
image_tokens = calculate_image_tokens(input_ids, attention_mask, processor)


print("-" * 50)
print("Total tokens: ", total_tokens)
print("-" * 50)
print("Text tokens: ", text_tokens)
print("-" * 50)
print("Image tokens: ", image_tokens)
print("-" * 50)
print("Special tokens: ", total_tokens - image_tokens - text_tokens)
print("-" * 50)
