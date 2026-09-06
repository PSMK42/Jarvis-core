from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading model weights...")
AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Download complete!")