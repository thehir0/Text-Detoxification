import torch
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = AutoModelForSeq2SeqLM.from_pretrained('models/pegasus-best').to(device)
tokenizer = AutoTokenizer.from_pretrained('models/pegasus-best')

model.eval()

def inference(inference_request):
    input_ids = tokenizer(inference_request, return_tensors="pt", padding=True).input_ids.to(device)
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True,temperature=0)



def main(input):
    print(inference(input))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for making predictions.")
    
    parser.add_argument("--input", help="Input file for predictions", required=True)
    
    args = parser.parse_args()
    
    main(args.input)
