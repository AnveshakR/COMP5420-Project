from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import random
import argparse

parser = argparse.ArgumentParser(description='Generate an output using the seq2seq model checkpoint.')
parser.add_argument('--num_tokens', type=int, default=32, help='Number of tokens to use.')
parser.add_argument('--ckpt', type=str, default="llm_final.pt", help='Checkpoint to use.')
parser.add_argument('--device', type=str, default='cuda', help='Device to train on.')
parser.add_argument('--encoder_layers', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help='Encoder layers to use.')
parser.add_argument('--decoder_layers', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help='Decoder layers to use.')
args = parser.parse_args()

num_tokens = args.num_tokens
ckpt = args.ckpt
encoder_layers = args.encoder_layers
decoder_layers = args.decoder_layers
if args.device == 'cuda':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cpu')

print("Checkpoint selected: ", ckpt)

encode_tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
decode_tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")

print("Loading encoder...")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de")
encoder = model.get_encoder()
encoder.encoder.layer = nn.ModuleList([encoder.encoder.layer[i] for i in encoder_layers])
print("encoder layers: ", len(encoder.encoder.layer))

print("Loading decoder...")
model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
decoder = model.get_decoder()
decoder.bert.encoder.layer = nn.ModuleList([decoder.bert.encoder.layer[i] for i in decoder_layers])
print("decoder layers: ", len(decoder.bert.encoder.layer))

def generate(input, context, generation_length=100):
    input_ids = encode_tokenizer(input, return_tensors="pt").input_ids.to(DEVICE)
    context_ids = encode_tokenizer(context, return_tensors="pt").input_ids.to(DEVICE)
    

    while len(input_ids) < generation_length:
        try:
            output = llm(input_ids[-32:], context_ids)[0]
        except:
            return decode_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        next_token_logits = output[:, -1, :]
        next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    return decode_tokenizer.decode(input_ids[0])

class contextLLM(nn.Module):
    def __init__(self):
        super(contextLLM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids, context_ids):
        encoder_outputs = self.encoder(input_ids)
        decoder_outputs = self.decoder(context_ids, encoder_outputs)
        return decoder_outputs
    
print("Loading model...")
llm = contextLLM()
llm.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)

cont = 'y'
while cont == 'y':
    input_text = input("Enter input text: ")
    context_text = input("Enter context text: ")
    print("Generating...")
    print(generate(input_text, context_text, 1000))
    print("\n----------------------------------------\n")
    cont = input("Continue? (y/n): ")