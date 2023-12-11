import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import random
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser(description='Fine-tune a seq2seq model.')
parser.add_argument('--train_epochs', type=int, default=1, help='Number of epochs to train for.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--num_tokens', type=int, default=32, help='Number of tokens to use.')
parser.add_argument('--encoder_layers', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6], help='Encoder layers to use.')
parser.add_argument('--decoder_layers', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6], help='Decoder layers to use.')
parser.add_argument('--file', type=str, default="merged.csv", help='File to train on.')
parser.add_argument('--device', type=str, default='cuda', help='Device to train on.')
args = parser.parse_args()

train_epochs = args.train_epochs
batch_size = args.batch_size
num_tokens = args.num_tokens
encoder_layers = args.encoder_layers
decoder_layers = args.decoder_layers
file = args.file
if args.device == 'cuda':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cpu')


df = pd.read_csv(file)
df = df.dropna(subset=['text'])


def get_fragmented_df(df, batch_size):
    unique_contexts = df['context'].unique()
    
    fragmented_df = pd.DataFrame(columns=df.columns)

    while len(unique_contexts) > 0:
        selected_context = random.choice(unique_contexts)
        selected_rows = df[df['context'] == selected_context]
        batch_size = min(batch_size, len(selected_rows))

        selected_texts = selected_rows.sample(n=batch_size)
        fragmented_df = pd.concat([fragmented_df, selected_texts], ignore_index=True)
        df = df.drop(selected_texts.index, inplace=False)

        unique_contexts = df['context'].unique()

    return fragmented_df

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

print("Fragmenting dataset...")
df = get_fragmented_df(df, batch_size)

encode_tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
decode_tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")

tokenized_text = torch.empty((len(df), num_tokens), dtype=torch.long)
tokenized_context = torch.empty((len(df), num_tokens), dtype=torch.long)

print("Tokenizing...")
for i in range(len(df)):
    tokenized_text[i] = encode_tokenizer(df.iloc[i]["text"],\
                                            max_length=num_tokens,\
                                            truncation=True,\
                                            padding='max_length',\
                                            return_tensors="pt").input_ids
    tokenized_text[i] = tokenized_text[i].squeeze()
    tokenized_context[i] = encode_tokenizer(df.iloc[i]['context'],\
                                            max_length=num_tokens,\
                                            truncation=True,\
                                            padding='max_length',\
                                            return_tensors="pt").input_ids
    tokenized_context[i] = tokenized_context[i].squeeze()

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

class contextLLM(nn.Module):
    def __init__(self):
        super(contextLLM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids, context_ids):
        encoder_outputs = self.encoder(input_ids)
        decoder_outputs = self.decoder(context_ids, encoder_outputs)
        return decoder_outputs

llm = contextLLM().to(DEVICE)

optimizer = torch.optim.Adam(llm.parameters(), lr=0.001)

print("Training...")
for epoch in range(train_epochs):
    print(f"Epoch {epoch}")
    running_loss = 0.0
    for i in range(0, len(df), batch_size):
        optimizer.zero_grad()
        x = tokenized_text[i:i+batch_size].to(DEVICE)
        y = tokenized_text[i+1:i+batch_size+1].to(DEVICE)
        y = nn.functional.one_hot(y, num_classes=encoder.encoder.config.vocab_size)
        context = tokenized_context[i:i+batch_size].to(DEVICE)

        output = llm(x, context)[0]
        if output.shape != y.shape:
            continue
        loss = nn.functional.cross_entropy(input=output.float(), target=y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch} loss: {running_loss/100}")
    if epoch % 2 == 0:
        torch.save(llm.state_dict(), f"llm_ckpt{epoch}.pt")

    print("Epoch Test:")
    random_input = random.randint(0, len(tokenized_text))
    input_text = df.iloc[random_input]["text"]
    input_context = df.iloc[random_input]["context"]
    print("Input: ", input_text)
    print("Context: ", input_context)
    print("Output: ", generate(input_text, input_context))
    print("\n----------------\n")

torch.save(llm.state_dict(), "llm_final.pt")
random_input = random.randint(0, len(tokenized_text))
input_text = df.iloc[random_input]["text"]
input_context = df.iloc[random_input]["context"]
print("Final Test:")
print("Input: ", input_text)
print("Context: ", input_context)
print("Output: ", generate(input_text, input_context))
print("\n----------------\n")