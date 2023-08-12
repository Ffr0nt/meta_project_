import torch
from nltk import tokenize
import torch
from transformers import BertTokenizer, AdamW, AutoModel


class BertBlockEmbedding:

    def __init__(
            self,
            pretrained_model: str,
            lr=2e-5,
            eps=1e-8,
            bert_output_size=64
    ) -> None:

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model,
            do_lower_case=True
        )

        self.model = AutoModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True,
            output_attentions=False,  # Whether the model returns attentions weights.
        )

        # Freeze all layers except for the last one
        for name, param in self.model.named_parameters():
            if name.startswith('encoder.layer.11'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            eps=eps,
            no_deprecation_warning=True
        )

        self.bert_output_size = bert_output_size

    def _prepare_text(self, text: str, max_length=64):
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in text:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                truncation=True,
                max_length=max_length,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def forward(self, text):

        prepared_text = self._prepare_text(text)

        b_input_ids = prepared_text[0]  # .to(self.device)
        b_input_mask = prepared_text[1]  # .to(self.device)

        self.model.train()

        self.model.zero_grad()

        hidden_states = self.model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   )

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # The pooler_output tensor contains a fixed-length embedding for the whole input text.
        token_vecs = hidden_states.pooler_output

        text_embedding = torch.mean(token_vecs, dim=0)

        self.optimizer.step()

        return text_embedding[:self.bert_output_size]
