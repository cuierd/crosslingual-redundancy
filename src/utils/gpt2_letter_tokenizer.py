from transformers import AutoTokenizer, GPT2Tokenizer, BertTokenizer
import string

# class CustomGPT2Tokenizer(GPT2Tokenizer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def tokenize(self, text, *args, **kwargs):
#         # Use GPT-2 tokenizer
#         tokens = super().tokenize(text, *args, **kwargs)

#         # Post-process to get character-level tokens
#         char_tokens = list("".join(tokens))
#         return char_tokens

class CustomGPT2Tokenizer(GPT2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text, *args, **kwargs):
        word_tokens = []
        words = text.strip(" ") 
        token = super().tokenize(words)
        # print(words, ":::", token)
        if token:
            if len(token) > 1:
                word_tokens.append("ĠunkĠ")
            else:
                word_tokens.append(token[-1])
        return word_tokens

'''class mGPTTokenizer(GPT2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text, *args, **kwargs):
        # Use mGPT tokenizer
        tokens = super().tokenize(text, *args, **kwargs)
    
        # Post-process to get letter-by-letter tokens
        char_tokens = list("".join(tokens))
        return char_tokens

class mBERTTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text, *args, **kwargs):
        # Use mBERT tokenizer
        tokens = super().tokenize(text, *args, **kwargs)

        # Post-process to get character-level tokens
        char_tokens = list("".join(tokens))
        return char_tokens'''

class mGPTTokenizer(GPT2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text, *args, **kwargs):
        char_tokens = []
        for i, letter in enumerate(text):
            if letter == ' ':
                continue
            elif text[i-1]==' ':
                char_tokens.append('Ġ'+super().tokenize(letter)[0])
            else:
                char_tokens.append(super().tokenize(letter)[0])
        return char_tokens
    

class CustomBERTTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    def tokenize(self, text, *args, **kwargs):
        word_tokens = []
        words = text.split(" ")
        for w in words:
            tokens = self.tokenizer.tokenize(w)
            # if the word is split into multiple tokens, replace with UNK. Change the condition if you want to keep the sub-tokens
            if len(tokens) > 1:
                word_tokens.append("UNK")
            else:
                word_tokens.append(tokens[-1])
        return word_tokens


# class CustomBERTTokenizer(BertTokenizer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

#     def tokenize(self, text, *args, **kwargs):
#         word_tokens = []
#         words = text.split(" ")
#         print(text)
#         print(words)
#         for w in words:
#             tokens = self.tokenizer.tokenize(w)
#             if len(tokens) == 1 :
#                 print("append", tokens)
#                 word_tokens.append(tokens[-1])
#             else:
#                 print("append unk")
#                 word_tokens.append("UNK")
#         return word_tokens



class mBERTTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, text, *args, **kwargs):
        char_tokens = []
        for i, letter in enumerate(text):
            if letter == ' ':
                continue
            elif i == 0 or text[i-1]==' ' or letter in string.punctuation:
                char_tokens.append(letter)
            else:
                char_tokens.append('##'+letter)

        return char_tokens