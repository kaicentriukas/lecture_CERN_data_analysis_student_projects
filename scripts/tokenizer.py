# tokenizer.py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_char_tokenizer(latex_texts, num_chars=5000): #Creates a list of numbers representing each word in the text
    """
    Builds a character-level tokenizer for LaTeX formulas.
    Every individual character becomes one token:
    '\', '{', '}', 'a', 'b', '^', '_', etc.
    """
    tok = Tokenizer(num_words=num_chars, filters="", lower=False, char_level = True)
    tok.fit_on_texts(latex_texts) 
    return tok 


def texts_to_sequences(tok, texts, max_len=150):
    sequences = [] #Will hold all encoded formulas 

    # Convert each text into a list of token indices
    for text in texts: 
        seq = tok.texts_to_sequences([text]) # returns [[numbers]]
        seq = seq[0] #To get [numbers]
        sequences.append(seq)

        # Padding the sequences to the same length
        padded_sequences = pad_sequences(sequences, maxlen = max_len, padding = 'post') 
        #this adds zeroes at the end, so the shapes of all the sequences are the same and therefore the model can train

        return padded_sequences

        

def sequence_to_text(tok, seq):
    index_to_char = {}

    # Reverse tokenizer dictionary
    for ch, idx in tok.word_index.items():
        index_to_char[idx] = ch

    chars = []

    for number in seq:
        if number == 0:
            continue #skip padding (means nothing)
        if number in index_to_char:
            chars.append(index_to_char[number])
        else:
            chars.append('') #unknown token 
    
    return ''.join(chars) #no spaces dude
