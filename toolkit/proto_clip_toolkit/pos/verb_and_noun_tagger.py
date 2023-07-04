from flair.data import Sentence
from flair.models import SequenceTagger

class VerbAndNounTagger:
    def __init__(self, verb_dictionary_path, noun_dictionary_path):

        """Load the flair sequence tagger and the allowed verb and noun dictionary."""
        self.tagger = SequenceTagger.load("flair/pos-english")

        verb_file = open(verb_dictionary_path, "r")
        verb_list = verb_file.readlines()
        self.allowed_verb_set = set([x.strip("\n") for x in verb_list])


        noun_file = open(noun_dictionary_path, "r")
        noun_list = noun_file.readlines()
        self.allowed_noun_set = set([x.replace("_", " ").strip("\n") for x in noun_list])
    
    def tag_sentence(self, text):
        """Tag a sentence with its POS and return the noun and verb from the sentence if they are present in the dictionary."""
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        
        word_tag_list = []

        for entity in sentence.get_labels():
            # Uncomment this if you want to debug the POS tag for each word you speak.
            # print(entity)
            word = entity.shortstring.split("/")[0].strip("\"").lower()
            word_tag_list.append((word, entity.value))
        
        return word_tag_list

    def find_valid_noun_and_verb(self, text):
        """Tag a sentence with its POS and return the noun and verb from the sentence if they are present in the dictionary."""
        word_tag_list = self.tag_sentence(text)
        parsed_verb = None
        parsed_noun = None
        
        idx = 0
        while idx < len(word_tag_list):
            curr_word, curr_tag = word_tag_list[idx]

            # For nouns(objects) like mustard bottle, we need to detect both the NN's together. 
            # Therefore, we move ahead of the array and combine subsequences with same tags into a single word.
            while idx + 1 < len(word_tag_list) and curr_tag==word_tag_list[idx+1][1]:
                curr_word += " " + word_tag_list[idx+1][0]
                idx += 1

            if (curr_tag=="NN" or curr_tag=="NNP" or curr_tag=="NNS") and curr_word in self.allowed_noun_set:
                parsed_noun = curr_word
            if curr_tag=="VB" and curr_word in self.allowed_verb_set:
                parsed_verb = curr_word

            idx += 1
        
        return parsed_verb, parsed_noun