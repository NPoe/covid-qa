import json
import re
import string
import torch

class QADataset:
    def __init__(self):
        self.contexts = []

    def add_context(self, context):
        self.contexts.append(context)

    @property
    def questions(self):
        questions = []
        for context in self.contexts:
            questions.extend(context.questions)
        return questions
    
    @property
    def answers(self):
        answers = []
        for context in self.contexts:
            answers.extend(context.answers)
        return answers

    @classmethod
    def from_file(cls, infile, load_answers = False):
        dataset = cls()

        with open(infile) as handle:
            for datapoint_i, datapoint in enumerate(json.load(handle)["data"]):
                for paragraph_i, paragraph in enumerate(datapoint["paragraphs"]):
                    context = QAContext(\
                        uid = paragraph.get("document_id", 
                            f"{datapoint_i}_{paragraph_i}"),
                        text = paragraph["context"])
    
                    for qa in paragraph["qas"]:
                        question = QAQuestion(\
                            uid = str(qa["id"]),
                            text = qa["question"],
                            context = context)

                        if load_answers and "answers" in qa:
                            for ans_i, ans in enumerate(qa["answers"]):
                                answer = QAAnswer(\
                                    uid = str(question.uid) + "." + str(ans_i),
                                    text = ans["text"],
                                    start = ans["answer_start"],
                                    question = question)

                                question.add_answer(answer)
                        context.add_question(question)
                    dataset.add_context(context)
        return dataset


class QAItem:
    def __init__(self, uid, text):
        self.uid = uid
        self.text = text
        self.tokens = {}
        self.token2start = {}
        self.token2end = {}

    def __hash__(self):
        return id(self)

    def add_tokenizer(self, tokenizer):
        self.tokens[id(tokenizer)] = []
        self.token2start[id(tokenizer)] = []
        self.token2end[id(tokenizer)] = []
        
        text_pos = 0

        whitespace_rgx = re.compile("([" + string.whitespace + "]+)")
        punct_rgx = re.compile("([" + string.punctuation + "]+)")

        for whitespace_token in whitespace_rgx.split(self.text):
            if whitespace_rgx.match(whitespace_token) or len(whitespace_token) == 0:
                text_pos += len(whitespace_token)
                
            else:
                for token in punct_rgx.split(whitespace_token):
                    if len(token) == 0: continue
                    text_pos += len(token)

                    wordpieces = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                    self.tokens[id(tokenizer)].extend(wordpieces)
                    self.token2start[id(tokenizer)].extend([text_pos - len(token)] * len(wordpieces))
                    self.token2end[id(tokenizer)].extend([text_pos] * len(wordpieces))

        self.token2start[id(tokenizer)] = torch.tensor(self.token2start[id(tokenizer)]) 
        self.token2end[id(tokenizer)] = torch.tensor(self.token2end[id(tokenizer)]) 

        assert text_pos == len(self.text)
        assert len(self.text[text_pos:].strip()) == 0


class QAContext(QAItem):
    def __init__(self, uid, text):
        super(QAContext, self).__init__(uid = uid, text = text)
        self.questions = []
        
    def add_question(self, question):
        self.questions.append(question)
    
    @property
    def answers(self):
        answers = []
        for question in self.questions:
            answers.extend(question.answers)
        return answers

class QAQuestion(QAItem):
    def __init__(self, uid, text, context):
        super(QAQuestion, self).__init__(uid = uid, text = text)
        self.context = context
        self.answers = []
    
    def add_answer(self, answer):
        self.answers.append(answer)

    def slice_context(self, tokenizer, active_portion = 0.5):
        max_length = min(512, tokenizer.max_len)

        if not id(tokenizer) in self.tokens:
           self.add_tokenizer(tokenizer) 
        
        if not id(tokenizer) in self.context.tokens:
           self.context.add_tokenizer(tokenizer) 
        
        dummy_input = [100, 200, 300]
        num_special_tokens = len(tokenizer.encode(dummy_input, dummy_input)) - len(dummy_input) * 2
        
        slice_len = int(active_portion * (max_length - len(self.tokens[id(tokenizer)]) - num_special_tokens))
        
        samples = []
        active_slices = []

        for slice_start in range(0, len(self.context.tokens[id(tokenizer)]), slice_len):
            slice_end = slice_start + slice_len

            context_active = self.context.tokens[id(tokenizer)][slice_start:slice_end]
            context_left_rev = []
            context_right = []

            for offset in range(len(self.context.tokens[id(tokenizer)])):
                if slice_end + offset >= len(self.context.tokens[id(tokenizer)]) and slice_start - offset < 0:
                    break
                
                if len(context_active) + len(context_left_rev) + len(context_right) + len(self.tokens[id(tokenizer)]) + \
                        num_special_tokens == max_length:
                    break

                if slice_end + offset < len(self.context.tokens[id(tokenizer)]):
                    context_right.append(self.context.tokens[id(tokenizer)][slice_end + offset])
                
                if len(context_active) + len(context_left_rev) + len(context_right) + len(self.tokens[id(tokenizer)]) + \
                        num_special_tokens == max_length:
                    break

                if slice_start - offset - 1 >= 0:
                    context_left_rev.append(self.context.tokens[id(tokenizer)][slice_start - offset - 1])

            marker = -999
            context_full = list(reversed(context_left_rev)) + [marker] + context_active + context_right
            
            if tokenizer.padding_side == "right":
                first, second = self.tokens[id(tokenizer)], context_full
            elif tokenizer.padding_side == "left":
                first, second = context_full, self.tokens[id(tokenizer)]

            sample = tokenizer.encode(
                    first, second, 
                    max_length = max_length+1, 
                    truncation_strategy = "do_not_truncate",
                    pad_to_max_length = True)

            active_span_start = sample.index(marker)
            active_span_end = active_span_start + len(context_active)
            sample = sample[:active_span_start] + sample[active_span_start + 1:]

            assert len(sample) == max_length
            assert tuple(sample[active_span_start:active_span_end]) == tuple(context_active)

            samples.append(sample)
            active_slices.append(slice(active_span_start, active_span_end))
            
        return samples, active_slices

class QAAnswer(QAItem):
    def __init__(self, uid, text, start, question):
        super(QAAnswer, self).__init__(uid = uid, text = text)
        self.start = start
        self.question = question

    @property
    def context(self):
        return self.question.context
