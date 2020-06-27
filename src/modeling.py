import torch
from transformers import BertForQuestionAnswering, BertTokenizer

def infer_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def infer_batch_size(device):
    try:
        cuda_properties = torch.cuda.get_device_properties(device)
    except ValueError: # if device is not a cuda device
        cuda_properties = None
   
    # We have run our experiments with batch size 128, on a GPU with total memory 11.7MB
    if cuda_properties is not None:
        return 128 * cuda_properties.total_memory // 11721506816

    return 128


class QAModel:
    def __init__(self, args):
        
        self.num_nbest = args.num_nbest
        self.max_answer_chars = args.max_answer_chars

        self.model_device = infer_device() if args.model_device is None else args.model_device
        self.embedding_device = self.model_device if args.embedding_device is None else args.embedding_device
        self.batch_size = infer_batch_size(self.model_device) if args.batch_size is None else args.batch_size
        
        self.model = BertForQuestionAnswering.from_pretrained(args.pretrained_model_name_or_path) 
        self.model = self.model.eval().to(device = self.model_device)

        self.tokenizers = [BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)]
        
        if args.embeddingprefix is not None:
            if args.embedding_device is None:
                self.embedding_device = "cpu"
            
            self.model.get_input_embeddings().to(device = self.embedding_device)

            new_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
            self.tokenizers.append(new_tokenizer)
            self.expand_embedding_layer(new_tokenizer, args.embeddingprefix)


    def expand_embedding_layer(self, tokenizer, embeddingprefix):
        previous_vocab_size = len(tokenizer)
        embedding = torch.load(embeddingprefix + "-vectors.pt")
    
        with open(embeddingprefix + "-vocab.txt") as handle:
            tokens = [token.strip() for token in handle]
    
        assert len(tokens) == embedding.shape[0]
    
        added = []
        for token_i, token in enumerate(tokens):
            if not token in tokenizer.vocab:
                tokenizer.vocab[token] = len(tokenizer.vocab)
                tokenizer.ids_to_tokens[tokenizer.vocab[token]] = token
                added.append(token_i)

        self.model.resize_token_embeddings(len(tokenizer))

        added = torch.tensor(added, dtype = torch.long)
        self.model.get_input_embeddings().weight.data[previous_vocab_size:,:] = embedding[added].to(device = self.embedding_device)


    def get_outputs_at_token_level(self, context, tokenizer):
        input_ids, active_slices, sample_i_to_question_i, sample_i_to_tokenizer_i = [], [], [], []
        outputs = [[] for _ in context.questions]
       
        for question_i, question in enumerate(context.questions):
            question_input_ids, question_active_slices = question.slice_context(tokenizer)

            input_ids.extend(question_input_ids)
            active_slices.extend(question_active_slices)
            sample_i_to_question_i.extend([question_i] * len(question_input_ids))

        input_ids = torch.tensor(input_ids, dtype = torch.long, device = self.embedding_device)
        attention_mask = (input_ids > 0).to(dtype = torch.long, device = self.model_device)
        token_type_ids = torch.ones_like(attention_mask, device = self.model_device)
        
        uniq_input_ids, uniq_input_ids_inverse = torch.unique(input_ids, return_inverse = True)
        uniq_inputs_embeds = self.model.get_input_embeddings()(uniq_input_ids)        
        uniq_input_ids_inverse = uniq_input_ids_inverse.to(device = self.model_device)
        uniq_inputs_embeds = uniq_inputs_embeds.to(device = self.model_device)
    
        for sample_i, question_i in enumerate(sample_i_to_question_i):
            question_tokens = context.questions[question_i].tokens[id(tokenizer)]
            token_type_ids[sample_i, :len(question_tokens) + 2] = 0

        for batch_start in range(0, input_ids.shape[0], self.batch_size):
            batch_slice = slice(batch_start, min(input_ids.shape[0], batch_start + self.batch_size))
            
            batch_inputs = {\
                "attention_mask": attention_mask[batch_slice],
                "token_type_ids": token_type_ids[batch_slice],
                "inputs_embeds": uniq_inputs_embeds[uniq_input_ids_inverse[batch_slice]]}
    
            batch_outputs = torch.stack(self.model(**batch_inputs)[:2], -1)
            for sample_i, active_slice in enumerate(active_slices[batch_slice]):
                question_i = sample_i_to_question_i[batch_start + sample_i]
                outputs[question_i].append(batch_outputs[sample_i, active_slice])
    
        for question_i in range(len(context.questions)):
            outputs[question_i] = torch.cat(outputs[question_i], 0)

        outputs = torch.stack(outputs, 0)
        return outputs.detach().to(device = "cpu")


    def get_outputs_at_word_level(self, context):
        for tokenizer_i, tokenizer in enumerate(self.tokenizers):
            outputs = self.get_outputs_at_token_level(context, tokenizer)
            
            if tokenizer_i == 0:
                outputs_word = torch.zeros((len(context.questions), len(context.word2start_char), len(self.tokenizers), 2)) - float("Inf")

            for token_i in range(outputs.shape[1]):
                word_i = context.token2word[id(tokenizer)][token_i]

                for dim_i in range(2):
                    outputs_word[:, word_i, tokenizer_i, dim_i] = torch.max(\
                            outputs_word[:, word_i, tokenizer_i, dim_i], 
                            outputs[:, token_i, dim_i])

        return outputs_word.mean(axis = 2) # (questions, word_len, 2)


    def get_best_scores_and_answers(self, context):
        scores_word = self.get_outputs_at_word_level(context)
        best_scores, best_answers = [], []

        for question_i, question in enumerate(context.questions):
            current_best_scores = [-float("Inf") for _ in range(self.num_nbest)]
            current_best_answers = [None for _ in range(self.num_nbest)]
        
            max_end_score = scores_word[question_i,:,1].max()
        
            for start in range(scores_word.shape[1]):
                start_score = scores_word[question_i, start, 0]
            
                if start_score + max_end_score > current_best_scores[-1]:
                    max_end = (context.word2start_char[start]-context.word2end_char[start:]+self.max_answer_chars >= 0).nonzero()[-1][0] + start

                    topk = torch.topk(scores_word[question_i, start:max_end+1, 1], k=min(self.num_nbest, max_end-start+1))
                    for score, end in zip(topk[0] + start_score, topk[1] + start):
                        score = score.item()

                        if score > current_best_scores[-1]:
                            answer = context.text[context.word2start_char[start]:context.word2end_char[end]]
                            delpos = current_best_answers.index(answer) if answer in current_best_answers else -1
                        
                            if score > current_best_scores[delpos]:
                                current_best_scores.pop(delpos)
                                current_best_answers.pop(delpos)

                                inspos = 0
                                while inspos < len(current_best_scores) and current_best_scores[inspos] > score:
                                    inspos += 1

                                current_best_scores.insert(inspos, score)
                                current_best_answers.insert(inspos, answer)


            best_scores.append(current_best_scores)
            best_answers.append(current_best_answers)
    
        best_scores = torch.tensor(best_scores)
        return best_scores, torch.softmax(best_scores, -1), best_answers
