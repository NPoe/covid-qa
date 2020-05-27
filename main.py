import json
import tqdm
import argparse
import torch

from src.data import QADataset
from src.modeling import QAModel

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--infile", type = str, required = True,
            help = "Input json file (SQuAD-style)")
    argparser.add_argument("--outprefix", type = str, required = True, 
            help = "File prefix of output files (<outprefix>.predictions.json and <outprefix>.nbest_predictions.json)")
    argparser.add_argument("--pretrained_model_name_or_path", type = str, 
            default = "bert-large-uncased-whole-word-masking-finetuned-squad", help = "Pretrained BERT QA model")
    argparser.add_argument("--embeddingprefix", type = str, default = None,
            help = "File prefix of additional embeddings (<embeddingprefix>-vocab.txt and <embeddingprefix>-vectors.pt)")
    argparser.add_argument("--model_device", default = None,
            help = "Model device. If None, we infer if cuda is available.")
    argparser.add_argument("--embedding_device", default = None,
            help = "Embedding device. If None, we infer if cuda is available. If None, and an embeddingprefix is provided, cpu.")
    argparser.add_argument("--batch_size", type = int, default = None,
            help = "Inference batch size. If None, we infer it from GPU size.")
    argparser.add_argument("--verbose", action = "store_true",
            help = "Whether to show progress bar")
    argparser.add_argument("--max_answer_chars", type = int, default = 500,
            help = "Maximum length of answer span in characters")
    argparser.add_argument("--num_nbest", type = int, default = 20,
            help = "Size of nbest list to be returned per question")
    
    return argparser.parse_args()


def main(args):
    with torch.no_grad():
        
        dataset = QADataset.from_file(args.infile)
        model = QAModel(args)
        nbest_predictions = {}

        contexts = tqdm.tqdm(dataset.contexts, desc = "Predicting") if args.verbose else dataset.contexts     
        for context_i, context in enumerate(contexts):
            scores, probs, answers = model.get_best_scores_and_answers(context)

            for question_i, question in enumerate(context.questions):
                nbest_predictions[question.uid] = []
                
                for nbest_i in range(args.num_nbest):
                    if probs[question_i, nbest_i] > 0:
                        prediction = {\
                                "text": answers[question_i][nbest_i],
                                "score": scores[question_i, nbest_i].item(),
                                "probability": probs[question_i, nbest_i].item()}
                        nbest_predictions[question.uid].append(prediction)

        with open(args.outprefix + ".nbest_predictions.json", "w") as whandle:
            json.dump(nbest_predictions, whandle, indent = 2)
    
        with open(args.outprefix + ".predictions.json", "w") as whandle:
            onebest = {uid: predictions[0]["text"] for uid, predictions in nbest_predictions.items()}
            json.dump(onebest, whandle, indent = 2)



if __name__ == "__main__":
    main(parse_args())

    
