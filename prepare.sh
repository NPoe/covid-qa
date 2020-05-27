squadeval_url="https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/"
data_url="https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/data/question-answering/200423_covidQA.json"
cord19_vocab_url="https://www.cis.uni-muenchen.de/~poerner/blobs/covid-qa/cord-19-vocab.txt"
cord19_vectors_url="https://www.cis.uni-muenchen.de/~poerner/blobs/covid-qa/cord-19-vectors.pt"

if [[ -f data ]]; then
    rm -r data
fi

mkdir data


wget $squadeval_url -O "evaluate-v2.0.py"
wget $data_url -O "data/dataset.json"

# the Covid-QA question IDs are integers, which breaks the SQuAD evaluation script
# therefore, we convert them to string IDs in dataset_fixed_ids.json

python3 -c "
import json
data = json.load(open('data/dataset.json'))
for datapoint in data['data']:
    for paragraph in datapoint['paragraphs']:
        for qa in paragraph['qas']:
            qa['id'] = str(qa['id'])
json.dump(data, open('data/dataset_fixed_ids.json', 'w'), indent=2, ensure_ascii=False)"

wget $cord19_vocab_url -O "data/cord-19-vocab.txt"
wget $cord19_vectors_url -O "data/cord-19-vectors.pt"
