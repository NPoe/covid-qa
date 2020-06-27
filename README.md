# Covid-QA with inexpensive domain adaptation

This repository contains code for the Covid-19 QA experiment from the following paper:

[Inexpensive Domain Adaptation for Pretrained Language Models: Case Studies on Biomedical NER and Covid-19 QA](https://arxiv.org/abs/2004.03354)

## Step-by-step

Start by setting up the environment:
```console
conda env create -f environment.yaml
conda activate covid-qa
```

We provide a script that downloads all necessary files:
```console
chmod +x prepare.sh
./prepare.sh
```

The downloaded files include:
* the aligned CORD-19 Word2Vec vectors: `data/cord-19-vectors.pt`
* the associated Word2Vec vocabulary: `data/cord-19-vocab.txt`
* the SQuAD scorer: `evaluate-v2.0.py`
* the Covid-QA dataset (from 2020-04-23): `data/dataset.json`

The original dataset uses integer question IDs, which breaks the SQuAD scorer.
Therefore, the script has created a version with string question IDs in `data/dataset_fixed_ids.json`.

Now you can run the GreenCovidSQuADBERT model:
```bash
python3 main.py --verbose --infile data/dataset_fixed_ids.json --outprefix data/GreenCovidSQuADBERT --embeddingprefix data/cord-19 --batch_size 128
```

The script saves its predictions in the `data` directory.
Evaluate the predictions with the SQuAD scorer:
```bash
python3 evaluate-v2.0.py data/dataset_fixed_ids.json data/GreenCovidSQuADBERT.predictions.json
```

The output should look like this:
```javascript
{
  "exact": 34.62109955423477,
  "f1": 60.094416813966,
  "total": 2019,
  "HasAns_exact": 34.62109955423477,
  "HasAns_f1": 60.094416813966,
  "HasAns_total": 2019
}
```

Now run and evaluate the baseline SQuADBERT model, without aligned Word2Vec vectors:
```bash
python3 main.py --verbose --infile data/dataset_fixed_ids.json --outprefix data/SQuADBERT --batch_size 128
python3 evaluate-v2.0.py data/dataset_fixed_ids.json data/SQuADBERT.predictions.json
```

The output should look like this:
```javascript
{
  "exact": 33.03615651312531,
  "f1": 58.236683765572,
  "total": 2019,
  "HasAns_exact": 33.03615651312531,
  "HasAns_f1": 58.236683765572,
  "HasAns_total": 2019
}
```
