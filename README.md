# Transformer

## Dataset

**IWSLT2017** [IWSLT/iwslt2017 · Datasets at Hugging Face](https://huggingface.co/datasets/IWSLT/iwslt2017)

**Description**: 

The IWSLT 2017 Multilingual Task addresses text translation, including zero-shot translation, with a single MT system across all directions including English, German, Dutch, Italian and Romanian. As unofficial task, conventional bilingual text translation is offered between English and Arabic, French, Japanese, Chinese, German and Korean.

## Environment Setup

```
conda create xx
conda activate xx
pip install -r requirements.txt
```

## Project Structure

- **`README.md`**: Project documentation and overview.
- **src/:** model.py, data.py, train.py, utils.py
- **scripts/:** run.sh
- **results/:** model.pt, train_curves_seq2seq.png
- **requirements.txt**

## Training Step

```
python train.py
```

or

```
bash run.sh
```

