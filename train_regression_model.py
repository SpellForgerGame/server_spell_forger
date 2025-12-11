


from model.matrix_regression_model import MatrixRegressionModel
from transformers import AutoTokenizer
#from train import train_model


from data.data_definitions import SPELL_FIELDS, get_spell_field_max




def normalize_spell_features(features):
    normalized = []
    for i, value in enumerate(features):
        field_name = SPELL_FIELDS[i]
        max_val = get_spell_field_max(field_name)  # max index (e.g. 255 for colors)
        normalized_value = value / max_val if max_val > 0 else 0
        normalized.append(normalized_value)
    return normalized

def denormalize_spell_features(normalized_features):

  features = []

  for i, normalized_value in enumerate(normalized_features):
    field_name = SPELL_FIELDS[i]
    max_val = get_spell_field_max(field_name) # max index (e.g. 255 for colors)
    value = normalized_value * max_val if max_val > 0 else 0
    features.append(value)

  return features

def load_data_from_json(path):
    """
    Load training examples from a JSONL file with fields 'description' and 'labels'.
    Returns texts (list of str) and matrices (list of list of floats).
    """
    import json
    texts, matrices = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {lineno} due to JSON decode error: {e}")
                continue
            if 'description' not in obj or 'labels' not in obj:
                print(f"Warning: skipping line {lineno}: missing 'description' or 'labels'")
                continue
            texts.append(obj['description'])
            matrices.append([obj['labels']])

            if 'keywords' in obj:
                for keyword in obj['keywords']:
                    texts.append(keyword)
                    matrices.append([obj['labels']])

    return texts, matrices

def combine_data(data, other_data):
  data[0].extend(other_data[0])
  data[1].extend(other_data[1])
  return data





def train_regression_model():
    model = MatrixRegressionModel()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')



    texts, matrices = load_data_from_json('train.jsonl')
    texts, matrices = combine_data((texts, matrices), load_data_from_json('train_unverified.jsonl'))

    matrices = [normalize_spell_features(matrix[0]) for matrix in matrices]

    epochs = 9 # 19/05/25 -> seems to not learn further after this many
    model = train_model(texts, matrices, model, tokenizer, epochs=epochs)

    save_model(model, tokenizer, './bert_matrix_model')