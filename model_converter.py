import torch
# Supondo que suas classes de modelo (MultiHeadModel, SpellMatrixClassifier) e
# o tokenizer já estejam definidos como no seu script.

# --- Exportando o MultiHeadModel ---
print("Exportando MultiHeadModel...")
model_simple = MultiHeadModel() # Ou carregue o modelo salvo
# Carregue os pesos treinados se necessário:
# model_simple.load_state_dict(torch.load("caminho/para/os/pesos/multiheadmodel.pth")) # Se você salvou apenas o state_dict
# Ou se você usou trainer.save_model():
# model_simple = MultiHeadModel.from_pretrained("./saved_models/spell_simple_features_model") # Ajuste conforme necessário

model_simple.eval() # Colocar o modelo em modo de avaliação

# Exemplo de entrada para o modelo (necessário para o exportador ONNX traçar o modelo)
tokenizer_name = "distilbert-base-uncased"
tokenizer_onnx = AutoTokenizer.from_pretrained(tokenizer_name)
dummy_text = "a fiery spell"
inputs = tokenizer_onnx(dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
dummy_input_ids = inputs['input_ids']
dummy_attention_mask = inputs['attention_mask']

# Definir nomes de entrada e saída (importante para o Sentis)
input_names_simple = ["input_ids", "attention_mask"]
# Os nomes das saídas devem corresponder ao que você espera no Unity
# O MultiHeadModel retorna um dicionário com "logits", que é um stack de tensores.
# Para Sentis, é melhor se o modelo ONNX retornar diretamente os tensores.
# Você pode precisar ajustar a função forward do seu MultiHeadModel para retornar
# uma tupla de tensores se torch.onnx.export tiver problemas com o tensor empilhado diretamente.
# Ou, se o "logits" empilhado for um único tensor, nomeie-o.
# Vamos supor que a saída "logits" seja um único tensor [batch_size, num_fields]
output_names_simple = ["output_logits_simple"]


# Ajuste no forward do MultiHeadModel para exportação ONNX (opcional, mas pode ajudar)
# Se o forward original for: return {"logits": torch.stack([l.argmax(dim=-1) for l in logits], dim=1)}
# Para exportação, você pode querer os logits brutos antes do argmax, ou o argmax se preferir.
# Se você quiser os índices diretamente (após argmax):
# (no forward): return torch.stack([l.argmax(dim=-1) for l in logits], dim=1)
# Se você quiser os logits brutos de cada cabeça (mais flexível):
# (no forward): return tuple(logits) # Isso daria múltiplas saídas
# Vamos assumir que seu forward atual retorna um tensor único para "logits" após o stack e argmax.

torch.onnx.export(
    model_simple,
    (dummy_input_ids, dummy_attention_mask),
    "spell_simple_features.onnx",
    input_names=input_names_simple,
    output_names=output_names_simple,
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "output_logits_simple": {0: "batch_size"}
    },
    opset_version=11 # Ou uma versão mais recente suportada pelo Sentis
)
print("MultiHeadModel exportado para spell_simple_features.onnx")

# --- Exportando o SpellMatrixClassifier ---
print("\nExportando SpellMatrixClassifier...")
model_matrix = SpellMatrixClassifier() # Ou carregue o modelo salvo
# model_matrix.load_state_dict(torch.load("caminho/para/os/pesos/spellmatrix.pth"))
# model_matrix = SpellMatrixClassifier.from_pretrained("./saved_models/spell_status_effect_model")

model_matrix.eval()

# A saída "logits" aqui é um tensor de shape (batch, NUM_OUTPUTS, NUM_CLASSES)
# ou (batch, 20, 3) no seu caso. Se o forward faz argmax, ajuste.
# Vamos assumir que o forward retorna os logits brutos antes do argmax:
# (no forward): return torch.stack([head(cls_output) for head in self.classifiers], dim=1)
output_names_matrix = ["output_logits_matrix"]

torch.onnx.export(
    model_matrix,
    (dummy_input_ids, dummy_attention_mask), # Mesma entrada dummy
    "spell_matrix_classifier.onnx",
    input_names=input_names_simple, # Mesmos nomes de entrada
    output_names=output_names_matrix,
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "output_logits_matrix": {0: "batch_size"}
    },
    opset_version=11
)
print("SpellMatrixClassifier exportado para spell_matrix_classifier.onnx")