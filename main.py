#Inicia o servidor e lida com as rotas
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from run_regression_model import run_model, predict
from train_regression_model import denormalize_spell_features 
from status_matrix_bert_matrix_model.status_model import load_model, predict_effects
import os
app = FastAPI()

class SpellRequest(BaseModel):
    description: str

model, tokenizer = (None, None)
status_model, status_tokenizer = (None, None)

@app.post("/generate-spell/")
def generate_spell(request: SpellRequest):
 
    #return {"result": f"Spell generated from: {request.description}"}
    #model, tokenizer = run_model()
    global model, tokenizer, status_model, status_tokenizer
    if model == None:
        model, tokenizer = run_model()
    if status_model == None:
        status_model, status_tokenizer = load_model('status_matrix_model')
    results = predict(request.description, model, tokenizer) 
    #results = results[0]
    results = denormalize_spell_features(results)
   
    #print(results)

    results_effects = predict_effects(request.description, status_model, status_tokenizer)
    results_effects = results_effects.tolist() 
    #results_effects = [v for v in results_effects]
    #print('results_effects')
    #print(results_effects) 
    #print(type(results_effects))
    #json_serializable_results = results.tolist()
    return {"result": results, 'effects': results_effects} 



if (__name__ == "__main__" ) :
    
    #uvicorn.run("main:app", host="127.0.0.1", port=8000)
    
    port = int(os.environ.get("PORT", 8000))
    # MUST listen on 0.0.0.0 to be accessible outside the container's localhost
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    model, tokenizer = run_model()
