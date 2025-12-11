import site
site.addsitedir(r'C:\Users\matts\AppData\Roaming\Python\Python313\site-packages')

from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import sys

def build_prompt():
    return f"""
    Você é um especialista em design de sistemas de RPG. Com base no feitiço abaixo, gere uma matriz binária onde:

    - Cada LINHA é um TRIGGER (ação + condição, como "acertar inimigo com armadura")
    - Cada COLUNA é um STATUS EFFECT (efeito como "queimado", "atordoado", etc)
    - Os valores são 1 se o trigger causa o efeito, e 0 caso contrário.

    Responda apenas com:
    Triggers: [lista]
    Status Effects: [lista]
    Matriz: [matriz binária]
    """


if __name__ == "__main__":
    # print("Script Python foi chamado")
    # print("Input recebido da Unity:", sys.argv[1])

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    user_input = "Pegar o Feitiço da interface de usuário"

    messages = [
        SystemMessage(build_prompt()),
        HumanMessage(f"Feitiço: Bola de fogo que explode")
    ]

    modelo = ChatOpenAI(model="gpt-3.5-turbo")

    answer = modelo.invoke(messages)

    print(answer)