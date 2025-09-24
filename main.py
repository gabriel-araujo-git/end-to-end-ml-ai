import fitz  # PyMuPDF
import cohere
import numpy as np
from numpy.linalg import norm

# Configure sua API key do Cohere
co = cohere.Client("JxLEMwMs74BaSN8k6l6c0SGpJyZPjDHPS0UnOZj2")

# Função para extrair texto do PDF
def extrair_texto_pdf(caminho_pdf):
    doc = fitz.open(caminho_pdf)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

# Função para dividir em chunks
def dividir_em_chunks(texto, tamanho=500):
    palavras = texto.split()
    return [" ".join(palavras[i:i + tamanho]) for i in range(0, len(palavras), tamanho)]

# Gera embedding com Cohere
def gerar_embedding(texto):
    response = co.embed(texts=[texto], model="embed-english-v3.0")
    return np.array(response.embeddings[0])

# Busca os chunks mais parecidos com a pergunta
def buscar_chunks(pergunta, chunks, embeddings, top_k=3):
    emb_pergunta = gerar_embedding(pergunta)
    distancias = [np.dot(emb, emb_pergunta) / (norm(emb) * norm(emb_pergunta)) for emb in embeddings]
    top_indices = np.argsort(distancias)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Usa modelo de linguagem para gerar resposta
def responder(pergunta, contexto):
    prompt = f"Baseado nos textos a seguir, responda a pergunta jurídica:\n\n{contexto}\n\nPergunta: {pergunta}"
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300,
        temperature=0.5
    )
    return response.generations[0].text.strip()

# Executar tudo
if __name__ == "__main__":
    texto = extrair_texto_pdf("docs/constituicao.pdf")
    chunks = dividir_em_chunks(texto)
    print(f"{len(chunks)} chunks gerados.")

    embeddings = [gerar_embedding(chunk) for chunk in chunks]
    print("Embeddings gerados com sucesso.")

    while True:
        pergunta = input("\nDigite sua pergunta jurídica (ou 'sair'): ")
        if pergunta.lower() == "sair":
            break
        contexto = "\n\n".join(buscar_chunks(pergunta, chunks, embeddings))
        resposta = responder(pergunta, contexto)
        print(f"\nResposta:\n{resposta}")
