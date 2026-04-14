"""
===========================================================
Passo 1: Geração de Dataset Sintético
===========================================================

Domínio escolhido: Suporte Técnico de TI

Pré-requisitos:
    1. Crie um arquivo .env com:
           OPENAI_API_KEY=sk-sua_chave_aqui
    2. Instale as dependências:
           pip install openai python-dotenv
    3. Execute:
           python gerar_dataset.py

Saída:
    - treino.jsonl   (90% dos dados)
    - teste.jsonl    (10% dos dados)
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


CHAVE_API      = os.getenv("OPENAI_API_KEY")
MODELO         = "gpt-4o-mini"
TOTAL          = 80      
PROPORCAO_TREINO = 0.9      
SEMENTE        = 7        
TENTATIVAS_MAX = 3  
PAUSA_SEGUNDOS = 0.5         

PASTA_SAIDA  = Path(__file__).parent
ARQ_TREINO   = PASTA_SAIDA / "treino.jsonl"
ARQ_TESTE    = PASTA_SAIDA / "teste.jsonl"


TOPICOS = [
    "Troubleshooting de Active Directory e autenticação Windows",
    "Monitoramento de infraestrutura com Zabbix e Grafana",
    "Configuração de VPN site-to-site com IPSec",
    "Hardening de servidores Linux (CIS Benchmark)",
    "Containerização com Docker e orquestração com Kubernetes",
    "Análise de logs com ELK Stack (Elasticsearch, Logstash, Kibana)",
    "Recuperação de desastres e plano de continuidade de negócios",
    "Automação de infraestrutura com Ansible e Terraform",
]


INSTRUCAO_SISTEMA = (
    "Você atua como engenheiro sênior de suporte técnico de TI.\n"
    "Gere exatamente um par no formato JSON com as chaves:\n"
    '  "instruction": pergunta ou problema real enfrentado por um técnico ou usuário\n'
    '  "response": solução detalhada, prática e tecnicamente precisa\n\n'
    "Regras obrigatórias:\n"
    "- Escreva em português brasileiro\n"
    "- Não inclua texto fora do JSON\n"
    "- Não use markdown, blocos de código ou comentários\n"
    "- A resposta deve ter ao menos 3 frases explicativas\n"
)




def par_valido(dado: dict) -> bool:
    
    for campo in ("instruction", "response"):
        valor = dado.get(campo, "")
        if not isinstance(valor, str) or len(valor.strip()) < 10:
            return False
    return True


def chamar_api(cliente: OpenAI, topico: str) -> dict:
    
    mensagem_usuario = (
        f"Tópico: {topico}\n"
        "Crie uma situação realista de suporte técnico relacionada a esse tópico "
        "e forneça a solução correspondente."
    )

    resposta = cliente.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": INSTRUCAO_SISTEMA},
            {"role": "user",   "content": mensagem_usuario},
        ],
        temperature=0.85,
        max_tokens=700,
        response_format={"type": "json_object"},
    )

    conteudo = resposta.choices[0].message.content
    par = json.loads(conteudo)

    if not par_valido(par):
        raise ValueError(f"Par inválido retornado pela API: {par}")

    return {
        "instruction": par["instruction"].strip(),
        "response":    par["response"].strip(),
    }


def gerar_dataset(cliente: OpenAI, total: int = TOTAL) -> list[dict]:
    
    dataset: list[dict] = []

    base        = total // len(TOPICOS)
    extras      = total % len(TOPICOS)
    quantidades = [base + (1 if i < extras else 0) for i in range(len(TOPICOS))]

    for topico, qtd in zip(TOPICOS, quantidades):
        print(f"\n▸ Tópico: {topico}  ({qtd} amostras)")
        gerados = 0

        while gerados < qtd:
            for tentativa in range(1, TENTATIVAS_MAX + 1):
                try:
                    par = chamar_api(cliente, topico)
                    dataset.append(par)
                    gerados += 1
                    print(f"  [{gerados}/{qtd}] gerado com sucesso")
                    time.sleep(PAUSA_SEGUNDOS)
                    break
                except Exception as erro:
                    print(f"  tentativa {tentativa}/{TENTATIVAS_MAX} falhou: {erro}")
                    time.sleep(PAUSA_SEGUNDOS)
            else:
                print("  ✗ amostra ignorada após todas as tentativas")
                break   

    return dataset


def salvar_jsonl(caminho: Path, registros: list[dict]) -> None:
    
    with caminho.open("w", encoding="utf-8") as arq:
        for registro in registros:
            arq.write(json.dumps(registro, ensure_ascii=False) + "\n")


def dividir_e_salvar(dataset: list[dict]) -> None:
    
    minimo_exigido = 50
    if len(dataset) < minimo_exigido:
        raise RuntimeError(
            f"Dataset insuficiente: {len(dataset)} amostras geradas "
            f"(mínimo exigido: {minimo_exigido})."
        )

    random.seed(SEMENTE)
    random.shuffle(dataset)

    corte        = int(len(dataset) * PROPORCAO_TREINO)
    dados_treino = dataset[:corte]
    dados_teste  = dataset[corte:]

    salvar_jsonl(ARQ_TREINO, dados_treino)
    salvar_jsonl(ARQ_TESTE,  dados_teste)

    print("\n" + "=" * 50)
    print(f"  treino : {len(dados_treino)} amostras  →  {ARQ_TREINO}")
    print(f"  teste  : {len(dados_teste)} amostras   →  {ARQ_TESTE}")
    print(f"  total  : {len(dataset)} amostras")
    print("=" * 50)


def main() -> None:
    print("=" * 50)
    print("  Dataset Sintético — Suporte Técnico de TI")
    print("=" * 50)

    if not CHAVE_API:
        raise EnvironmentError(
            "Variável OPENAI_API_KEY não encontrada.\n"
            "Adicione-a ao arquivo .env antes de executar o script."
        )

    cliente = OpenAI(api_key=CHAVE_API)

    print(f"\nGerando {TOTAL} amostras em {len(TOPICOS)} tópicos...\n")
    dataset = gerar_dataset(cliente, TOTAL)

    print("\nSalvando arquivos...")
    dividir_e_salvar(dataset)

    print("\n✓ Concluído com sucesso!")


if __name__ == "__main__":
    main()