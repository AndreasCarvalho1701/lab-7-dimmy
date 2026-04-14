"""
Inferência com o adaptador LoRA treinado no Laboratório 07.

Carrega o modelo base quantizado em 4 bits, aplica o adaptador
salvo em ./adaptador_lora e gera respostas para perguntas de
Suporte Técnico de TI no mesmo formato usado no fine-tuning.

Execução:
    python inference.py


"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./adaptador_lora"   # gerado pelo finetuning.py

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.7
TOP_P          = 0.9

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


perguntas_teste = [
    "Como adicionar um usuário ao grupo de administradores no Active Directory via PowerShell?",
    "O Zabbix parou de receber métricas de um host. Quais são as causas mais comuns e como diagnosticar?",
    "Como configurar um túnel VPN IPSec site-to-site entre dois roteadores Linux com StrongSwan?",
    "Quais comandos do CIS Benchmark devo aplicar primeiro para fazer o hardening de um Ubuntu Server?",
    "Um container Docker está reiniciando em loop. Como identificar a causa pelo log?",
]




def carregar_modelo():
    
    print("Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("Carregando modelo base com quantização 4-bit...")
    modelo_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Aplicando adaptador LoRA de '{ADAPTER_PATH}'...")
    modelo = PeftModel.from_pretrained(modelo_base, ADAPTER_PATH)
    modelo.eval()

    print("Modelo pronto!\n")
    return modelo, tokenizer


def gerar_resposta(modelo, tokenizer, instrucao: str) -> str:
    
    prompt = (
        "### Instrução:\n"
        f"{instrucao.strip()}\n\n"
        "### Resposta:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)

    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokens_novos = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(tokens_novos, skip_special_tokens=True).strip()


def main():
    print("=" * 70)
    print("  Testando o modelo fine-tunado — Suporte Técnico de TI")
    print("  Laboratório 07 — iCEV")
    print("=" * 70)

    modelo, tokenizer = carregar_modelo()

    for i, pergunta in enumerate(perguntas_teste, 1):
        print(f"\n[Pergunta {i}/{len(perguntas_teste)}]: {pergunta}")
        print("-" * 50)
        resposta = gerar_resposta(modelo, tokenizer, pergunta)
        print(f"[Resposta]: {resposta}\n")

    print("=" * 70)
    print("  Inferência concluída.")
    print("=" * 70)


if __name__ == "__main__":
    main()