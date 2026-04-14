"""
Pipeline de fine-tuning supervisionado no domínio de Suporte Técnico de TI.

Passos cobertos:
    Passo 2 — Quantização 4-bit via BitsAndBytes (nf4, compute_dtype float16)
    Passo 3 — LoRA com r=64, alpha=16, dropout=0.1  (tarefa CAUSAL_LM)
    Passo 4 — SFTTrainer com paged_adamw_32bit, cosine scheduler, warmup 3%

Compatível com os arquivos gerados por gerar_dataset.py:
    dataset/treino.jsonl
    dataset/teste.jsonl

Execução:
    python finetuning.py


"""

# ---------------------------------------------------------------------------
# Dependências
# ---------------------------------------------------------------------------
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer



MODELO_BASE   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


ARQ_TREINO    = Path("dataset/treino.jsonl")
ARQ_TESTE     = Path("dataset/teste.jsonl")

DIR_RESULTADOS = "./resultados"
DIR_ADAPTADOR  = "./adaptador_lora"

COMP_MAX_SEQUENCIA = 512
EPOCAS             = 3




cfg_quantizacao = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)




cfg_lora = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  
        "gate_proj", "up_proj", "down_proj", 
    ],
)




cfg_treino = SFTConfig(
    output_dir=DIR_RESULTADOS,

    
    num_train_epochs=EPOCAS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,

    
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    
    fp16=True,
    bf16=False,
    group_by_length=True,
    gradient_checkpointing=True,

    
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,

    
    logging_steps=10,
    report_to="none",
)




def verificar_arquivos() -> None:
    
    for arq in (ARQ_TREINO, ARQ_TESTE):
        if not arq.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {arq}\n"
                "Execute gerar_dataset.py antes de rodar este script."
            )


def montar_prompt(exemplo: dict) -> str:
    
    instrucao = exemplo.get("instruction", "").strip()
    resposta  = exemplo.get("response", "").strip()

    if not instrucao or not resposta:
        raise ValueError(f"Exemplo com campos vazios: {exemplo}")

    return (
        f"### Instrução:\n{instrucao}\n\n"
        f"### Resposta:\n{resposta}"
    )


def carregar_tokenizer(nome_modelo: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(nome_modelo, trust_remote_code=True)
    tok.pad_token    = tok.eos_token  
    tok.padding_side = "right" 
    return tok


def carregar_modelo(nome_modelo: str) -> AutoModelForCausalLM:
    modelo = AutoModelForCausalLM.from_pretrained(
        nome_modelo,
        quantization_config=cfg_quantizacao,
        device_map="auto",
        trust_remote_code=True,
    )
    modelo.config.use_cache       = False 
    modelo.config.pretraining_tp  = 1 
    return prepare_model_for_kbit_training(modelo)




def main() -> None:
    separador = "=" * 70

    print(separador)
    print("  Fine-Tuning QLoRA — Suporte Técnico de TI")
    print("  Laboratório 07 — iCEV")
    print(separador)


    print("\n[1/6] Verificando arquivos de dataset...")
    verificar_arquivos()
    print("      OK — treino.jsonl e teste.jsonl encontrados.")


    print("[2/6] Carregando tokenizer...")
    tokenizer = carregar_tokenizer(MODELO_BASE)

    print("[3/6] Carregando modelo base com quantização 4-bit (QLoRA)...")
    modelo = carregar_modelo(MODELO_BASE)


    print("[4/6] Lendo datasets JSONL...")
    ds_treino = load_dataset("json", data_files=str(ARQ_TREINO), split="train")
    ds_teste  = load_dataset("json", data_files=str(ARQ_TESTE),  split="train")
    print(f"      Treino : {len(ds_treino)} amostras")
    print(f"      Teste  : {len(ds_teste)} amostras")


    print("[5/6] Configurando SFTTrainer e iniciando treinamento...")
    treinador = SFTTrainer(
        model=modelo,
        train_dataset=ds_treino,
        eval_dataset=ds_teste,
        peft_config=cfg_lora,
        formatting_func=montar_prompt,
        processing_class=tokenizer,
        args=cfg_treino,
        max_seq_length=COMP_MAX_SEQUENCIA,
    )
    treinador.train()


    print(f"\n[6/6] Salvando adaptador LoRA em '{DIR_ADAPTADOR}'...")
    treinador.model.save_pretrained(DIR_ADAPTADOR)
    tokenizer.save_pretrained(DIR_ADAPTADOR)

    print(f"\n{separador}")
    print("  Treinamento concluído com sucesso!")
    print(f"  Adaptador salvo em: {DIR_ADAPTADOR}/")
    print(separador)


if __name__ == "__main__":
    main()