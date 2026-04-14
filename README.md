# Laboratório 07 — Fine-Tuning com LoRA e QLoRA

**Disciplina:** Inteligência Artificial Aplicada  
**Instituição:** Instituto iCEV  
**Aluno:** Andreas Carvalho  
**Professor:** Prof. Dimmy  
**Entrega:** versão `v1.0`

> *"Partes geradas/complementadas com IA, revisadas por Andreas Carvalho"*  
> O Claude (Anthropic) foi utilizado como apoio na geração de templates de código e documentação. Todo o conteúdo foi revisado e validado criticamente antes da submissão.

---

## Sobre o Projeto

Pipeline de fine-tuning supervisionado do modelo `TinyLlama/TinyLlama-1.1B-Chat-v1.0` especializado no domínio de **Suporte Técnico de TI**, implementado com as técnicas **LoRA** e **QLoRA** para viabilizar o treinamento em hardware com memória limitada.

O domínio escolhido cobre tópicos avançados de infraestrutura:

- Active Directory e autenticação Windows
- Monitoramento com Zabbix e Grafana
- VPN site-to-site com IPSec
- Hardening de servidores Linux (CIS Benchmark)
- Containerização com Docker e Kubernetes
- Análise de logs com ELK Stack
- Recuperação de desastres e continuidade de negócios
- Automação com Ansible e Terraform

---

## Estrutura do Repositório

```
lab07-lora-qlora/
│
├── dataset/
│   ├── genarator.py        # Gera pares instrução/resposta via API OpenAI
│   ├── treino.jsonl        # 72 amostras (90% do total)
│   └── teste.jsonl         #  8 amostras (10% do total)
│
├── finetuning.py           # Passos 2, 3 e 4 — treinamento com QLoRA
├── inference.py            # Teste do modelo fine-tunado
├── jsonl.py                # Utilitário de correção de escape nos .jsonl
├── requirements.txt        # Dependências
└── README.md
```

---

## Pré-requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (mínimo 8GB VRAM) **ou** Google Colab com GPU T4

Instale as dependências:

```bash
pip install transformers datasets accelerate peft trl bitsandbytes openai python-dotenv
```

---

## Como Executar

### 1. Gerar o dataset

Crie um arquivo `.env` na raiz com sua chave:

```
OPENAI_API_KEY=sk-sua_chave_aqui
```

Execute:

```bash
python dataset/genarator.py
```

Serão gerados 80 pares instrução/resposta distribuídos entre os 8 tópicos, salvos em `treino.jsonl` (90%) e `teste.jsonl` (10%).

### 2. Corrigir escapes nos JSONL (opcional)

Caso o dataset apresente erros de parsing JSON por barras invertidas inválidas:

```bash
python jsonl.py
```

### 3. Executar o fine-tuning

```bash
python finetuning.py
```

O adaptador LoRA treinado é salvo em `./adaptador_lora/`.

### 4. Testar o modelo

```bash
python inference.py
```

---

## Decisões Técnicas

### Passo 2 — Quantização 4-bit (QLoRA)

O modelo é carregado em 4 bits usando `BitsAndBytesConfig` com quantização `nf4` (NormalFloat 4-bit) e `compute_dtype=float16`. Isso reduz o consumo de VRAM de ~24GB (full precision) para ~4GB, viabilizando o treinamento em GPUs de consumo.

### Passo 3 — LoRA

O LoRA congela todos os pesos originais e injeta matrizes de decomposição de baixo rank nas camadas de atenção e FFN. Apenas essas matrizes são atualizadas durante o treino.

Hiperparâmetros utilizados (conforme enunciado):

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `r` | 64 | Rank das matrizes de decomposição |
| `lora_alpha` | 16 | Fator de escala dos novos pesos |
| `lora_dropout` | 0.1 | Regularização para evitar overfitting |

### Passo 4 — Otimizador e Scheduler

| Configuração | Valor | Motivo |
|-------------|-------|--------|
| `optim` | `paged_adamw_32bit` | Pagina estados do otimizador para a RAM, evitando OOM |
| `lr_scheduler_type` | `cosine` | Decaimento suave da taxa de aprendizado |
| `warmup_ratio` | `0.03` | Aquece a LR nos primeiros 3% dos steps |

---

## Referências

- Hu et al. (2021) — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023) — [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Documentação PEFT — Hugging Face](https://huggingface.co/docs/peft)
- [Documentação TRL — SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [TinyLlama no Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)