"""
Utilitário de sanitização de arquivos JSONL.

Varre cada linha dos arquivos gerados por gerar_dataset.py e tenta
reparar sequências de escape inválidas antes do fine-tuning.

Execução:
    python jsonl.py

Arquivos processados:
    dataset/treino.jsonl
    dataset/teste.jsonl

"""

import json
from dataclasses import dataclass, field
from pathlib import Path


ESCAPES_JSON = frozenset('"\\\/bfnrtu')

ARQUIVOS = [
    Path("dataset/treino.jsonl"),
    Path("dataset/teste.jsonl"),
]




@dataclass
class Relatorio:
    caminho: Path
    total: int = 0
    corrigidas: int = 0
    falhas: int = 0
    ignoradas: int = 0
    erros: list[str] = field(default_factory=list)




def escapar_barras_invalidas(texto: str) -> str:
    
    saida: list[str] = []
    pos = 0

    while pos < len(texto):
        c = texto[pos]
        if c != "\\":
            saida.append(c)
            pos += 1
            continue

        proximo = texto[pos + 1] if pos + 1 < len(texto) else ""
        if proximo in ESCAPES_JSON:
            saida.append(c)  
        else:
            saida.append("\\\\")   

        pos += 1

    return "".join(saida)


def tentar_reparar(linha: str) -> tuple[str, bool]:
    
    candidata = escapar_barras_invalidas(linha)
    try:
        json.loads(candidata)
        return candidata, True
    except json.JSONDecodeError:
        return linha, False




def processar(caminho: Path) -> Relatorio:

    rel = Relatorio(caminho=caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não localizado: {caminho}\n"
            "Certifique-se de ter executado gerar_dataset.py antes."
        )

    linhas_brutas = caminho.read_text(encoding="utf-8").splitlines()
    linhas_finais: list[str] = []

    for num, bruta in enumerate(linhas_brutas, start=1):
        linha = bruta.strip()

        if not linha:
            rel.ignoradas += 1
            continue

        rel.total += 1

        
        try:
            json.loads(linha)
            linhas_finais.append(linha)
            continue
        except json.JSONDecodeError:
            pass

        
        reparada, sucesso = tentar_reparar(linha)
        linhas_finais.append(reparada)

        if sucesso:
            rel.corrigidas += 1
            print(f"    [reparada] linha {num}")
        else:
            rel.falhas += 1
            msg = f"linha {num}: reparo não foi suficiente"
            rel.erros.append(msg)
            print(f"    [falha]    {msg}")

    caminho.write_text(
        "\n".join(linhas_finais) + "\n",
        encoding="utf-8",
    )

    return rel




def exibir_relatorio(rel: Relatorio) -> None:
    print(f"\n  ▸ {rel.caminho}")
    print(f"    linhas processadas : {rel.total}")
    print(f"    reparadas          : {rel.corrigidas}")
    print(f"    falhas             : {rel.falhas}")
    print(f"    vazias ignoradas   : {rel.ignoradas}")
    if rel.erros:
        print("    detalhes das falhas:")
        for e in rel.erros:
            print(f"      • {e}")




def main() -> None:
    print("=" * 55)
    print("  Sanitização de arquivos JSONL")
    print("=" * 55)

    relatorios: list[Relatorio] = []

    for arq in ARQUIVOS:
        print(f"\nProcessando {arq} ...")
        try:
            rel = processar(arq)
            relatorios.append(rel)
        except FileNotFoundError as erro:
            print(f"  [AVISO] {erro}")

    print("\n" + "=" * 55)
    print("  Relatório final")
    print("=" * 55)
    for rel in relatorios:
        exibir_relatorio(rel)

    total_falhas = sum(r.falhas for r in relatorios)
    print("\n" + ("✓ Nenhuma falha irreparável encontrada."
                  if total_falhas == 0
                  else f"⚠ {total_falhas} linha(s) não puderam ser corrigidas — revise manualmente."))


if __name__ == "__main__":
    main()