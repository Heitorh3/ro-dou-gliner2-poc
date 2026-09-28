from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")

texto_teste = """
Pedro decidiu jogar a toalha depois de
tantas tentativas fracassadas.

Ana precisou quebrar o gelo antes de
iniciar a reunião.

O diretor resolveu dar o braço a torcer
e aceitar a proposta.

A equipe conseguiu virar o jogo durante
a negociação.

O cozinheiro colocou a mão na massa
para preparar o pão.
"""

schema = model.create_schema().entities(
    {
        "expressao_idiomatica": (
            "Expressão idiomática da língua portuguesa "
            "formada por várias palavras e utilizada "
            "com sentido figurado. "
            "Exemplos: sair do papel, ficar em cima "
            "do muro, bater as botas. "
            "Não extrair nomes próprios ou expressões "
            "utilizadas em sentido literal."
        )
    }
)

resultado = model.extract(
    texto_teste, schema, threshold=0.01, include_confidence=True, include_spans=True
)

print(resultado)

# resultado = model.extract(texto, schema, include_confidence=True, include_spans=True)

# print(resultado)

# import inspect

# print(inspect.signature(model.extract))
