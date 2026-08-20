# Grader RAG estruturado

A Etapa 5 impede que qualquer conjunto nao vazio de chunks seja enviado
automaticamente ao gerador. Depois de `retrieve_rag`, o node `grade_rag` decide
se o contexto sustenta uma resposta para a pergunta exata.

## Contrato

O adapter do grader usa structured output do modelo e valida novamente o
resultado com Pydantic:

```python
class RagGrade(BaseModel):
    relevant: bool
    confidence: float  # 0.0 <= confidence <= 1.0
    reason: str        # nao vazio
```

Campos extras, tipos incorretos, confidence fora do intervalo e reason vazio
sao rejeitados. Uma falha de validacao interrompe o grafo; ela nao e convertida
silenciosamente em decisao positiva ou negativa.

O adapter de chat escolhido deve implementar `with_structured_output`. Falta de
suporte do provider e tratada como erro de configuracao, sem liberar o ramo de
resposta.

## Criterio

`relevant=true` significa que os chunks contem evidencia suficiente para uma
resposta direta e totalmente apoiada. Proximidade apenas tematica nao basta. A
instrucao tambem determina que o conteudo recuperado seja tratado como dado, e
nao como instrucao para o grader.

`confidence` registra a confianca do modelo na propria decisao. Ela ainda nao
controla a rota e nao possui threshold nesta etapa. `reason` fica no estado
interno para diagnostico; a saida publica permanece `answer` e `source`.

## Rotas

```text
grade.relevant = true                         -> generate_answer
grade.relevant = false e ainda ha tentativas -> rewrite_query
grade.relevant = false no limite             -> no_answer
```

Sem documentos, o adapter produz uma decisao negativa deterministica com
confidence `1.0`, sem invocar o modelo do grader. O roteamento da Etapa 6 ainda
pode tentar uma consulta reescrita antes de chegar a `no_answer`.

## Limites

O grader apenas decide e nao altera queries. Rewrite, contador e limite ficam
nos nodes do grafo e estao descritos em [`query-rewrite.md`](query-rewrite.md).
