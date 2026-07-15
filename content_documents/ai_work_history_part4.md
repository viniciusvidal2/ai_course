# Parte 4 — Retrieval-Augmented Generation (RAG) — Fundamentos Matemáticos e Arquitetura Completa

---

# Capítulo 11 — Embeddings

## Introdução

Até este ponto vimos que um sistema RAG divide documentos em pequenos trechos (chunks). Entretanto, computadores não compreendem texto da mesma maneira que seres humanos.

Para um computador, as frases:

> "O cachorro está correndo."

e

> "Um cão corre rapidamente."

são completamente diferentes quando comparadas caractere por caractere.

Mesmo possuindo praticamente o mesmo significado, não existe nenhuma similaridade lexical evidente.

Para resolver esse problema surgiram os **embeddings**.

Um embedding transforma texto em um vetor numérico capaz de representar seu significado semântico.

Assim, frases semanticamente semelhantes tornam-se vetores próximos em um espaço vetorial.

---

## Representação Vetorial

Considere a frase:

```text
O cachorro está brincando.
```

Após passar pelo modelo de embeddings, ela pode ser representada por um vetor como:

\[
\mathbf{x}
=
[0.28,\,-0.17,\,...,\,0.91]
\]

Dependendo do modelo, esse vetor pode possuir:

- 384 dimensões
- 512 dimensões
- 768 dimensões
- 1024 dimensões
- 1536 dimensões
- 3072 dimensões

Cada dimensão representa uma característica aprendida automaticamente durante o treinamento do modelo.

Importante destacar que essas dimensões não possuem interpretação humana direta. Não existe uma dimensão "animal" ou "felicidade". O significado está distribuído por todo o vetor.

---

## Espaço Vetorial

Imagine um espaço tridimensional.

Cada documento ocupa um ponto.

```text
                Documento B

                    •

               •

Documento A

                              •

                      Documento C
```

Agora imagine esse mesmo espaço com 1536 dimensões.

Embora impossível de visualizar, a matemática continua exatamente a mesma.

Documentos semelhantes ficam próximos.

Documentos diferentes ficam distantes.

---

## Propriedades dos Embeddings

Modelos modernos aprendem propriedades extremamente interessantes.

Por exemplo:

```text
Brasil

↓

Brasília
```

e

```text
França

↓

Paris
```

produzem relações vetoriais semelhantes.

Em linguagem matemática:

\[
\vec{Brasil}
-
\vec{Brasília}
\approx
\vec{França}
-
\vec{Paris}
\]

Esse comportamento mostra que embeddings representam relações semânticas profundas.

---

## Como os Embeddings são Obtidos

Os modelos de embeddings normalmente são derivados de Transformers.

Durante o treinamento em Self-Supervised Learning, o modelo aprende representações internas extremamente ricas.

Em vez de utilizar a camada responsável por prever o próximo token, utiliza-se uma camada intermediária como representação do texto.

Essa representação passa a ser o embedding.

---

# Capítulo 12 — Tokenização

Antes da vetorização, todo texto passa por um processo chamado **tokenização**.

---

## O que é um Token?

Um token não corresponde necessariamente a uma palavra.

Exemplo.

Texto:

```text
Inteligência Artificial
```

Pode ser dividido em:

```text
Intelig

ência

Artificial
```

Outro exemplo:

```text
ChatGPT
```

Pode tornar-se

```text
Chat

GPT
```

Cada modelo possui seu próprio tokenizador.

---

## Por que Tokenizar?

Redes neurais trabalham com números.

Logo:

Texto

↓

Tokens

↓

IDs numéricos

↓

Embeddings

↓

Rede Neural

---

## Vocabulary

Todo modelo possui um vocabulário.

Exemplo simplificado:

```text
cachorro → 514

gato → 219

carro → 980

cidade → 1834
```

Durante o processamento, cada token é convertido em seu identificador correspondente.

---

# Capítulo 13 — Similaridade Vetorial

Uma vez que documentos tornam-se vetores, surge a pergunta:

Como descobrir quais documentos são semelhantes?

Diversas métricas podem ser utilizadas.

---

## Distância Euclidiana

É a distância tradicional entre dois pontos.

Dados dois vetores

\[
A=(a_1,\ldots,a_n)
\]

e

\[
B=(b_1,\ldots,b_n)
\]

tem-se

\[
d(A,B)
=
\sqrt{
\sum_{i=1}^{n}
(a_i-b_i)^2
}
\]

Quanto menor a distância, mais semelhantes os documentos.

---

## Produto Escalar

Outra medida bastante utilizada.

\[
A\cdot B
=
\sum_i
a_i b_i
\]

Quanto maior o produto interno, maior tende a ser a similaridade.

---

## Similaridade por Cosseno

A métrica mais utilizada em sistemas RAG.

Ela mede o ângulo entre dois vetores.

\[
\cos(\theta)
=
\frac
{A\cdot B}
{||A||\,||B||}
\]

Valores possíveis:

```text
1

↓

Vetores praticamente iguais
```

```text
0

↓

Sem relação
```

```text
-1

↓

Opostos
```

Na prática:

Pergunta:

```text
Como funciona PostgreSQL?
```

Documento:

```text
Introdução ao PostgreSQL
```

Possui alta similaridade.

Já um documento sobre jardinagem produzirá valor muito menor.

---

## Por que usar Cosseno?

A magnitude do vetor deixa de importar.

Importa apenas sua direção.

Isso torna a busca muito mais robusta.

---

# Capítulo 14 — Bancos Vetoriais

Após gerar embeddings para milhares ou milhões de documentos, surge outro problema.

Como pesquisar rapidamente?

Uma busca sequencial teria custo:

\[
O(n)
\]

Para milhões de documentos isso torna-se inviável.

Surgem então os **Vector Databases**.

---

## Objetivo

Armazenar embeddings e realizar consultas extremamente rápidas.

---

## Exemplos

Entre os bancos vetoriais mais conhecidos destacam-se:

- ChromaDB
- FAISS
- Milvus
- Pinecone
- Weaviate
- Qdrant
- LanceDB
- pgvector (PostgreSQL)

Cada um implementa diferentes algoritmos de indexação.

---

# Capítulo 15 — Approximate Nearest Neighbors (ANN)

## Busca Exata

Uma busca exata compara o vetor da pergunta com absolutamente todos os documentos.

```text
Pergunta

↓

Doc1

↓

Doc2

↓

Doc3

↓

...

↓

Doc10 milhões
```

É precisa.

Entretanto é lenta.

---

## Busca Aproximada

A ideia consiste em encontrar documentos extremamente próximos sem necessariamente comparar todos.

Isso reduz drasticamente o tempo.

Exemplo.

Busca exata:

5 segundos.

Busca aproximada:

20 milissegundos.

Com perda mínima de qualidade.

Essa abordagem recebe o nome:

Approximate Nearest Neighbors.

---

# Capítulo 16 — FAISS

O FAISS foi desenvolvido pelo Facebook AI Research.

É atualmente uma das bibliotecas mais utilizadas para busca vetorial.

Seu objetivo é indexar milhões ou bilhões de embeddings.

---

## Funcionamento

Fluxo simplificado.

```text
Embeddings

↓

Indexação

↓

Estrutura otimizada

↓

Consulta rápida
```

O usuário apenas fornece um vetor.

O FAISS retorna os vetores mais próximos.

---

## Tipos de Índice

O FAISS implementa diversos algoritmos.

Entre eles:

- Flat
- IVF
- PQ
- HNSW
- OPQ
- IVFPQ

Cada um apresenta diferentes compromissos entre memória, velocidade e precisão.

---

# Capítulo 17 — IVF

IVF significa

**Inverted File Index.**

---

## Ideia

Em vez de procurar em todos os documentos, primeiro divide-se o espaço vetorial em regiões.

```text
Espaço

↓

Cluster 1

Cluster 2

Cluster 3

...

Cluster N
```

Durante a consulta:

Pergunta

↓

Descobre cluster

↓

Pesquisa apenas nele.

Isso reduz drasticamente o custo computacional.

---

## Clustering

A divisão normalmente é realizada utilizando K-Means.

Objetivo:

\[
\min
\sum
||x-c_i||^2
\]

onde

\(c_i\)

representa o centro do cluster.

---

# Capítulo 18 — HNSW

HNSW significa:

Hierarchical Navigable Small World Graph.

É atualmente um dos algoritmos mais eficientes para busca vetorial.

---

## Ideia

Os vetores tornam-se vértices de um grafo.

Cada vetor conecta-se apenas aos vizinhos mais próximos.

Durante a consulta:

```text
Entrada

↓

Nível Superior

↓

Nível Intermediário

↓

Nível Inferior

↓

Documento
```

Em vez de comparar milhões de vetores, navega-se pelo grafo.

---

## Vantagens

- extremamente rápido;
- excelente precisão;
- baixa latência;
- escalável.

Por isso tornou-se padrão em diversos bancos vetoriais modernos.

---

# Capítulo 19 — Product Quantization (PQ)

Embora embeddings ocupem pouca memória individualmente, bilhões deles tornam-se extremamente caros.

Exemplo.

1536 dimensões

×

4 bytes

≈ 6 KB por vetor.

Para um bilhão de documentos seriam necessários vários terabytes.

---

## Solução

Product Quantization.

O vetor é dividido em pequenos blocos.

Cada bloco é representado por um código muito menor.

Exemplo.

```text
1536 números

↓

96 blocos

↓

Código compacto
```

Assim torna-se possível armazenar bilhões de embeddings em memória.

---

# Capítulo 20 — Pipeline Completo do RAG

Agora podemos compreender todo o funcionamento.

```text
Documentos

↓

OCR

↓

Limpeza

↓

Chunking

↓

Embeddings

↓

Banco Vetorial

↓

Pergunta

↓

Embedding da Pergunta

↓

Busca Vetorial

↓

Top-K Documentos

↓

Reranking

↓

Prompt Final

↓

LLM

↓

Resposta
```

Cada etapa influencia diretamente a qualidade final.

---

# Capítulo 21 — Top-K Retrieval

Normalmente não se retorna apenas um documento.

Selecionam-se os K melhores.

Exemplo.

```text
Top 5

↓

Top 10

↓

Top 20
```

Esses documentos são enviados ao modelo.

O valor de K depende:

- contexto disponível;
- tamanho dos documentos;
- custo;
- modelo utilizado.

---

# Capítulo 22 — Reranking

A busca vetorial recupera documentos candidatos.

Entretanto, nem sempre sua ordem é ideal.

Por isso utiliza-se um segundo modelo chamado **Reranker**.

Fluxo.

```text
Busca Vetorial

↓

20 documentos

↓

Cross Encoder

↓

Nova ordenação

↓

Top 5
```

O reranker é mais lento.

Entretanto produz qualidade significativamente superior.

---

# Capítulo 23 — Busca Híbrida

Nem toda informação depende apenas da semântica.

Às vezes o usuário procura:

```text
Erro ORA-12541
```

Nesse caso, busca lexical pode ser melhor.

Por isso muitos sistemas utilizam:

Busca Vetorial

+

BM25

+

Busca por palavras-chave

Essa abordagem recebe o nome:

Hybrid Search.

Ela combina similaridade semântica e similaridade lexical.

---

# Capítulo 24 — Prompt Final

Após recuperar os documentos, monta-se o prompt.

Exemplo.

```text
Você é um especialista.

Utilize exclusivamente os documentos abaixo.

Caso a informação não esteja presente, responda que não encontrou evidências.

Documentos:

...

Pergunta:

...
```

Observe que o LLM continua sendo responsável pela geração da resposta.

O RAG apenas fornece contexto adicional.

---

# Capítulo 25 — Benefícios do RAG

Entre as principais vantagens destacam-se:

- redução significativa de alucinações;
- acesso a documentos privados;
- atualização imediata da base de conhecimento;
- dispensa novo treinamento do modelo;
- maior rastreabilidade das respostas;
- possibilidade de citar fontes;
- menor custo comparado ao fine-tuning contínuo.

Essas características fizeram do RAG um dos pilares da IA Generativa moderna.

Entretanto, sistemas atuais evoluíram ainda mais. Em vez de apenas consultar documentos, passaram a utilizar ferramentas externas, múltiplos agentes especializados e protocolos padronizados de comunicação. Essa nova geração será apresentada na próxima parte, abordando **Agentic AI**, **Model Context Protocol (MCP)** e a colaboração entre agentes inteligentes.