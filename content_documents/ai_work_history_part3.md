# Parte 3 — Reinforcement Learning e Retrieval-Augmented Generation (RAG) — Fundamentos

---

# Capítulo 9 — Reinforcement Learning

## Introdução

Até este ponto foram apresentados diversos paradigmas de aprendizado utilizados pela Inteligência Artificial moderna:

- Machine Learning supervisionado;
- Deep Learning;
- Transfer Learning;
- Zero-Shot Learning;
- Few-Shot Learning;
- Self-Supervised Learning.

Todos eles possuem uma característica em comum: o algoritmo aprende a partir de exemplos previamente existentes.

Entretanto, muitos problemas não possuem exemplos corretos.

Como ensinar um robô a andar?

Como ensinar um carro autônomo a dirigir?

Como ensinar uma IA a jogar xadrez melhor que qualquer humano?

Nesses casos, não existe um conjunto de respostas corretas prontas.

A solução foi criar um paradigma completamente diferente:

**Reinforcement Learning (RL).**

Ao invés de aprender com exemplos, o agente aprende através da interação com o ambiente.

---

# O conceito de aprendizado por recompensa

Imagine ensinar uma criança a andar de bicicleta.

Você não fornece milhares de exemplos.

Em vez disso:

- ela tenta;
- cai;
- levanta;
- tenta novamente;
- recebe incentivo quando acerta.

Esse processo é exatamente o princípio do Reinforcement Learning.

O agente aprende por tentativa e erro.

Sempre que realiza uma boa ação recebe uma recompensa.

Quando realiza uma ação ruim recebe uma penalidade.

Após milhares ou milhões de tentativas, aprende uma política capaz de maximizar sua recompensa acumulada.

---

# Componentes do Reinforcement Learning

Todo problema de RL possui cinco componentes principais.

## Agente

É quem toma decisões.

Exemplos:

- robô;
- carro autônomo;
- personagem de videogame;
- modelo de IA.

---

## Ambiente

Representa tudo com que o agente interage.

Exemplos:

- estrada;
- jogo;
- fábrica;
- internet;
- usuário.

---

## Estado

Representa a situação atual do ambiente.

Exemplo:

Um carro autônomo pode definir como estado:

- velocidade;
- posição;
- distância para outros veículos;
- faixa atual;
- sinalização.

Formalmente:

\[
s_t
\]

representa o estado no instante \(t\).

---

## Ação

Cada decisão possível do agente.

Exemplo:

```text
Acelerar

Frear

Virar à esquerda

Virar à direita

Manter velocidade
```

Formalmente:

\[
a_t
\]

---

## Recompensa

Após cada ação o ambiente fornece um valor numérico.

\[
r_t
\]

Exemplo:

```text
+10
Chegar ao destino

-100
Colidir

-1
Consumir combustível

+2
Economizar energia
```

A recompensa define o comportamento desejado.

---

# Objetivo matemático

O objetivo do Reinforcement Learning consiste em maximizar a recompensa futura.

Define-se:

\[
R_t
=
\sum_{k=0}^{\infty}
\gamma^k
r_{t+k}
\]

onde

- \(r\) é a recompensa;
- \(\gamma\) é o fator de desconto.

Quanto maior o desconto, mais importância o agente dá ao futuro.

---

# Cadeias de Markov

Grande parte do RL baseia-se nos chamados:

**Markov Decision Processes (MDP).**

Um MDP é composto por:

\[
(S,A,P,R,\gamma)
\]

onde

- S = conjunto de estados;
- A = ações;
- P = probabilidades de transição;
- R = recompensas;
- γ = desconto.

---

# Política

A política determina qual ação executar.

Formalmente:

\[
\pi(a|s)
\]

É a probabilidade de escolher uma ação dado um estado.

O treinamento procura encontrar:

\[
\pi^*
\]

ou seja,

a política ótima.

---

# Exploração versus Exploração

Um dos maiores desafios do RL.

O agente deve decidir:

## Explorar

Tentar algo novo.

Ou

## Explorar conhecimento

Executar aquilo que já sabe.

Exemplo.

Imagine um robô aprendendo a caminhar.

Ele pode:

```text
andar reto

↓

ganha recompensa
```

Mas talvez exista uma forma melhor.

Se nunca experimentar novos movimentos, jamais descobrirá.

Esse equilíbrio é conhecido como:

**Exploration vs Exploitation.**

---

# Q-Learning

Um dos algoritmos clássicos.

Define uma tabela:

\[
Q(s,a)
\]

representando a qualidade de executar uma ação em determinado estado.

Atualização:

\[
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
[
r
+
\gamma
\max_aQ(s',a)
-
Q(s,a)
]
\]

onde

α é a taxa de aprendizado.

---

# Deep Reinforcement Learning

Quando estados tornam-se muito grandes, tabelas deixam de ser viáveis.

Passa-se então a utilizar redes neurais.

Assim surgiu o Deep Reinforcement Learning.

A rede aprende diretamente:

\[
Q(s,a)
\]

ou

\[
\pi(a|s)
\]

---

# Grandes marcos históricos

## DeepMind Atari

Em 2015 a DeepMind demonstrou que uma única rede neural conseguia aprender dezenas de jogos Atari apenas observando pixels.

Sem programação específica.

---

## AlphaGo

Em 2016 ocorreu um marco histórico.

O AlphaGo derrotou Lee Sedol.

Até então acreditava-se que o jogo Go seria impossível para computadores.

A combinação de:

- Deep Learning;
- Monte Carlo Tree Search;
- Reinforcement Learning

mudou completamente essa percepção.

---

## AlphaZero

Posteriormente surgiu AlphaZero.

Recebeu apenas:

```text
Regras do jogo
```

Aprendeu sozinho:

- Go
- Xadrez
- Shogi

Superando todos os campeões humanos.

---

# RLHF

Os Large Language Models modernos utilizam uma adaptação conhecida como:

**Reinforcement Learning from Human Feedback (RLHF).**

O treinamento ocorre em três etapas.

## Etapa 1

Self-Supervised Learning.

Treinamento na internet.

---

## Etapa 2

Supervised Fine-Tuning.

Humanos escrevem respostas ideais.

---

## Etapa 3

RLHF.

Humanos classificam respostas.

Exemplo.

Pergunta:

```text
Como funciona SQL?
```

Resposta A

Muito boa.

Resposta B

Pouco detalhada.

O modelo aprende a preferir respostas semelhantes à A.

---

# PPO

O algoritmo mais conhecido para RLHF é:

Proximal Policy Optimization.

Ele impede mudanças muito bruscas na política.

Isso estabiliza o treinamento.

---

# Limitações

Apesar do enorme sucesso, RL apresenta dificuldades.

Entre elas:

- treinamento caro;
- milhões de interações;
- definição difícil da recompensa;
- instabilidade;
- alto custo computacional.

Mesmo assim, tornou-se peça fundamental dos LLMs modernos.

---

# Capítulo 10 — Retrieval-Augmented Generation (RAG)

## Introdução

Embora os Large Language Models sejam extremamente poderosos, eles possuem uma limitação importante.

Seu conhecimento está limitado ao período em que foram treinados.

Além disso:

- não conhecem documentos privados;
- não acessam bancos internos;
- não sabem informações recém-publicadas;
- podem alucinar respostas.

Treinar novamente um modelo sempre que um documento muda seria inviável.

Surge então o paradigma conhecido como:

**Retrieval-Augmented Generation (RAG).**

---

# O que é RAG?

Retrieval-Augmented Generation consiste em combinar um modelo de linguagem com um sistema de recuperação de documentos.

Em vez de responder apenas utilizando seus parâmetros internos, o modelo primeiro consulta uma base de conhecimento.

Somente depois gera sua resposta.

Fluxo geral.

```text
Pergunta

↓

Busca documentos

↓

Seleciona trechos relevantes

↓

Envia contexto ao LLM

↓

Resposta
```

---

# Vantagens

Entre os principais benefícios destacam-se:

- conhecimento atualizado;
- documentos privados;
- menor alucinação;
- maior confiabilidade;
- facilidade de manutenção.

Não é necessário treinar novamente o modelo.

Basta atualizar a base documental.

---

# Exemplo

Imagine uma empresa.

Possui:

- contratos;
- normas internas;
- manuais;
- especificações técnicas.

Pergunta:

```text
Qual é o procedimento para solicitar férias?
```

O LLM não conhece o regulamento interno.

Entretanto o RAG localiza:

```text
Manual RH

Capítulo 12

Solicitação de férias
```

Esse trecho é enviado ao modelo.

A resposta passa a ser baseada no documento oficial.

---

# Arquitetura geral

Um sistema RAG normalmente possui os seguintes componentes.

```text
Documentos

↓

Pré-processamento

↓

Chunking

↓

Embeddings

↓

Banco Vetorial

↓

Busca Semântica

↓

LLM

↓

Resposta
```

Cada etapa será detalhada nos próximos tópicos.

---

# Etapa 1 — Coleta de documentos

Os documentos podem vir de diversas fontes.

Exemplos:

- PDFs;
- Word;
- planilhas;
- páginas web;
- bancos SQL;
- APIs;
- SharePoint;
- Notion;
- GitHub;
- Confluence.

Todos são convertidos para texto.

---

# Etapa 2 — Limpeza

Remove-se:

- cabeçalhos;
- rodapés;
- numeração;
- caracteres especiais;
- OCR incorreto;
- espaços duplicados.

Essa etapa melhora significativamente a qualidade dos embeddings.

---

# Etapa 3 — Chunking

Documentos grandes não podem ser enviados integralmente ao LLM.

Eles precisam ser divididos.

Esses blocos recebem o nome de:

**Chunks.**

Exemplo.

Documento:

```text
100 páginas
```

↓

Divide-se em

```text
500 trechos
```

Cada trecho normalmente possui:

- 300 tokens;
- 500 tokens;
- 1000 tokens.

Dependendo do modelo utilizado.

---

# Estratégias de Chunking

Existem diversas formas.

## Fixed Size

Divide por tamanho fixo.

```text
500 tokens
```

---

## Sentence Chunking

Divide respeitando frases.

---

## Paragraph Chunking

Mantém parágrafos inteiros.

---

## Semantic Chunking

Utiliza IA para detectar mudanças de assunto.

Essa estratégia produz resultados superiores.

---

# Overlap

Normalmente utiliza-se sobreposição entre chunks.

Exemplo.

Chunk 1

```text
tokens 1–500
```

Chunk 2

```text
tokens 450–950
```

Isso evita perda de contexto.

---

# Próximos capítulos

Até este ponto foi apresentada a arquitetura geral do RAG.

Na Parte 4 serão explorados os aspectos matemáticos que tornam esse paradigma possível, incluindo:

- embeddings;
- vetorização de documentos;
- tokenização;
- espaço vetorial;
- similaridade por cosseno;
- distância euclidiana;
- produto interno;
- bancos vetoriais;
- FAISS;
- HNSW;
- IVF;
- Product Quantization (PQ);
- Approximate Nearest Neighbors (ANN);
- reranking;
- recuperação híbrida;
- geração final utilizando LLMs.

Esses conceitos formam o núcleo matemático e computacional dos sistemas RAG modernos e são fundamentais para compreender como grandes volumes de informação podem ser pesquisados de forma eficiente e utilizados para aumentar significativamente a qualidade das respostas produzidas por modelos de linguagem.