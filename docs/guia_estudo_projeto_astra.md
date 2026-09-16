# ASTRA Analytics — Guia Definitivo de Estudos e Apresentação do Projeto
**Cadeira:** Projeto Interdisciplinar de Sistemas de Informação 3 (PISI 3) — UFRPE  
**Objetivo:** Manual técnico e conceitual para alinhamento da equipe, domínio da base de código, embasamento estatístico e defesa do projeto perante a banca.

---

## 1. Visão Geral: O que é o ASTRA e qual problema ele resolve?

### 1.1 A Dor do Estudante Universitário
Na vida universitária moderna, impera a crença simplista de que *"quanto mais horas você estudar, melhores serão seus resultados"*. No entanto, a realidade observada nos campi é marcada por sobrecarga cognitiva, noites viradas em claro, níveis alarmantes de estresse mental e dispersão digital (redes sociais e excesso de telas).

### 1.2 A Proposta de Valor do ASTRA
O **ASTRA (Ambiente de Suporte, Tendências e Rendimento Acadêmico)** é uma plataforma analítica e preditiva projetada para desmistificar o comportamento discente. Em vez de simplesmente cobrar mais horas de dedicação, o ASTRA cruza hábitos diários (sono, frequência, atividade física, café, uso de IA e redes sociais) para responder:
- **Qual é o ponto de retorno decrescente do esforço?**
- **Dormir bem e gerenciar o estresse melhora mais o GPA do que estudar sem descanso?**
- **O uso de Inteligência Artificial realmente melhora o desempenho ou atua como dispersão?**

---

## 2. O Dataset: 10.000 Alunos e 27 Variáveis

O projeto analisa a base canônica `global_university_students_performance_habits_10000.csv`, localizada em `Data/`.

### 2.1 Eixos Temáticos dos Dados
1. **Identificação & Demografia:** `student_id`, `age` (18-28 anos), `gender`, `country` (13 países), `major` (10 cursos), `university_year` (1º ao 4º ano), `family_income_level` (Baixa/Média/Alta), `relationship_status`.
2. **Desempenho Acadêmico:**
   - **`GPA`:** Média geral ponderada (escala de 0.0 a 4.0). É a métrica mais confiável e sensível.
   - **`final_exam_score`:** Nota do exame final (56 a 100 pontos).
   - **`assignment_score`:** Nota dos trabalhos e projetos práticos (60 a 100 pontos).
   - **`class_attendance_percent`:** Taxa de frequência presencial (44% a 100%).
3. **Hábitos de Estudo:** `study_hours_per_day`, `note_taking_method` (Digital, Caderno, Misto), `exam_preparation_days`.
4. **Bem-Estar & Saúde:** `sleep_hours`, `mental_stress_level` (1 a 10), `exercise_hours_per_week`, `coffee_consumption_per_day`, `part_time_job`, `extracurricular_hours_per_week`.
5. **Hábitos Digitais:** `screen_time_hours`, `social_media_hours`, `gaming_hours`, `internet_quality`, `AI_tool_usage_hours`, `favorite_AI_tool`.

### 2.2 Diagnóstico de Qualidade: O Caso dos Nulos em `favorite_AI_tool`
- Das 27 variáveis, **26 têm zero valores nulos**.
- Apenas `favorite_AI_tool` possui **1.720 registros ausentes (17,2%)**.
- **O que descobrimos no EDA?**
  - Dos 1.720 alunos com ferramenta favorita nula, **1.276 têm uso de IA igual a 0.0h** (não utilizam IA).
  - Porém, **444 estudantes utilizam IA (0.1h a 0.2h/dia)**, mas não informaram uma ferramenta preferida.
- **Por que isso importa na apresentação?** Mostra rigor metodológico. Explicar à banca que **não imputamos "Nenhuma" às cegas**, pois distorceríamos a realidade dos 444 estudantes que de fato utilizam ferramentas de IA. No código (`Data/limpeza.py`), diferenciamos `"Não Utiliza IA"` de `"Usuário sem Favorita"`.

### 2.3 O Fenômeno Estatístico do Efeito Teto (*Ceiling Effect*)
- No exame final (`final_exam_score`), a média é 96,05 e a mediana é 100. **61,2% de todos os alunos obtiveram nota 100**.
- Nos trabalhos (`assignment_score`), **28,8% obtiveram nota 100**.
- **Por que isso é fundamental dominar?**
  - Quando a variável de nota atinge o teto máximo da escala, perde-se variabilidade na cauda superior. Isso enfraquece relações lineares diretas com outras variáveis.
  - Por essa razão técnica, **o GPA (que varia continuamente até 4.0 sem compressão extrema) é a nossa variável-alvo prioritária**.

---

## 3. As 36 Correlações de Pearson e Perguntas de Pesquisa

Calculamos o coeficiente linear de Pearson ($r$, variando de $-1$ a $+1$) cruzando os 12 hábitos contra as 3 métricas de rendimento.

### 3.1 O que os Dados Revelam:

| Hábito | $r$ com GPA | Interpretação Científica |
| :--- | :---: | :--- |
| **Frequência às Aulas** | **+0.744** | **Preditivo #1:** Estar presencialmente em sala é o maior determinante de sucesso. |
| **Horas de Estudo Diário** | **+0.717** | **Preditivo #2:** Dedicação contínua dia após dia constrói notas altas. |
| **Horas de Sono** | **+0.232** | **Aliado Positivo:** Dormir mais horas por noite correlaciona-se com melhor rendimento. |
| **Nível de Estresse** | **-0.274** | **Detrator #1:** Ansiedade e sobrecarga cognitiva reduzem o GPA. |
| **Redes Sociais** | **-0.233** | **Detrator #2:** Dispersão contínua degrada o foco e reduz as notas. |
| **Tempo Total de Tela** | **-0.177** | **Detrator #3:** Sobrecarga digital. |
| **Uso de IA** | **+0.032** | **Praticamente Nulo:** Usar IA sem frequência e estudo não gera ganhos de GPA. |
| **Videogames** | **-0.003** | **Mito Quebrado:** Jogos eletrônicos não possuem correlação mensurável com queda de rendimento. |
| **Café Diário** | **-0.002** | **Mito Quebrado:** Consumo de café não afeta o GPA nem reduz o tempo de sono ($r = -0.006$). |

### 3.2 Resposta às Perguntas Norteadoras da Pesquisa

#### Pergunta 1: O sono e o estresse modulam o retorno das horas de estudo?
- **Sim, categoricamente.**
- Estudantes com alta carga de estudo ($\ge 4.5\text{ h/dia}$):
  - Com sono adequado ($\ge 7\text{h}$) e baixo estresse: **GPA médio de 3.98**.
  - Com sono crítico ($< 6\text{h}$) e alto estresse: **GPA médio de 3.78**.
- **Insight:** Há uma penalidade de $0.20$ ponto no GPA para quem tenta compensar a falta de sono estudando sob estresse. Há um ponto de saturação cognitiva.

#### Pergunta 2: A IA compensa o tempo perdido em redes sociais?
- **Não.** A correlação de redes sociais com perda de rendimento ($r = -0.233$) é muito mais forte do que a associação tênue do uso isolado de IA ($r = +0.032$). Alunos assíduos que não usam IA têm rendimento substancialmente superior àqueles que usam IA mas faltam às aulas.

#### Pergunta 3: Relações Hábito a Hábito (Vias Indiretas)
- **Exercício e Estresse ($r = -0.296$):** A atividade física não aumenta a nota diretamente, mas funciona como protetor biológico contra o estresse.
- **Sono e Estresse ($r = -0.310$):** Dormir bem reduz o estresse, protegendo indiretamente o GPA.
- **Redes Sociais e Tempo de Tela ($r = +0.770$):** O feed de redes sociais é o motor central do tempo excessivo de tela.

---

## 4. Arquitetura dos Códigos: O que está sendo feito e por que

### 4.1 Por que refatoramos o código monolítico de 940 linhas?
O código inicial concentrava leitura de arquivos, limpeza de strings, cálculos matriciais, treinamento de KMeans, estilização CSS de 150 linhas e geração de 12 gráficos Plotly em um único script procedimental sequencial.
- **Problemas:** Dificuldade de manutenção, acoplamento extremo, impossibilidade de testes unitários e alto risco de merge conflicts quando 4 desenvolvedores tentam commitar no mesmo arquivo.

### 4.2 Arquitetura Modular em 4 Camadas:

```
projeto/
│
├── Data/
│   ├── carregar.py          # Leitura resiliente com detecção dinâmica de caminhos
│   └── limpeza.py           # Tratamento metódico de nulos e engenharia de faixas
│
├── analysis/
│   ├── eda.py               # Script oficial de análise estatística e hipóteses
│   ├── estatisticas.py      # Métricas descritivas, IQR e detecção de teto
│   ├── correlacoes.py       # Matriz das 36 correlações e relações hábito-a-hábito
│   └── clusters.py          # Pipeline de Machine Learning (K-Means)
│
├── visualization/
│   ├── estilos.py           # Tokens de cor (ASTRA_COLORS), CSS e padronização
│   └── graficos.py          # Funções modulares geradoras de figuras Plotly
│
└── ASTRA_Dashboard/
    └── app.py               # Orquestrador limpo baseado em abas (st.tabs)
```

### 4.3 O que cada arquivo faz:

#### 1. `Data/carregar.py`
- **O que faz:** Função `carregar_dados_brutos()` que busca o CSV no diretório canônico `Data/`.
- **Por que faz:** Evita que scripts quebrem dependendo de onde o terminal foi aberto (seja na raiz, na pasta `docs/` ou em `ASTRA_Dashboard/`). Centraliza o acesso aos dados em uma única fonte da verdade.

#### 2. `Data/limpeza.py`
- **O que faz:** Executa `tratar_dados_ia()` e `adicionar_faixas_comportamentais()`.
- **Por que faz:** Cria variáveis ricas como `faixa_sono` (Crítico, Alerta, Adequado, Alto), `categoria_estresse` e `faixa_estudo`, além de preservar os dados originais sem mutações destrutivas.

#### 3. `analysis/eda.py`
- **O que faz:** Script executável que gera todas as estatísticas no terminal: univariadas, nulos, efeito teto, as 36 correlações e cruzamentos de cenários de sono x estresse.
- **Por que faz:** Garante reprodutibilidade científica. Qualquer professor ou colega pode rodar `python analysis/eda.py` e auditar os números instantaneamente.

#### 4. `analysis/clusters.py` (Machine Learning K-Means)
- **O que faz:**
  1. Seleciona 6 features críticas: `study_hours_per_day`, `sleep_hours`, `mental_stress_level`, `class_attendance_percent`, `social_media_hours` e `GPA`.
  2. Aplica `StandardScaler` do Scikit-Learn.
  3. Treina o modelo `KMeans(n_clusters=k, random_state=42)`.
  4. Ordena os clusters automaticamente pela média de GPA e nomeia os perfis ("Alto Desempenho & Foco", "Rendimento Equilibrado", "Risco de Burnout", etc.).
- **Por que usamos `StandardScaler`?**  
  *Atenção para a banca:* Frequência varia de 0 a 100, enquanto GPA varia de 0 a 4.0. Se não normalizássemos os dados para média 0 e desvio padrão 1, a distância euclidiana do K-Means seria dominada pela coluna de frequência, ignorando o GPA. A normalização garante que todos os hábitos tenham peso equivalente.

#### 5. `visualization/estilos.py`
- **O que faz:** Define a paleta `ASTRA_COLORS` (modo escuro com alto contraste: ciano, esmeralda, âmbar, rosa) e a função `estilizar_grafico(fig, titulo)`.
- **Por que faz:** Elimina repetição de layout Plotly em cada gráfico, garantindo consistência visual premium em todo o painel.

#### 6. `visualization/graficos.py`
- **O que faz:** Funções especializadas: `criar_scatter_com_tendencia()`, `criar_heatmap_correlacao()`, `criar_donuts_demograficos()`, `criar_boxplot_sono()`.
- **Por que faz:** Cada função tem responsabilidade única (Single Responsibility Principle).

#### 7. `ASTRA_Dashboard/app.py`
- **O que faz:** Interface visual no Streamlit. Lê os filtros da barra lateral (`st.sidebar`), filtra o DataFrame em memória e distribui a exibição em abas (`st.tabs`):
  - Aba 1: Visão Geral & KPIs
  - Aba 2: Hábitos & Correlações
  - Aba 3: Perfis Inteligentes K-Means & Simulador
  - Aba 4: Metodologia & Dicionário

---

## 5. Roteiro Sugerido para a Apresentação na Banca

Para que os 4 integrantes se destaquem de forma harmônica durante a apresentação:

### Bloco 1: Introdução, Problema e Hipóteses (Integrante 1)
- Apresentar o propósito do ASTRA e a contextualização da vida universitária.
- Apresentar as duas perguntas norteadoras da pesquisa e os objetivos gerais.
- Mostrar a relevância pedagógica e humana de cuidar do sono e estresse acadêmico.

### Bloco 2: Engenharia de Dados, Qualidade e EDA (Integrante 2)
- Apresentar a base de dados (10.000 estudantes, 27 atributos).
- Explicar a integridade dos dados e o diagnóstico rigoroso dos nulos em ferramentas de IA (diferença entre não usar e não ter favorita).
- Destacar o Efeito Teto no exame final e a justificativa para focar no GPA.

### Bloco 3: Correlações, Mitos Desmistificados e Descobertas (Integrante 3)
- Apresentar a matriz das 36 correlações de Pearson.
- Explicar o ranking de fatores: Presença (+0.74) e Estudo (+0.72) vs. Estresse (-0.27) e Redes Sociais (-0.23).
- Explicar a quebra de mitos: Videogames e Café não explicam queda de GPA; IA isolada não é atalho.
- Apresentar o gráfico de interação: como sono ruim e estresse alto degradam o retorno das horas de estudo (queda de 3.98 para 3.78).

### Bloco 4: Arquitetura, Machine Learning (K-Means) e Demonstração do Dashboard (Integrante 4)
- Explicar por que o projeto foi refatorado em uma arquitetura modular em 4 camadas.
- Explicar a modelagem de Machine Learning: por que K-Means, por que usamos `StandardScaler` e quais os perfis encontrados.
- Fazer a navegação ao vivo no Dashboard Streamlit, demonstrando a reatividade dos filtros e os KPIs.

---

## 6. Possíveis Perguntas da Banca e Como Responder

**P1: Por que vocês usaram K-Means e não um algoritmo supervisionado de classificação?**  
*Resposta:* "O objetivo na segmentação comportamental não era rotular previamente quem é 'bom' ou 'ruim', mas sim descobrir agrupamentos naturais não-supervisionados de estilo de vida entre os 10.000 alunos. O K-Means nos permitiu mapear grupos que sofrem com estresse alto mesmo tendo notas razoáveis (risco de burnout), algo que um classificador supervisionado simples não revelaria."

**P2: Por que vocês não substituíram os nulos de `favorite_AI_tool` pela ferramenta mais usada (moda)?**  
*Resposta:* "Ao cruzar a coluna ausente com as horas diárias de uso de IA, descobrimos que 74% dos nulos pertenciam a estudantes com 0.0h de uso (não usuários). Se tivéssemos imputado pela moda (ChatGPT), teríamos inserido 1.276 falsos usuários de ChatGPT na base, gerando um viés estatístico grave. Por isso, criamos categorias conscientes diferenciando não-usuários de usuários ativos sem favorita."

**P3: Por que a correlação do exame final é menor do que a do GPA com os hábitos?**  
*Resposta:* "Identificamos estatisticamente a ocorrência do Efeito Teto (*Ceiling Effect*) no exame final: mais de 61% dos estudantes atingiram a pontuação máxima de 100. O truncamento da cauda superior comprime a variância da variável, reduzindo a inclinação da reta de regressão linear. O GPA, por ser contínuo e mais distribuído, reflete com mais fidelidade a influência contínua dos hábitos diários."

---
*Documento homologado para estudo interno da equipe de PISI 3 — UFRPE.*
