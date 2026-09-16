# Relatório de Análise Exploratória de Dados (EDA) — Projeto ASTRA
**Cadeira:** Projeto Interdisciplinar de Sistemas de Informação 3 (PISI 3)  
**Instituição:** Universidade Federal Rural de Pernambuco (UFRPE)  
**Base Analisada:** `global_university_students_performance_habits_10000.csv`  
**Amostra:** 10.000 estudantes universitários | 27 atributos originais  

---

## 1. Sumário Executivo

Este documento apresenta os resultados da Análise Exploratória de Dados (EDA) conduzida sobre a base de 10.000 estudantes universitários para fundamentar as decisões de produto e modelagem preditiva da plataforma **ASTRA**.

O objetivo primordial foi identificar como variáveis comportamentais (dedicação aos estudos, sono, estresse, frequência às aulas, tempo de tela e uso de tecnologias) se relacionam com as métricas de rendimento acadêmico: **GPA** (média ponderada geral), **final_exam_score** (exame final) e **assignment_score** (trabalhos práticos).

### Principais Conclusões em Destaque:
1. **Os Dois Maiores Impulsionadores do GPA:**
   - **Frequência às aulas ($r = +0.744$):** É o indicador de maior poder associativo na base.
   - **Horas de estudo diário ($r = +0.717$):** O esforço diário consistente é o segundo maior fator de sucesso.
   - Há ainda forte correlação entre frequência e estudo ($r = +0.697$), evidenciando um padrão de engajamento conjunto.
2. **Os Três Maiores Detratores do Rendimento:**
   - **Nível de estresse mental ($r = -0.274$):** Tensão elevada reduz a capacidade cognitiva e impacta negativamente o GPA.
   - **Redes sociais ($r = -0.233$):** Principal elemento dispersivo do tempo de tela diário ($r = +0.770$ com tempo de tela).
   - **Tempo total de tela ($r = -0.177$):** Sobrecarga digital associada à redução do foco.
3. **Mitos Desmistificados:**
   - **Videogames ($r = -0.003$):** O tempo jogando games não apresenta impacto mensurável no GPA (correlação estritamente nula).
   - **Consumo de café ($r = -0.002$):** Não tem efeito direto no desempenho nem no total de sono ($r = -0.006$).
   - **Uso isolado de IA ($r = +0.032$):** O número de horas de uso de IA não eleva o desempenho por si só; sem frequência e estudo contínuo, a tecnologia não produz ganhos de GPA.
4. **Efeito Teto (*Ceiling Effect*):**
   - **61,2% dos estudantes** (6.124 alunos) obtiveram nota máxima (100) no exame final, gerando truncamento severo na cauda superior. Por conta disso, o **GPA contínuo (0 a 4.0)** é a métrica mais confiável e sensível para modelagem analítica e preditiva.
5. **Mitigação do Estresse:**
   - Horas adequadas de sono ($r = -0.310$) e prática de atividade física regular ($r = -0.296$) correlacionam-se negativamente com o estresse, funcionando como escudos protetivos contra o declínio de desempenho.

---

## 2. Dicionário Resumido & Estrutura da Base

O dataset é composto por 10.000 instâncias sem duplicatas e com 27 atributos distribuídos em 5 eixos:
- **Demografia:** `student_id`, `age`, `gender`, `country`, `major`, `university_year`, `family_income_level`, `relationship_status`.
- **Desempenho Acadêmico:** `GPA`, `final_exam_score`, `assignment_score`, `class_attendance_percent`.
- **Rotina de Estudo:** `study_hours_per_day`, `note_taking_method`, `exam_preparation_days`.
- **Estilo de Vida & Bem-Estar:** `sleep_hours`, `exercise_hours_per_week`, `mental_stress_level`, `coffee_consumption_per_day`, `part_time_job`, `extracurricular_hours_per_week`.
- **Hábitos Digitais:** `screen_time_hours`, `social_media_hours`, `gaming_hours`, `internet_quality`, `AI_tool_usage_hours`, `favorite_AI_tool`.

---

## 3. Diagnóstico de Qualidade dos Dados & Valores Ausentes

A integridade do conjunto é expressivamente alta, com **26 das 27 colunas sem nenhum valor nulo**. Apenas uma coluna apresenta registros faltantes:
- `favorite_AI_tool`: **1.720 valores ausentes (17,20%)**.

### Investigação dos Nulos:
Ao cruzar `favorite_AI_tool` ausente com as horas diárias de uso (`AI_tool_usage_hours`):
- **1.276 estudantes** têm `AI_tool_usage_hours == 0.0` (não utilizam ferramentas de IA).
- **444 estudantes** registram uso de IA ($0.1\text{ h}$ a $0.2\text{ h/dia}$), mas não indicaram uma ferramenta predileta.

> **Decisão Metodológica:** Não imputar indiscriminadamente a categoria `"Nenhuma"` ou moda para todos os nulos, pois 444 desses indivíduos são usuários ativos. No módulo [`data/limpeza.py`](file:///c:/Users/renato/Documents/UFRPE/pisi3/data/limpeza.py), a coluna foi codificada em status distinguindo `"Não Utiliza IA"` de `"Usuário sem Favorita"`.

---

## 4. Estatísticas Descritivas das Variáveis Numéricas

| Variável | Média | Desv. Padrão | Mínimo | Q1 (25%) | Mediana (50%) | Q3 (75%) | Máximo | IQR | Assimetria (Skewness) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`age`** | 22.95 | 3.16 | 18.00 | 20.00 | 23.00 | 26.00 | 28.00 | 6.00 | +0.03 |
| **`university_year`** | 2.51 | 1.12 | 1.00 | 1.75 | 3.00 | 4.00 | 4.00 | 2.25 | -0.01 |
| **`GPA`** | **3.62** | **0.40** | **1.92** | **3.36** | **3.71** | **4.00** | **4.00** | **0.64** | **-0.96** |
| **`study_hours_per_day`** | 3.51 | 1.47 | 0.50 | 2.50 | 3.50 | 4.50 | 9.40 | 2.00 | +0.13 |
| **`class_attendance_percent`** | 90.49 | 9.95 | 44.00 | 84.00 | 93.00 | 100.00 | 100.00 | 16.00 | -0.92 |
| **`sleep_hours`** | 7.01 | 1.19 | 3.50 | 6.20 | 7.00 | 7.80 | 10.00 | 1.60 | -0.04 |
| **`screen_time_hours`** | 5.79 | 1.88 | 1.00 | 4.50 | 5.80 | 7.10 | 12.00 | 2.60 | +0.06 |
| **`social_media_hours`** | 2.80 | 1.45 | 0.00 | 1.70 | 2.80 | 3.80 | 8.00 | 2.10 | +0.15 |
| **`gaming_hours`** | 1.58 | 1.16 | 0.00 | 0.60 | 1.50 | 2.30 | 6.40 | 1.70 | +0.51 |
| **`exercise_hours_per_week`** | 3.07 | 1.88 | 0.00 | 1.70 | 3.00 | 4.40 | 10.00 | 2.70 | +0.26 |
| **`mental_stress_level`** | 4.06 | 1.28 | 1.00 | 3.20 | 4.10 | 4.90 | 9.00 | 1.70 | +0.02 |
| **`AI_tool_usage_hours`** | 1.26 | 0.91 | 0.00 | 0.50 | 1.20 | 1.90 | 5.00 | 1.40 | +0.46 |
| **`exam_preparation_days`** | 12.04 | 4.96 | 1.00 | 9.00 | 12.00 | 15.00 | 30.00 | 6.00 | +0.09 |
| **`coffee_consumption_per_day`** | 1.80 | 1.34 | 0.00 | 1.00 | 2.00 | 3.00 | 8.00 | 2.00 | +0.70 |
| **`extracurricular_hours_per_week`** | 4.02 | 1.94 | 0.00 | 2.70 | 4.00 | 5.30 | 11.40 | 2.60 | +0.10 |
| **`final_exam_score`** | **96.05** | **6.82** | **56.00** | **94.00** | **100.00** | **100.00** | **100.00** | **6.00** | **-2.01** |
| **`assignment_score`** | **92.96** | **7.27** | **60.00** | **89.00** | **95.00** | **100.00** | **100.00** | **11.00** | **-1.03** |

---

## 5. As 36 Correlações de Pearson Planejadas

Foram computadas 36 correlações lineares cruzando os **12 hábitos mensurados** com as **3 variáveis de desempenho acadêmico**:

| Hábito Avaliado | $r$ com GPA | $r$ com Exame Final | $r$ com Trabalhos | Classificação de Força (GPA) |
| :--- | :---: | :---: | :---: | :--- |
| **`class_attendance_percent`** | **+0.744** | **+0.611** | **+0.573** | **Positiva Forte** |
| **`study_hours_per_day`** | **+0.717** | **+0.593** | **+0.488** | **Positiva Forte** |
| **`sleep_hours`** | **+0.232** | **+0.100** | **+0.126** | **Positiva Fraca/Moderada** |
| **`exercise_hours_per_week`** | +0.021 | +0.005 | +0.008 | Nula / Desprezível direta |
| **`AI_tool_usage_hours`** | +0.032 | +0.006 | +0.011 | Nula / Desprezível direta |
| **`extracurricular_hours_per_week`** | +0.001 | -0.007 | +0.010 | Nula / Desprezível |
| **`exam_preparation_days`** | -0.000 | -0.004 | -0.002 | Nula / Desprezível |
| **`gaming_hours`** | -0.003 | +0.004 | -0.002 | Nula / Desprezível |
| **`coffee_consumption_per_day`** | -0.002 | -0.011 | -0.014 | Nula / Desprezível |
| **`screen_time_hours`** | **-0.177** | -0.079 | -0.107 | **Negativa Fraca** |
| **`social_media_hours`** | **-0.233** | -0.103 | -0.139 | **Negativa Moderada** |
| **`mental_stress_level`** | **-0.274** | -0.120 | -0.155 | **Negativa Moderada** |

---

## 6. Resposta Formal às Questões Norteadoras de Pesquisa

### Questão 1: Como o equilíbrio entre sono e estresse modula o impacto das horas de estudo no GPA?
> **Hipótese ASTRA:** Estudo extenuante sem sono e sob estresse alto gera saturação cognitiva e perda de rendimento.

**Evidência dos Dados:**
Ao segmentar os estudantes que mais estudam ($\ge 4.5\text{ h/dia}$, quartil superior):
- **Cenário A (Alta dedicação + Sono $\ge 7\text{h}$ + Baixo Estresse $\le 3.5$):**  
  $\to$ **GPA Médio: 3.98** (Desvio Padrão: 0.06 | $N = 616$ alunos).
- **Cenário B (Alta dedicação + Sono crítico $< 6\text{h}$ + Alto Estresse $> 5.0$):**  
  $\to$ **GPA Médio: 3.78** (Desvio Padrão: 0.25 | $N = 199$ alunos).

**Conclusão:** Mesmo dedicando as mesmas horas aos livros, alunos com privação de sono e estresse sofrem uma penalidade de cerca de **0,20 ponto no GPA**, com variabilidade substancialmente maior. Dormir bem e gerenciar o estresse é condição mandatória para materializar o esforço em notas altas.

---

### Questão 2: Qual a correlação do uso de Inteligência Artificial no desempenho quando contrastado com tempo em redes sociais e frequência às aulas?

**Evidência dos Dados:**
- Estudantes com alto consumo de IA ($\ge 3\text{ h/dia}$) têm GPA médio de **3.63**, enquanto estudantes que não utilizam IA (0 h/dia) têm GPA médio de **3.59**.
- A correlação linear de IA com GPA é de apenas **$r = +0.032$**.
- Em contraste, a presença às aulas possui $r = \mathbf{+0.744}$ e as redes sociais $r = \mathbf{-0.233}$.

**Conclusão:** O uso de IA generativa por si só não atua como atalho para excelência acadêmica. O tempo desperdiçado em redes sociais degrada as notas muito mais do que o uso de IA é capaz de compensar.

---

### Questão 3: Relações Hábito a Hábito e Vias Indiretas de Impacto
- **Sono e Estresse ($r = -0.310$):** Mais horas de sono reduzem os níveis de estresse.
- **Exercício e Estresse ($r = -0.296$):** A prática esportiva não aumenta notas diretamente, mas atua como amortecedor de estresse, preservando indiretamente o rendimento.
- **Redes Sociais e Tempo de Tela ($r = +0.770$):** Redes sociais são o componente preponderante da exposição a telas.
- **Café e Sono ($r = -0.006$):** Não foi observada relação linear entre xícaras de café e redução de horas de sono na amostra.

---

## 7. Análise de Variáveis Demográficas e Categóricas

- **Método de Anotação (`note_taking_method`):**
  - Digital: GPA médio = 3.62 | Exame final = 96.09
  - Handwritten (Caderno): GPA médio = 3.62 | Exame final = 96.17
  - Mixed (Misto): GPA médio = 3.61 | Exame final = 95.91
  - *Conclusão:* A metodologia de anotação não cria disparidades significativas no rendimento.
- **Renda Familiar (`family_income_level`):**
  - Alta (High): GPA 3.62 | Exame 96.16
  - Média (Middle): GPA 3.62 | Exame 96.07
  - Baixa (Low): GPA 3.61 | Exame 95.95
- **Trabalho Meio Período (`part_time_job`):**
  - Trabalha (Yes): GPA 3.62 | Exame 96.19
  - Não Trabalha (No): GPA 3.62 | Exame 95.98
- **Gênero (`gender`):**
  - Feminino: GPA 3.61 | Exame 95.93
  - Masculino: GPA 3.62 | Exame 96.18

*Síntese Demográfica:* A base apresenta características controladas e distribuição equilibrada entre subgrupos, indicando que as variações no desempenho derivam primordialmente de hábitos comportamentais (frequência, estudo diário, sono e estresse).

---

## 8. Implicações para a Arquitetura do Produto ASTRA

1. **Feature de Alerta de Sono e Estresse:** Criar no dashboard indicadores de saturação cognitiva que alertem o estudante quando as horas de estudo aumentarem sem o correspondente descanso noturno ($\ge 7\text{h}$).
2. **Priorização de Foco:** Estimular o controle de tempo em redes sociais antes de tentar aumentar as horas brutas de estudo.
3. **Métrica Alvo Central:** Utilizar o GPA como variável contínua para modelagem preditiva e segmentação (K-Means), mitigando os vieses de teto do exame final.
