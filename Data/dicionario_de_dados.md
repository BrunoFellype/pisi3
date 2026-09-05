# Dicionário de Dados — ASTRA / PISI3

**Arquivo:** `global_university_students_performance_habits_10000.csv`  
**Total de Registros:** 10.000 estudantes  
**Total de Colunas:** 27 atributos  
**Domínio:** Hábitos de estudo, estilo de vida, saúde mental e desempenho acadêmico de universitários globais.

---

## 1. Categorias de Variáveis

O dataset é estruturado em cinco dimensões fundamentais:
1. **Identificação & Demografia:** Perfil socioeconômico e pessoal dos estudantes.
2. **Desempenho Acadêmico:** Métricas oficiais de notas, presença e período letivo.
3. **Hábitos de Estudo & Aprendizagem:** Dedicação diária e metodologia adotada.
4. **Estilo de Vida & Bem-Estar:** Sono, atividade física, estresse e trabalho.
5. **Tecnologia & Hábitos Digitais:** Uso de telas, redes sociais, games e ferramentas de Inteligência Artificial.

---

## 2. Tabela Completa do Dicionário de Dados

| # | Coluna | Tipo de Dado | Descrição | Escala / Categorias | Valores (Min - Max / Exemplos) | Valores Nulos |
|---|---|---|---|---|---|---|
| 1 | `student_id` | Texto (`str`) | Identificador único do estudante | Código alfanumérico | `S00001` a `S10000` | 0 |
| 2 | `age` | Inteiro (`int64`) | Idade do estudante em anos | Anos completos | 18 a 28 anos (Média: 22,9) | 0 |
| 3 | `gender` | Categórico (`str`) | Gênero biológico/autodeclarado | `Female`, `Male` | `Female`, `Male` | 0 |
| 4 | `country` | Categórico (`str`) | País onde o aluno cursa a graduação | 13 países | `Australia`, `Bangladesh`, `Brazil`, `Canada`, `France`, `Germany`, `India`, `Italy`, `Japan`, `Netherlands`, `South Korea`, `UK`, `USA` | 0 |
| 5 | `major` | Categórico (`str`) | Curso de graduação do aluno | 10 cursos | `Biology`, `Business`, `Computer Science`, `Economics`, `Engineering`, `Law`, `Mathematics`, `Medicine`, `Physics`, `Psychology` | 0 |
| 6 | `university_year` | Inteiro (`int64`) | Ano / período acadêmico atual | 1 a 4 (1º ao 4º ano) | 1, 2, 3, 4 | 0 |
| 7 | `GPA` | Decimal (`float64`) | Média ponderada de notas (*Grade Point Average*) | Escala americana de 0.0 a 4.0 | 1.92 a 4.00 (Média: 3,62; Mediana: 3,71) | 0 |
| 8 | `study_hours_per_day` | Decimal (`float64`) | Quantidade média diária de horas dedicadas aos estudos fora da sala de aula | Horas / dia | 0.5 a 9.4 h/dia (Média: 3,5 h) | 0 |
| 9 | `class_attendance_percent` | Inteiro (`int64`) | Taxa percentual de frequência presencial às aulas | Porcentagem (0% a 100%) | 44% a 100% (Média: 90,5%; Mediana: 93%) | 0 |
| 10 | `sleep_hours` | Decimal (`float64`) | Média diária de horas de sono por noite | Horas / noite | 3.5 a 10.0 h/noite (Média: 7,0 h) | 0 |
| 11 | `screen_time_hours` | Decimal (`float64`) | Tempo total diário de exposição a telas digitais (celular, PC, TV) | Horas / dia | 1.0 a 12.0 h/dia (Média: 5,8 h) | 0 |
| 12 | `social_media_hours` | Decimal (`float64`) | Horas diárias gastas especificamente em redes sociais | Horas / dia | 0.0 a 8.0 h/dia (Média: 2,8 h) | 0 |
| 13 | `gaming_hours` | Decimal (`float64`) | Horas diárias gastas jogando videogames / jogos online | Horas / dia | 0.0 a 6.4 h/dia (Média: 1,6 h) | 0 |
| 14 | `exercise_hours_per_week` | Decimal (`float64`) | Horas semanais dedicadas a atividades físicas ou esportes | Horas / semana | 0.0 a 10.0 h/sem (Média: 3,1 h) | 0 |
| 15 | `part_time_job` | Booleano / Categórico (`str`) | Indica se o estudante trabalha em emprego de meio período | `Yes`, `No` | `Yes` (Trabalha), `No` (Apenas estuda) | 0 |
| 16 | `relationship_status` | Categórico (`str`) | Status de relacionamento afetivo do aluno | `Single`, `In a Relationship` | `Single` (Solteiro), `In a Relationship` (Em relacionamento) | 0 |
| 17 | `family_income_level` | Categórico (`str`) | Faixa de renda econômica familiar | `Low`, `Middle`, `High` | `Low` (Baixa), `Middle` (Média), `High` (Alta) | 0 |
| 18 | `internet_quality` | Categórico (`str`) | Qualidade relatada da conexão de internet utilizada para estudo | `Poor`, `Average`, `Good` | `Poor` (Ruim), `Average` (Média), `Good` (Boa) | 0 |
| 19 | `mental_stress_level` | Decimal (`float64`) | Nível autorreportado de estresse psicológico e mental | Escala de 1.0 a 10.0 | 1.0 a 9.0 (Média: 4,06; Mediana: 4,10) | 0 |
| 20 | `AI_tool_usage_hours` | Decimal (`float64`) | Horas diárias de utilização de assistentes e ferramentas de IA | Horas / dia | 0.0 a 5.0 h/dia (Média: 1,26 h) | 0 |
| 21 | `favorite_AI_tool` | Categórico (`str`) | Ferramenta de inteligência artificial favorita / mais utilizada | 4 ferramentas principais ou ausente | `ChatGPT`, `Claude`, `Copilot`, `Gemini`, `None` / Ausente | 1.720 (17,2% não utilizam IA) |
| 22 | `note_taking_method` | Categórico (`str`) | Método predominante de anotação durante os estudos | 3 categorias | `Digital` (Computador/Tablet), `Handwritten` (Caderno/Caneta), `Mixed` (Misto) | 0 |
| 23 | `exam_preparation_days` | Inteiro (`int64`) | Quantidade de dias de antecedência dedicados à preparação para o exame final | Dias de preparação | 1 a 30 dias (Média: 12,0 dias) | 0 |
| 24 | `coffee_consumption_per_day` | Inteiro (`int64`) | Número de xícaras de café consumidas por dia | Xícaras / dia | 0 a 8 xícaras (Média: 1,8 xícaras) | 0 |
| 25 | `extracurricular_hours_per_week` | Decimal (`float64`) | Horas semanais gastas em atividades extracurriculares (voluntariado, clubes, atlética) | Horas / semana | 0.0 a 11.4 h/sem (Média: 4,0 h) | 0 |
| 26 | `final_exam_score` | Inteiro (`int64`) | Nota obtida no exame final da disciplina/período | Escala de 0 a 100 pontos | 56 a 100 pontos (Média: 96,1; Mediana: 100) | 0 |
| 27 | `assignment_score` | Inteiro (`int64`) | Nota média obtida nos trabalhos, projetos e listas de exercícios | Escala de 0 a 100 pontos | 60 a 100 pontos (Média: 93,0; Mediana: 95) | 0 |

---

## 3. Estatísticas Descritivas das Variáveis Numéricas

| Atributo | Mínimo | 25% (Q1) | Mediana | 75% (Q3) | Máximo | Média | Desvio Padrão |
|---|---|---|---|---|---|---|---|
| **`age`** | 18,0 | 21,0 | 23,0 | 25,0 | 28,0 | 22,95 | 2,87 |
| **`university_year`** | 1,0 | 2,0 | 3,0 | 4,0 | 4,0 | 2,51 | 1,12 |
| **`GPA`** | 1,92 | 3,42 | 3,71 | 3,96 | 4,00 | 3,62 | 0,39 |
| **`study_hours_per_day`** | 0,50 | 2,60 | 3,50 | 4,40 | 9,40 | 3,51 | 1,28 |
| **`class_attendance_percent`** | 44,0 | 85,0 | 93,0 | 98,0 | 100,0 | 90,49 | 9,79 |
| **`sleep_hours`** | 3,50 | 6,20 | 7,00 | 7,80 | 10,00 | 7,01 | 1,17 |
| **`screen_time_hours`** | 1,00 | 4,40 | 5,80 | 7,10 | 12,00 | 5,79 | 1,98 |
| **`social_media_hours`** | 0,00 | 1,70 | 2,80 | 3,80 | 8,00 | 2,80 | 1,48 |
| **`gaming_hours`** | 0,00 | 0,20 | 1,50 | 2,70 | 6,40 | 1,58 | 1,50 |
| **`exercise_hours_per_week`** | 0,00 | 1,70 | 3,00 | 4,40 | 10,00 | 3,07 | 1,90 |
| **`mental_stress_level`** | 1,00 | 2,90 | 4,10 | 5,20 | 9,00 | 4,06 | 1,62 |
| **`AI_tool_usage_hours`** | 0,00 | 0,40 | 1,20 | 2,00 | 5,00 | 1,26 | 1,07 |
| **`exam_preparation_days`** | 1,0 | 6,0 | 12,0 | 18,0 | 30,0 | 12,04 | 7,20 |
| **`coffee_consumption_per_day`** | 0,0 | 1,0 | 2,0 | 3,0 | 8,0 | 1,80 | 1,22 |
| **`extracurricular_hours_per_week`** | 0,00 | 2,40 | 4,00 | 5,60 | 11,40 | 4,02 | 2,27 |
| **`final_exam_score`** | 56,0 | 94,0 | 100,0 | 100,0 | 100,0 | 96,05 | 6,82 |
| **`assignment_score`** | 60,0 | 88,0 | 95,0 | 100,0 | 100,0 | 92,96 | 7,93 |

---

## 4. Observações Técnicas & Regras de Negócio

1. **Valores Ausentes (`NaN`):**
   - Apenas a coluna `favorite_AI_tool` apresenta valores nulos (1.720 registros, representando 17,2% da coorte). Esses casos correspondem a alunos que responderam não ter ferramenta de IA preferida ou que não utilizam IA (`AI_tool_usage_hours == 0` ou sem preferência).
2. **Correlações Críticas com o Desempenho (`GPA`):**
   - **Positivas Fortes:** `class_attendance_percent` ($r = +0.744$) e `study_hours_per_day` ($r = +0.717$).
   - **Positiva Moderada:** `sleep_hours` ($r = +0.232$).
   - **Negativas Relevantes:** `mental_stress_level` ($r = -0.274$), `social_media_hours` ($r = -0.233$) e `screen_time_hours` ($r = -0.177$).
3. **Distribuição das Notas:**
   - As notas de exame (`final_exam_score`) e trabalhos (`assignment_score`) possuem concentração acentuada nos percentis superiores (mediana 100 e 95 respectivamente).
