"""
ASTRA Analytics — Script Oficial de Análise Exploratória de Dados (EDA)
Cadeira: PISI 3 (Projeto Interdisciplinar de Sistemas de Informação 3) - UFRPE
Dataset: global_university_students_performance_habits_10000.csv (10.000 estudantes, 27 variáveis)
"""

import os
import sys
import pandas as pd
import numpy as np

# Adiciona o diretório raiz ao path para importação dos módulos locais
diretorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if diretorio_raiz not in sys.path:
    sys.path.insert(0, diretorio_raiz)

import Data


def imprimir_cabecalho(titulo: str):
    print("\n" + "=" * 80)
    print(f" {titulo.upper()}")
    print("=" * 80)


def executar_eda():
    imprimir_cabecalho("1. Visão Geral e Integridade dos Dados")
    df_raw = Data.carregar_dados_brutos()
    df = Data.preparar_dataset_completo(df_raw)
    
    total_linhas, total_colunas = df_raw.shape
    print(f"• Total de Estudantes: {total_linhas:,}")
    print(f"• Total de Variáveis Originais: {total_colunas}")
    print(f"• Variáveis Enriquecidas: {df.shape[1] - total_colunas}")
    print(f"• Memória Utilizada: {df.memory_usage().sum() / (1024**2):.2f} MB")
    
    # Verificação de Valores Nulos
    imprimir_cabecalho("2. Diagnóstico de Qualidade e Valores Ausentes")
    nulos = df_raw.isnull().sum()
    nulos_presentes = nulos[nulos > 0]
    if nulos_presentes.empty:
        print("Nenhum valor nulo encontrado em nenhuma coluna.")
    else:
        for col, qtd in nulos_presentes.items():
            pct = (qtd / total_linhas) * 100
            print(f"Coluna '{col}': {qtd:,} nulos ({pct:.2f}%)")
            
        print("\nAnálise Detalhada dos Nulos em 'favorite_AI_tool':")
        ia_sem_tool = df_raw[df_raw['favorite_AI_tool'].isna()]['AI_tool_usage_hours'].value_counts()
        print("Horas de uso de IA entre quem possui 'favorite_AI_tool' nulo:")
        for horas, cont in ia_sem_tool.items():
            print(f"  - {horas} h/dia: {cont:,} estudantes")
            
        print("\n-> Conclusão: 1.276 não usam IA (0.0h). Porém, 444 utilizam IA (0.1h - 0.2h) sem ter")
        print("   uma ferramenta favorita. Portanto, imputações indiscriminadas geram distorções.")

    # Estatísticas Descritivas Univariadas
    imprimir_cabecalho("3. Estatísticas Descritivas Univariadas (Numéricas)")
    colunas_numericas = [
        'age', 'university_year', 'GPA', 'study_hours_per_day',
        'class_attendance_percent', 'sleep_hours', 'screen_time_hours',
        'social_media_hours', 'gaming_hours', 'exercise_hours_per_week',
        'mental_stress_level', 'AI_tool_usage_hours', 'exam_preparation_days',
        'coffee_consumption_per_day', 'extracurricular_hours_per_week',
        'final_exam_score', 'assignment_score'
    ]
    
    stats_df = df_raw[colunas_numericas].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
    stats_df['Skewness'] = df_raw[colunas_numericas].skew()
    stats_df = stats_df.round(2)
    print(stats_df.to_string())

    # Diagnóstico de Efeito Teto (Ceiling Effect)
    imprimir_cabecalho("4. Diagnóstico de Efeito Teto (Notas Truncadas em 100)")
    alunos_nota_100_exame = (df_raw['final_exam_score'] == 100).sum()
    alunos_nota_100_trab = (df_raw['assignment_score'] == 100).sum()
    print(f"• final_exam_score == 100: {alunos_nota_100_exame:,} estudantes ({(alunos_nota_100_exame/total_linhas)*100:.1f}%)")
    print(f"• assignment_score == 100: {alunos_nota_100_trab:,} estudantes ({(alunos_nota_100_trab/total_linhas)*100:.1f}%)")
    print("-> Alerta Estatístico: Metade dos estudantes tem nota 100 no exame final (mediana=100).")
    print("   O truncamento reduz a sensibilidade linear e atenua correlações com o exame.")
    print("   O 'GPA' (escala contínua até 4.0) é o indicador acadêmico mais fidedigno da base.")

    # As 36 Correlações de Pearson
    imprimir_cabecalho("5. Matriz Completa das 36 Correlações de Pearson")
    habitos = [
        'class_attendance_percent',
        'study_hours_per_day',
        'sleep_hours',
        'exercise_hours_per_week',
        'AI_tool_usage_hours',
        'extracurricular_hours_per_week',
        'exam_preparation_days',
        'gaming_hours',
        'coffee_consumption_per_day',
        'screen_time_hours',
        'social_media_hours',
        'mental_stress_level'
    ]
    alvos = ['GPA', 'final_exam_score', 'assignment_score']
    
    corr_36 = df_raw[habitos + alvos].corr().loc[habitos, alvos].round(3)
    
    def classificar_forca(r):
        if r >= 0.7: return "Positiva Forte"
        if r >= 0.3: return "Positiva Moderada"
        if r >= 0.1: return "Positiva Fraca"
        if r > -0.1: return "Nula / Desprezível"
        if r > -0.3: return "Negativa Fraca"
        return "Negativa Moderada/Forte"
        
    tabela_classificada = corr_36.copy()
    tabela_classificada['Classificação GPA'] = tabela_classificada['GPA'].apply(classificar_forca)
    print(tabela_classificada.to_string())

    # Correlações Hábito a Hábito
    imprimir_cabecalho("6. Correlações Hábito a Hábito Relevantes")
    pares_habitos = [
        ('sleep_hours', 'mental_stress_level', "Sono vs. Estresse"),
        ('exercise_hours_per_week', 'mental_stress_level', "Exercício vs. Estresse"),
        ('screen_time_hours', 'social_media_hours', "Tempo de Tela vs. Redes Sociais"),
        ('coffee_consumption_per_day', 'sleep_hours', "Consumo de Café vs. Sono"),
        ('study_hours_per_day', 'mental_stress_level', "Horas de Estudo vs. Estresse"),
        ('study_hours_per_day', 'class_attendance_percent', "Horas de Estudo vs. Presença")
    ]
    for h1, h2, desc in pares_habitos:
        r_val = df_raw[[h1, h2]].corr().iloc[0, 1]
        print(f"• {desc:40s} | r = {r_val:+.3f} ({classificar_forca(r_val)})")

    # Pergunta de Pesquisa 1: Sono e Estresse modulando Horas de Estudo
    imprimir_cabecalho("7. Pergunta Norteadora 1: Interação Sono, Estresse e Estudo no GPA")
    print("Simulação do GPA médio para diferentes cruzamentos de Hábitos:")
    
    q_estudo_alto = df['study_hours_per_day'] >= 4.5
    q_estudo_medio = (df['study_hours_per_day'] >= 2.5) & (df['study_hours_per_day'] < 4.5)
    q_estudo_baixo = df['study_hours_per_day'] < 2.5
    
    cenarios = [
        ("Alta Dedicação (>= 4.5h) + Sono Bom (>= 7h) + Estresse Baixo (<= 3.5)",
         q_estudo_alto & (df['sleep_hours'] >= 7) & (df['mental_stress_level'] <= 3.5)),
        ("Alta Dedicação (>= 4.5h) + Sono Crítico (< 6h) + Estresse Elevado (> 5)",
         q_estudo_alto & (df['sleep_hours'] < 6) & (df['mental_stress_level'] > 5)),
        ("Dedicação Média (2.5h - 4.5h) + Sono Bom (>= 7h) + Estresse Baixo (<= 3.5)",
         q_estudo_medio & (df['sleep_hours'] >= 7) & (df['mental_stress_level'] <= 3.5)),
        ("Baixa Dedicação (< 2.5h) + Sono Bom (>= 7h) + Estresse Baixo (<= 3.5)",
         q_estudo_baixo & (df['sleep_hours'] >= 7) & (df['mental_stress_level'] <= 3.5)),
    ]
    
    for label, mask in cenarios:
        sub = df[mask]
        print(f"• {label}")
        print(f"  -> Quantidade de Alunos: {len(sub):,} | GPA Médio: {sub['GPA'].mean():.2f} (DP: {sub['GPA'].std():.2f})")

    # Pergunta de Pesquisa 2: IA vs. Redes Sociais vs. Presença
    imprimir_cabecalho("8. Pergunta Norteadora 2: Impacto Real do Uso de IA vs. Redes Sociais e Presença")
    print(f"• Correlação Uso de IA vs. GPA:         r = {df_raw[['AI_tool_usage_hours', 'GPA']].corr().iloc[0,1]:+.3f}")
    print(f"• Correlação Redes Sociais vs. GPA:     r = {df_raw[['social_media_hours', 'GPA']].corr().iloc[0,1]:+.3f}")
    print(f"• Correlação Presença em Aula vs. GPA:  r = {df_raw[['class_attendance_percent', 'GPA']].corr().iloc[0,1]:+.3f}")
    
    media_ia_alta = df[df['AI_tool_usage_hours'] >= 3.0]['GPA'].mean()
    media_ia_zero = df[df['AI_tool_usage_hours'] == 0.0]['GPA'].mean()
    print(f"\n-> Alunos com alto uso de IA (>= 3h/dia): GPA Médio = {media_ia_alta:.2f}")
    print(f"-> Alunos que não usam IA (0h/dia):        GPA Médio = {media_ia_zero:.2f}")
    print("-> Conclusão: A tecnologia de IA por si só não substitui o engajamento presencial (+0.74)")
    print("   nem anula a perda de foco causada por redes sociais (-0.23).")

    # Análise de Categóricas e Rendimento
    imprimir_cabecalho("9. Análise Categórica: Rendimento Acadêmico por Grupos")
    for cat_col in ['note_taking_method', 'family_income_level', 'part_time_job', 'gender']:
        print(f"\nDesempenho por '{cat_col}':")
        resumo_cat = df.groupby(cat_col).agg(
            total=('student_id', 'count'),
            gpa_medio=('GPA', 'mean'),
            final_exam_medio=('final_exam_score', 'mean'),
            estresse_medio=('mental_stress_level', 'mean')
        ).round(2)
        print(resumo_cat.to_string())

    imprimir_cabecalho("EDA Concluído com Sucesso!")
    return df, corr_36


if __name__ == "__main__":
    executar_eda()
