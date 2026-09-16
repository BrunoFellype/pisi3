"""
Script gerador de PDFs oficiais do Projeto ASTRA
- Relatório de Análise Exploratória de Dados (EDA)
- Guia Completo de Estudos e Apresentação para a Equipe

Cadeira: PISI 3 - UFRPE | Projeto ASTRA
"""

import os
import re
import markdown
from fpdf import FPDF


class DocumentoPDF(FPDF):
    def __init__(self, titulo_cabecalho="ASTRA Analytics", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.titulo_cabecalho = titulo_cabecalho
        # Registra fontes Unicode do Windows
        self.add_font('arial', '', 'C:/Windows/Fonts/arial.ttf')
        self.add_font('arial', 'B', 'C:/Windows/Fonts/arialbd.ttf')
        self.add_font('arial', 'I', 'C:/Windows/Fonts/ariali.ttf')
        if os.path.exists('C:/Windows/Fonts/arialbi.ttf'):
            self.add_font('arial', 'BI', 'C:/Windows/Fonts/arialbi.ttf')

    def header(self):
        if self.page_no() > 1:
            self.set_font('arial', 'I', 8)
            self.set_text_color(100, 116, 139) # Slate muted
            self.cell(0, 8, f'{self.titulo_cabecalho} | UFRPE / PISI 3', border='B', align='L')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('arial', 'I', 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', align='C')


def formatar_markdown_para_html(md_path: str) -> str:
    with open(md_path, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    # Simplifica fórmulas LaTeX inline para texto UTF-8 legível
    conteudo = conteudo.replace(r'\ge', '>=')
    conteudo = conteudo.replace(r'\le', '<=')
    conteudo = conteudo.replace(r'\times', 'x')
    conteudo = conteudo.replace(r'\to', '->')
    conteudo = conteudo.replace(r'\text{h}', 'h')
    conteudo = conteudo.replace(r'\text{h/dia}', 'h/dia')
    conteudo = conteudo.replace(r'\mathbf', '')
    conteudo = conteudo.replace('│', '|').replace('├──', '|--').replace('└──', '\\--').replace('─', '-')
    conteudo = re.sub(r'\$([^\$]+)\$', r'\1', conteudo)

    # Remove tags aninhadas dentro de células <td> para compatibilidade com o parser de tabelas
    def limpar_tags_tabela(match):
        tabela = match.group(0)
        return re.sub(r'</?(strong|b|em|i)>', '', tabela)

    html = markdown.markdown(
        conteudo,
        extensions=['tables', 'fenced_code']
    )
    html = re.sub(r'<table>.*?</table>', limpar_tags_tabela, html, flags=re.DOTALL)
    return html


def converter_md_para_pdf(md_nome: str, pdf_nome: str, titulo_cabecalho: str):
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(diretorio_atual, md_nome)
    pdf_path = os.path.join(diretorio_atual, pdf_nome)

    html_content = formatar_markdown_para_html(md_path)

    pdf = DocumentoPDF(titulo_cabecalho=titulo_cabecalho, orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('arial', size=10)

    pdf.write_html(html_content)

    pdf.output(pdf_path)
    print(f"PDF gerado com sucesso: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")


if __name__ == '__main__':
    # 1. Relatório Técnico Oficial do EDA
    converter_md_para_pdf(
        'relatorio_eda.md',
        'relatorio_eda.pdf',
        'ASTRA Analytics — Relatório Técnico de EDA'
    )
    # 2. Guia de Estudos e Apresentação para a Equipe
    converter_md_para_pdf(
        'guia_estudo_projeto_astra.md',
        'guia_estudo_projeto_astra.pdf',
        'ASTRA Analytics — Guia de Estudos & Apresentação'
    )
