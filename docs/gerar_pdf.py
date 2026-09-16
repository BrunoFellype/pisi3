"""
Script gerador de PDF oficial para o Relatório de Análise Exploratória de Dados (EDA)
Cadeira: PISI 3 - UFRPE | Projeto ASTRA
"""

import os
import re
import markdown
from fpdf import FPDF


class RelatorioPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Registra fontes Unicode do Windows
        self.add_font('Arial', '', 'C:/Windows/Fonts/arial.ttf')
        self.add_font('Arial', 'B', 'C:/Windows/Fonts/arialbd.ttf')
        self.add_font('Arial', 'I', 'C:/Windows/Fonts/ariali.ttf')

    def header(self):
        if self.page_no() > 1:
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 116, 139) # Slate muted
            self.cell(0, 8, 'ASTRA Analytics — Relatório Técnico de EDA | UFRPE / PISI 3', border='B', align='L')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
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
    conteudo = re.sub(r'\$([^\$]+)\$', r'\1', conteudo)

    html = markdown.markdown(
        conteudo,
        extensions=['tables', 'fenced_code']
    )
    return html


def gerar_relatorio_pdf():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(diretorio_atual, 'relatorio_eda.md')
    pdf_path = os.path.join(diretorio_atual, 'relatorio_eda.pdf')

    html_content = formatar_markdown_para_html(md_path)

    pdf = RelatorioPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', size=10)

    pdf.write_html(html_content)

    pdf.output(pdf_path)
    print(f"Relatório em PDF gerado com sucesso em: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")


if __name__ == '__main__':
    gerar_relatorio_pdf()
