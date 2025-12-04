import os
import re
import sys
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import requests
import time
from datetime import datetime

# =========================
# 1. CONFIGURAÇÃO DINÂMICA DO TESSERACT
# =========================
def setup_tesseract():
    """
    Configuração automática do Tesseract para diferentes ambientes
    """
    # Verifica se estamos no Streamlit Cloud (Linux)
    if os.path.exists('/app'):
        caminhos_tentar = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/usr/bin/tesseract-ocr'
        ]
    else:
        # Ambiente local
        caminhos_tentar = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract'
        ]
    
    for caminho in caminhos_tentar:
        try:
            if os.path.exists(caminho):
                pytesseract.pytesseract.tesseract_cmd = caminho
                pytesseract.get_tesseract_version()
                return True
        except:
            continue
    
    return False

tesseract_ready = setup_tesseract()
CUSTOM_CONFIG = r'--oem 3 --psm 6'

# =========================
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO
# =========================
def preprocess_image(img_array):
    """Pré-processamento de imagem para OCR"""
    if img_array is None:
        return None
    
    # Redimensionamento para otimizar processamento
    h, w = img_array.shape[:2]
    if h > 2000:
        escala = 2000 / h
        nova_largura = int(w * escala)
        img_array = cv2.resize(img_array, (nova_largura, 2000))
    
    # Conversão para tons de cinza
    if len(img_array.shape) == 3:
        cinza = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        cinza = img_array
    
    # Aumento de contraste
    cinza = cv2.convertScaleAbs(cinza, alpha=1.5, beta=0)
    
    # Redução de ruído
    cinza = cv2.medianBlur(cinza, 3)
    
    # Binarização adaptativa
    _, binario = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(binario)

# =========================
# 3. FUNÇÕES DE EXTRAÇÃO DE DADOS
# =========================
def extract_vale_number(texto):
    """Extrai número do vale do texto OCR"""
    if not texto:
        return None
    
    # Padrões para números de vale
    padroes = [
        r'Vale\s*[\.:ºNº°\-\s]*\s*(\d{5,})',
        r'V\.?\s*[\.:º°\-\s]*\s*(\d{5,})',
        r'N[º°]\s*[:\.\-\s]*\s*(\d{5,})',
        r'VALE\s*(\d{5,})',
    ]
    
    for padrao in padroes:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            numero = match.group(1)
            # Remove caracteres não numéricos
            numero_limpo = re.sub(r'[^\d]', '', numero)
            if len(numero_limpo) >= 5:
                return numero_limpo
    
    return None

def extract_supplier(texto):
    """Extrai nome do fornecedor do texto OCR"""
    if not texto:
        return None
    
    padroes = [
        r'Fornecedor\s*[\.:;\-]\s*([A-Za-zÀ-ÿ0-9\s\.\-&]+?)(?=\s*(Vale|Data|Nº|$|\.))',
        r'FORNECEDOR\s*[\.:;\-]\s*([A-Za-zÀ-ÿ0-9\s\.\-&]+)',
        r'Empresa\s*[\.:;\-]\s*([A-Za-zÀ-ÿ0-9\s\.\-&]+)',
    ]
    
    for padrao in padroes:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            fornecedor = match.group(1).strip()
            # Limpa caracteres extras
            fornecedor = re.split(r'\s{2,}|[-\|]', fornecedor)[0]
            if len(fornecedor) > 3:
                return fornecedor
    
    return None

def extract_due_date(texto):
    """Extrai data de vencimento (após 'a') do texto OCR"""
    if not texto:
        return None
    
    # Padrão específico para data após "a"
    padrao_principal = r'Data\s*[\.:]\s*\d{1,2}/\d{1,2}/\d{4}\s*[aAà]\s*(\d{1,2}/\d{1,2}/\d{4})'
    match = re.search(padrao_principal, texto)
    if match:
        return match.group(1)
    
    # Fallback para outras datas
    padroes_fallback = [
        r'Vencimento\s*[\.:]\s*(\d{1,2}/\d{1,2}/\d{4})',
        r'Venc\.?\s*[\.:]\s*(\d{1,2}/\d{1,2}/\d{4})',
    ]
    
    for padrao in padroes_fallback:
        match = re.search(padrao, texto)
        if match:
            return match.group(1)
    
    return None

# =========================
# 4. FUNÇÕES AUXILIARES
# =========================
def get_drive_file_id(link):
    """Extrai ID do arquivo do Google Drive"""
    padroes = [
        r'/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
    ]
    
    for padrao in padroes:
        match = re.search(padrao, link)
        if match:
            return match.group(1)
    return None

def download_from_drive(file_id):
    """Baixa imagem do Google Drive"""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, stream=True, timeout=30)
        
        # Para arquivos grandes, pode ser necessário confirmar
        if "confirm=" in response.url:
            confirm_token = re.search(r'confirm=([^&]+)', response.url)
            if confirm_token:
                url = f"https://drive.google.com/uc?export=download&confirm={confirm_token.group(1)}&id={file_id}"
                response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
    except Exception as e:
        st.error(f"Erro ao baixar do Drive: {str(e)}")
    
    return None

# =========================
# 5. INTERFACE PRINCIPAL
# =========================
def main():
    st.set_page_config(
        page_title="OCR para Vales - Sistema de Extração",
        layout="wide"
    )
    
    # Verificação do Tesseract
    if not tesseract_ready:
        st.error("""
        Tesseract OCR não está configurado corretamente.
        
        Para execução local:
        1. Instale o Tesseract OCR: https://github.com/tesseract-ocr/tesseract
        2. Para Windows: https://github.com/UB-Mannheim/tesseract/wiki
        
        Para deploy no Streamlit Cloud:
        Certifique-se de que o arquivo packages.txt contém:
        tesseract-ocr
        tesseract-ocr-por
        """)
        return
    
    st.title("Sistema de Extração OCR para Vales")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Configurações")
        
        mostrar_processamento = st.checkbox("Mostrar imagens processadas", value=False)
        mostrar_texto = st.checkbox("Mostrar texto extraído", value=True)
        minimo_digitos = st.slider("Mínimo de dígitos do vale", 5, 15, 8)
        
        st.header("Status")
        st.write(f"Tesseract: {'Configurado' if tesseract_ready else 'Não configurado'}")
    
    # Área de upload
    st.header("Carregamento de Imagens")
    
    arquivos_carregados = st.file_uploader(
        "Selecione imagens de vales para processar",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True
    )
    
    st.header("Google Drive")
    link_drive = st.text_input(
        "Link do Google Drive (opcional):",
        placeholder="Cole o link compartilhável aqui"
    )
    
    if st.button("Processar Imagens", type="primary"):
        processar_imagens(arquivos_carregados, link_drive, mostrar_processamento, mostrar_texto, minimo_digitos)

def processar_imagens(arquivos_carregados, link_drive, mostrar_processamento, mostrar_texto, minimo_digitos):
    """Processa as imagens e extrai informações"""
    
    imagens_processar = []
    
    # Processar arquivos carregados
    if arquivos_carregados:
        for arquivo in arquivos_carregados:
            try:
                # Ler o arquivo UMA VEZ e armazenar os bytes
                arquivo_bytes = arquivo.read()
                # Converter bytes para numpy array
                bytes_array = np.asarray(bytearray(arquivo_bytes), dtype=np.uint8)
                imagem = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
                
                if imagem is not None:
                    imagens_processar.append({
                        'nome': arquivo.name,
                        'imagem': imagem,
                        'tipo': 'upload'
                    })
                # Restaurar a posição do arquivo para possível uso futuro
                arquivo.seek(0)
            except Exception as e:
                st.error(f"Erro ao processar {arquivo.name}: {str(e)}")
    
    # Processar link do Drive
    if link_drive:
        file_id = get_drive_file_id(link_drive)
        if file_id:
            imagem = download_from_drive(file_id)
            if imagem is not None:
                imagens_processar.append({
                    'nome': 'imagem_drive.jpg',
                    'imagem': imagem,
                    'tipo': 'drive'
                })
    
    if not imagens_processar:
        st.warning("Nenhuma imagem válida para processar.")
        return
    
    resultados = []
    barra_progresso = st.progress(0)
    texto_status = st.empty()
    
    for idx, dados_imagem in enumerate(imagens_processar):
        nome = dados_imagem['nome']
        imagem = dados_imagem['imagem']
        
        texto_status.text(f"Processando: {nome} ({idx + 1}/{len(imagens_processar)})")
        
        with st.expander(f"Imagem: {nome}", expanded=False):
            coluna_esquerda, coluna_direita = st.columns(2)
            
            with coluna_esquerda:
                # CORREÇÃO PRINCIPAL: Exibição da imagem
                try:
                    if imagem is None:
                        st.warning("Imagem não disponível para exibição")
                    else:
                        # Converter BGR para RGB para exibição correta
                        if len(imagem.shape) == 3:
                            imagem_exibicao = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                        else:
                            # Se for escala de cinza, converter para RGB
                            imagem_exibicao = cv2.cvtColor(imagem, cv2.COLOR_GRAY2RGB)
                        
                        # Usar PIL Image para maior compatibilidade com Streamlit
                        imagem_pil = Image.fromarray(imagem_exibicao)
                        
                        st.image(
                            imagem_pil, 
                            caption=f"Imagem original: {nome}",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"Erro ao exibir imagem {nome}: {str(e)}")
                    # Tentar exibir diretamente como fallback
                    try:
                        st.image(imagem, caption=f"Imagem: {nome} (fallback)")
                    except:
                        st.error(f"Não foi possível exibir a imagem {nome}")
            
            with coluna_direita:
                imagem_processada = preprocess_image(imagem)
                
                if mostrar_processamento and imagem_processada:
                    # Converter imagem processada para exibição
                    if isinstance(imagem_processada, Image.Image):
                        imagem_processada_np = np.array(imagem_processada)
                        if len(imagem_processada_np.shape) == 2:  # Escala de cinza
                            imagem_processada_np = cv2.cvtColor(imagem_processada_np, cv2.COLOR_GRAY2RGB)
                        st.image(imagem_processada_np, caption="Imagem processada", use_container_width=True)
                
                try:
                    texto_ocr = pytesseract.image_to_string(
                        imagem_processada if imagem_processada else imagem,
                        lang='por',
                        config=CUSTOM_CONFIG
                    )
                    
                    numero_vale = extract_vale_number(texto_ocr)
                    fornecedor = extract_supplier(texto_ocr)
                    vencimento = extract_due_date(texto_ocr)
                    
                    # Aplica filtro de dígitos mínimos
                    if numero_vale and len(numero_vale) < minimo_digitos:
                        numero_vale = None
                    
                    # Exibir resultados
                    st.markdown("### Resultados da Extração")
                    
                    coluna1, coluna2, coluna3 = st.columns(3)
                    
                    with coluna1:
                        st.metric("Número do Vale", numero_vale or "Não encontrado")
                    with coluna2:
                        st.metric("Fornecedor", fornecedor or "Não encontrado")
                    with coluna3:
                        st.metric("Data de Vencimento", vencimento or "Não encontrado")
                    
                    if mostrar_texto and texto_ocr.strip():
                        with st.expander("Texto completo extraído"):
                            chave_unica = f"texto_ocr_{nome}_{idx}_{int(time.time())}"
                            st.text_area(
                                "Conteúdo do OCR:",
                                texto_ocr,
                                height=150,
                                key=chave_unica,
                                label_visibility="collapsed"
                            )
                    
                    resultados.append({
                        'Arquivo': nome,
                        'Origem': dados_imagem['tipo'],
                        'Número do Vale': numero_vale or "Não encontrado",
                        'Fornecedor': fornecedor or "Não encontrado",
                        'Vencimento': vencimento or "Não encontrado"
                    })
                    
                except Exception as e:
                    st.error(f"Erro durante o OCR: {str(e)}")
                    resultados.append({
                        'Arquivo': nome,
                        'Origem': dados_imagem['tipo'],
                        'Número do Vale': 'ERRO',
                        'Fornecedor': f'Erro: {str(e)[:50]}',
                        'Vencimento': ''
                    })
        
        barra_progresso.progress((idx + 1) / len(imagens_processar))
    
    texto_status.empty()
    barra_progresso.empty()
    
    # Apresentar resultados consolidados
    if resultados:
        st.header("Resultados Consolidados")
        
        df = pd.DataFrame(resultados)
        
        # Estatísticas
        total_processado = len(resultados)
        vales_encontrados = sum(1 for r in resultados if r['Número do Vale'] not in ['Não encontrado', 'ERRO'])
        
        coluna_stats1, coluna_stats2, coluna_stats3 = st.columns(3)
        
        with coluna_stats1:
            st.metric("Total Processado", total_processado)
        with coluna_stats2:
            st.metric("Vales Encontrados", vales_encontrados)
        with coluna_stats3:
            if total_processado > 0:
                taxa_sucesso = (vales_encontrados / total_processado) * 100
                st.metric("Taxa de Sucesso", f"{taxa_sucesso:.1f}%")
        
        # Tabela de resultados
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Exportação
        st.header("Exportar Resultados")
        
        coluna_exp1, coluna_exp2 = st.columns(2)
        
        with coluna_exp1:
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine='openpyxl') as escritor:
                df.to_excel(escritor, index=False, sheet_name='Resultados')
            
            st.download_button(
                label="Download Excel",
                data=buffer_excel.getvalue(),
                file_name="resultados_vales.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with coluna_exp2:
            buffer_csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="Download CSV",
                data=buffer_csv,
                file_name="resultados_vales.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
