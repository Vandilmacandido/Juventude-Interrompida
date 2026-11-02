"""
Dashboard Interativo Civic Tech — Juventude Interrompida (Pernambuco)
======================================================================
Análise multidimensional da mortalidade por causas externas em Pernambuco
com base nos microdados do SIM/DATASUS (sim_violencia_jovem_pe.csv) e
malha municipal vetorial oficial do IBGE (pe_municipios.geojson).

Autor: Desenvolvedor Python Sênior (Data Science & Civic Tech)
Execução: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. CONSTANTES, PALETA DE CORES CÍVICA E DICIONÁRIO GEOGRÁFICO (IBGE -> PE)
# ==============================================================================
DATA_PATH = Path(__file__).parent / "sim_violencia_jovem_pe.csv"
GEOJSON_PATH = Path(__file__).parent / "pe_municipios.geojson"
CONTORNO_UF_PATH = Path(__file__).parent / "pe_contorno_uf.json"

RECIFE_CENTER = {"lat": -8.0476, "lon": -34.8770}
PE_STATE_CENTER = {"lat": -8.3500, "lon": -37.2000}

CORES_VIOLENCIA: Dict[str, str] = {
    "Agressão": "#C81E1E",                # Vermelho Alerta Intenso
    "Intervenção Policial": "#EA580C",    # Vermelho-Alaranjado / Alerta Crítico
    "Acidente de Trânsito": "#2563EB",    # Azul Cívico
    "Outras Causas Externas": "#64748B",  # Cinza Ardósia
    "Não Informado": "#94A3B8",
}

CORES_SEXO: Dict[str, str] = {
    "Masculino": "#1E3A8A",
    "Feminino": "#0D9488",
    "Ignorado": "#94A3B8",
    "Não Informado": "#CBD5E1",
}

CORES_CLASSES_MAPA: List[str] = [
    "#FEE2E2",  # Faixa 1: Baixa
    "#FCA5A5",  # Faixa 2: Moderada
    "#F87171",  # Faixa 3: Média
    "#DC2626",  # Faixa 4: Alta
    "#991B1B",  # Faixa 5: Crítica
]

# Dicionário interno mapeando códigos IBGE (7 dígitos) de Pernambuco para
# (Nome do Município, Latitude, Longitude, Mesorregião)
MUNICIPIOS_PE_COORDS: Dict[int, Tuple[str, float, float, str]] = {
    2600054: ("Abreu e Lima", -7.9007, -34.8983, "Metropolitana do Recife"),
    2601201: ("Arcoverde", -8.4189, -37.0539, "Sertão Pernambucano"),
    2601904: ("Bezerros", -8.2333, -35.7500, "Agreste Pernambucano"),
    2602902: ("Cabo de Santo Agostinho", -8.2822, -35.0350, "Metropolitana do Recife"),
    2603454: ("Camaragibe", -8.0236, -34.9781, "Metropolitana do Recife"),
    2604007: ("Carpina", -7.8508, -35.2547, "Mata Pernambucana"),
    2604106: ("Caruaru", -8.2845, -35.9699, "Agreste Pernambucano"),
    2605152: ("Dormentes", -8.4469, -40.7711, "São Francisco Pernambucano"),
    2606002: ("Garanhuns", -8.8903, -36.4928, "Agreste Pernambucano"),
    2606200: ("Goiana", -7.5606, -35.0025, "Metropolitana do Recife"),
    2606408: ("Gravatá", -8.2011, -35.5647, "Agreste Pernambucano"),
    2606804: ("Igarassu", -7.8342, -34.9064, "Metropolitana do Recife"),
    2607208: ("Ipojuca", -8.3989, -35.0639, "Metropolitana do Recife"),
    2607901: ("Jaboatão dos Guararapes", -8.1128, -35.0147, "Metropolitana do Recife"),
    2608909: ("Limoeiro", -7.8747, -35.4503, "Agreste Pernambucano"),
    2609600: ("Olinda", -8.0089, -34.8553, "Metropolitana do Recife"),
    2609907: ("Ouricuri", -7.8825, -40.0817, "Sertão Pernambucano"),
    2610004: ("Palmares", -8.6842, -35.5892, "Mata Pernambucana"),
    2610608: ("Paudalho", -7.8967, -35.1742, "Mata Pernambucana"),
    2610707: ("Paulista", -7.9408, -34.8731, "Metropolitana do Recife"),
    2611101: ("Petrolina", -9.3986, -40.5008, "São Francisco Pernambucano"),
    2611606: ("Recife", -8.0476, -34.8770, "Metropolitana do Recife"),
    2612505: ("Santa Cruz do Capibaribe", -7.9575, -36.2047, "Agreste Pernambucano"),
    2613305: ("São Joaquim do Monte", -8.4325, -35.8044, "Agreste Pernambucano"),
    2613701: ("São Lourenço da Mata", -8.0022, -35.0183, "Metropolitana do Recife"),
    2613909: ("Serra Talhada", -7.9919, -38.2983, "Sertão Pernambucano"),
    2614204: ("Sirinhaém", -8.5908, -35.1161, "Mata Pernambucana"),
    2615300: ("Timbaúba", -7.5053, -35.3183, "Mata Pernambucana"),
    2616407: ("Vitória de Santo Antão", -8.1181, -35.2914, "Mata Pernambucana"),
}


# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E ESTILIZAÇÃO TEMA CLARO (CSS)
# ==============================================================================
def configurar_pagina() -> None:
    """Configura parâmetros globais do Streamlit e aplica CSS em tema claro."""
    st.set_page_config(
        page_title="Juventude Interrompida | Observatório PE",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .stApp {
                background-color: #FFFFFF;
                color: #1E293B;
            }
            section[data-testid="stSidebar"] {
                background-color: #F8FAFC;
                border-right: 1px solid #E2E8F0;
            }
            div[data-testid="stMetric"] {
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: 16px 20px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            div[data-testid="stMetricLabel"] > div {
                color: #475569 !important;
                font-weight: 600 !important;
                font-size: 0.92rem !important;
            }
            div[data-testid="stMetricValue"] > div {
                color: #0F172A !important;
                font-weight: 700 !important;
            }
            .civic-header {
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                padding: 1.1rem 1.5rem;
                border-radius: 10px;
                margin-bottom: 1.25rem;
            }
            .civic-header h1 {
                margin: 0;
                font-size: 1.65rem;
                color: #0F172A;
                font-weight: 800;
            }
            .civic-header p {
                margin: 0.35rem 0 0 0;
                color: #475569;
                font-size: 0.96rem;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] {
                gap: 8px !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label {
                width: 100% !important;
                background-color: #FFFFFF !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 8px !important;
                padding: 10px 14px !important;
                margin: 0 !important;
                cursor: pointer !important;
                transition: all 0.15s ease !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
                background-color: #EFF6FF !important;
                border-color: #2563EB !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p {
                color: #1D4ED8 !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
                display: none !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
                background-color: #2563EB !important;
                border-color: #2563EB !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
                color: #FFFFFF !important;
                font-weight: 700 !important;
            }
            /* Hover e seleção em Azul nos filtros (Selectbox / Dropdown) */
            div[data-baseweb="select"] > div {
                border-color: #CBD5E1 !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
            }
            div[data-baseweb="select"] > div:hover {
                border-color: #2563EB !important;
            }
            div[data-baseweb="select"] > div:focus-within {
                border-color: #2563EB !important;
                box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
            }
            ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover,
            li[role="option"]:hover {
                background-color: #DBEAFE !important;
                color: #1E3A8A !important;
            }
            ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"][aria-selected="true"],
            li[role="option"][aria-selected="true"] {
                background-color: #2563EB !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
            }
            /* Responsividade Mobile */
            @media (max-width: 768px) {
                .block-container {
                    padding-left: 0.8rem !important;
                    padding-right: 0.8rem !important;
                    padding-top: 1rem !important;
                }
                .civic-header {
                    padding: 0.85rem 1rem !important;
                }
                .civic-header h1 {
                    font-size: 1.25rem !important;
                }
                .civic-header p {
                    font-size: 0.85rem !important;
                }
                div[data-testid="stMetric"] {
                    padding: 12px 14px !important;
                    margin-bottom: 8px !important;
                }
                div[data-testid="stMetricValue"] > div {
                    font-size: 1.35rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. CARREGAMENTO DE DADOS E MALHA VETORIAL DO IBGE (PERNAMBUCO)
# ==============================================================================
@st.cache_data(show_spinner=False)
def carregar_malha_pe(caminho_geojson: Path = GEOJSON_PATH) -> Dict[str, Any]:
    """Carrega o arquivo GeoJSON com os polígonos dos 184 municípios continentais de PE."""
    with open(caminho_geojson, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def carregar_contorno_uf(caminho_contorno: Path = CONTORNO_UF_PATH) -> List[Any]:
    """Carrega o contorno externo oficial do estado de Pernambuco (IBGE)."""
    if caminho_contorno.exists():
        with open(caminho_contorno, encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data(show_spinner="Carregando e processando microdados do DATASUS/SIM...")
def carregar_dados(caminho_csv: Path) -> pd.DataFrame:
    """
    Lê o arquivo CSV local, realiza o tratamento defensivo de valores nulos/inválidos
    e mapeia os códigos IBGE (`id_municipio`) para nomes e coordenadas geográficas.
    """
    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado em: {caminho_csv.resolve()}"
        )

    df = pd.read_csv(caminho_csv, encoding="utf-8")

    colunas_esperadas = [
        "ano",
        "id_municipio",
        "faixa_etaria",
        "sexo_desc",
        "raca_desc",
        "tipo_violencia",
    ]
    colunas_faltantes = [c for c in colunas_esperadas if c not in df.columns]
    if colunas_faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes no CSV: {', '.join(colunas_faltantes)}"
        )

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["id_municipio"] = pd.to_numeric(df["id_municipio"], errors="coerce")
    df = df.dropna(subset=["ano", "id_municipio"]).copy()
    df["ano"] = df["ano"].astype(int)
    df["id_municipio"] = df["id_municipio"].astype(int)

    for col in ["faixa_etaria", "sexo_desc", "raca_desc", "tipo_violencia"]:
        df[col] = (
            df[col]
            .fillna("Ignorado")
            .astype(str)
            .str.strip()
            .replace({"": "Ignorado", "nan": "Ignorado", "None": "Ignorado"})
        )

    df["municipio_nome"] = df["id_municipio"].map(
        lambda cod: MUNICIPIOS_PE_COORDS.get(cod, (f"Município ({cod})", None, None, "Outra"))[0]
    )
    df["lat"] = df["id_municipio"].map(
        lambda cod: MUNICIPIOS_PE_COORDS.get(cod, (None, PE_STATE_CENTER["lat"], None, None))[1]
    )
    df["lon"] = df["id_municipio"].map(
        lambda cod: MUNICIPIOS_PE_COORDS.get(cod, (None, None, PE_STATE_CENTER["lon"], None))[2]
    )
    df["mesorregiao"] = df["id_municipio"].map(
        lambda cod: MUNICIPIOS_PE_COORDS.get(cod, (None, None, None, "Pernambuco"))[3]
    )

    return df


# ==============================================================================
# 4. MAPA COROPLÉTICO DO CONTORNO DE PERNAMBUCO (EM APENAS UM MAPA)
# ==============================================================================
def _extrair_aneis_poligono(geometria: Dict[str, Any]) -> List[List[List[float]]]:
    """Extrai todos os anéis de coordenadas (lon, lat) de um Polygon ou MultiPolygon."""
    tipo = geometria.get("type")
    coords = geometria.get("coordinates", [])
    if tipo == "Polygon":
        return coords
    if tipo == "MultiPolygon":
        aneis: List[List[List[float]]] = []
        for poligono in coords:
            aneis.extend(poligono)
        return aneis
    return []


def criar_mapa_coropletico_pe(
    df_filtrado: pd.DataFrame,
    cidade_selecionada: str = "Todas",
) -> go.Figure:
    """
    Desenha o mapa coroplético temático do contorno de Pernambuco dividido em seus
    184 municípios continentais (malha IBGE), pintando cada polígono municipal
    em classes discretas na escala vermelha de alerta — exatamente no formato das
    imagens de referência (com grade geográfica, legenda por faixas e contorno de PE).
    """
    geojson_pe = carregar_malha_pe(GEOJSON_PATH)
    contorno_uf = carregar_contorno_uf(CONTORNO_UF_PATH)
    cores_classes = CORES_CLASSES_MAPA

    # Agrega métricas para os municípios presentes no filtro
    agg_mun = (
        df_filtrado.groupby("id_municipio", as_index=False)
        .agg(
            total_obitos=("ano", "count"),
            obitos_jovens=(
                "faixa_etaria",
                lambda s: int((s == "Jovem (15 a 29 anos)").sum()),
            ),
            violencia_principal=(
                "tipo_violencia",
                lambda s: s.mode().iloc[0] if not s.mode().empty else "N/A",
            ),
        )
    )
    mapa_stats: Dict[int, Dict[str, Any]] = {}
    for _, row in agg_mun.iterrows():
        tot = int(row["total_obitos"])
        jov = int(row["obitos_jovens"])
        mapa_stats[int(row["id_municipio"])] = {
            "total": tot,
            "jovens": jov,
            "perc_jovens": round((jov / tot * 100), 1) if tot > 0 else 0.0,
            "causa": row["violencia_principal"],
        }

    # Define faixas discretas (classes coropléticas) baseadas na distribuição de valores
    valores_positivos = sorted([v["total"] for v in mapa_stats.values() if v["total"] > 0])
    if not valores_positivos:
        limites = [10, 30, 60, 120]
    else:
        n_vals = len(valores_positivos)
        q1 = valores_positivos[min(n_vals - 1, max(0, int(n_vals * 0.25)))]
        q2 = valores_positivos[min(n_vals - 1, max(0, int(n_vals * 0.50)))]
        q3 = valores_positivos[min(n_vals - 1, max(0, int(n_vals * 0.75)))]
        q4 = valores_positivos[min(n_vals - 1, max(0, int(n_vals * 0.90)))]
        limites = sorted(list({max(1, q1), max(2, q2), max(5, q3), max(10, q4)}))
        while len(limites) < 4:
            limites.append(limites[-1] * 2)

    b1, b2, b3, b4 = limites[0], limites[1], limites[2], limites[3]
    labels_classes = [
        f"[0, {b1}] (Baixa)",
        f"({b1}, {b2}] (Moderada)",
        f"({b2}, {b3}] (Média)",
        f"({b3}, {b4}] (Alta)",
        f"> {b4} (Crítica)",
    ]

    def classificar_valor(val: int) -> int:
        if val <= b1:
            return 0
        if val <= b2:
            return 1
        if val <= b3:
            return 2
        if val <= b4:
            return 3
        return 4

    # Agrupa coordenadas dos polígonos por classe de cor para renderização vetorial rápida
    xs_por_classe: Dict[int, List[float | None]] = {i: [] for i in range(5)}
    ys_por_classe: Dict[int, List[float | None]] = {i: [] for i in range(5)}

    hover_lons: List[float] = []
    hover_lats: List[float] = []
    hover_textos: List[str] = []

    for feat in geojson_pe["features"]:
        props = feat["properties"]
        cod = int(props["id_municipio"])
        nome = props["nome"]
        id_polo = int(props.get("id_polo", cod))
        nome_polo = props.get("nome_polo", nome)
        lat_c = float(props.get("lat", -8.28))
        lon_c = float(props.get("lon", -36.5))
        mesorregiao = MUNICIPIOS_PE_COORDS.get(id_polo, (None, None, None, "Pernambuco"))[3]

        if cidade_selecionada != "Todas":
            if nome == cidade_selecionada:
                info = mapa_stats.get(cod, {"total": 0, "jovens": 0, "perc_jovens": 0.0, "causa": "-"})
                idx_classe = 4
                sub_status = "Município Selecionado no Filtro"
            else:
                info = {"total": 0, "jovens": 0, "perc_jovens": 0.0, "causa": "-"}
                idx_classe = 0
                sub_status = "Demais Municípios de Pernambuco"
        elif cod in mapa_stats:
            info = mapa_stats[cod]
            idx_classe = classificar_valor(info["total"])
            sub_status = "Dados Municipais Diretos"
        elif id_polo in mapa_stats:
            info = mapa_stats[id_polo]
            # Municípios da microrregião recebem a classe do seu polo regional
            idx_classe = max(0, classificar_valor(info["total"]) - 1)
            sub_status = f"Região de Influência de {nome_polo}"
        else:
            info = {"total": 0, "jovens": 0, "perc_jovens": 0.0, "causa": "Sem registro"}
            idx_classe = 0
            sub_status = "Sem registros no recorte"

        for anel in _extrair_aneis_poligono(feat["geometry"]):
            xs_por_classe[idx_classe].extend([pt[0] for pt in anel] + [None])
            ys_por_classe[idx_classe].extend([pt[1] for pt in anel] + [None])

        hover_lons.append(lon_c)
        hover_lats.append(lat_c)
        hover_textos.append(
            f"<b>{nome}</b><br>"
            f"<i>{sub_status}</i><br><br>"
            f"• <b>Óbitos (Município/Polo):</b> {info['total']:,}<br>"
            f"• <b>Jovens (15–29 anos):</b> {info['jovens']:,} ({info['perc_jovens']:.1f}%)<br>"
            f"• <b>Causa Predominante:</b> {info['causa']}<br>"
            f"• <b>Polo Regional:</b> {nome_polo}<br>"
            f"• <b>Mesorregião:</b> {mesorregiao}".replace(",", ".")
        )

    fig = go.Figure()

    # 1. Camadas de polígonos preenchidos por faixa coroplética
    for idx_classe in range(5):
        xs = xs_por_classe[idx_classe]
        ys = ys_por_classe[idx_classe]
        if not xs:
            # Adiciona traço vazio apenas para manter todas as faixas visíveis na legenda
            xs, ys = [None], [None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=cores_classes[idx_classe],
                line=dict(color="#64748B", width=0.55),
                name=labels_classes[idx_classe],
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # 2. Contorno externo oficial de Pernambuco (destaque na silhueta estadual)
    if contorno_uf:
        uf_xs: List[float | None] = []
        uf_ys: List[float | None] = []
        for poligono in contorno_uf:
            for anel in poligono:
                uf_xs.extend([pt[0] for pt in anel] + [None])
                uf_ys.extend([pt[1] for pt in anel] + [None])
        fig.add_trace(
            go.Scatter(
                x=uf_xs,
                y=uf_ys,
                mode="lines",
                line=dict(color="#0F172A", width=1.8),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 3. Rótulos dos principais polos regionais sobre o mapa
    polos_rotulos = [
        ("PETROLINA", -9.22, -40.52),
        ("OURICURI", -7.90, -40.08),
        ("SERRA TALHADA", -7.99, -38.30),
        ("ARCOVERDE", -8.42, -37.06),
        ("GARANHUNS", -8.89, -36.49),
        ("CARUARU", -8.28, -35.97),
        ("VITÓRIA", -8.12, -35.29),
        ("GOIANA", -7.56, -35.00),
        ("PALMARES", -8.68, -35.59),
        ("RECIFE", -8.05, -34.82),
    ]
    fig.add_trace(
        go.Scatter(
            x=[p[2] for p in polos_rotulos],
            y=[p[1] for p in polos_rotulos],
            text=[p[0] for p in polos_rotulos],
            mode="text",
            textfont=dict(size=9.5, color="#0F172A", family="Arial Black, sans-serif"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 4. Camada interativa de hover em todos os 184 municípios
    fig.add_trace(
        go.Scatter(
            x=hover_lons,
            y=hover_lats,
            mode="markers",
            marker=dict(size=14, opacity=0.01, color="#0F172A"),
            text=hover_textos,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    # Enquadramento geográfico do estado completo de Pernambuco
    x_range = [-41.45, -34.60]
    y_range = [-9.60, -7.15]

    fig.update_layout(
        title=dict(
            text="<b>Mapa Coroplético de Mortalidade por Causas Externas em Pernambuco</b>",
            font=dict(size=16, color="#0F172A"),
            x=0.01,
        ),
        xaxis=dict(
            range=x_range,
            tickvals=[-41, -40, -39, -38, -37, -36, -35],
            ticktext=["41°W", "40°W", "39°W", "38°W", "37°W", "36°W", "35°W"],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            title="",
        ),
        yaxis=dict(
            range=y_range,
            tickvals=[-9, -8, -7],
            ticktext=["9°S", "8°S", "7°S"],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1.01,
            title="",
        ),
        legend=dict(
            title=dict(text="<b>Volume de Óbitos: </b>", font=dict(size=11.5)),
            orientation="h",
            x=0.5,
            y=-0.10,
            xanchor="center",
            yanchor="top",
            bgcolor="#F8FAFC",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11),
        ),
        annotations=[
            dict(
                text="<b>PERNAMBUCO</b>",
                x=0.96,
                y=0.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=20, color="#334155", family="sans-serif"),
            ),
        ],
        margin={"r": 15, "t": 50, "l": 40, "b": 80},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=540,
        autosize=True,
        hovermode="closest",
    )

    return fig


# ==============================================================================
# 5. BARRA LATERAL (NAVEGAÇÃO MULTIPÁGINAS E FILTROS GLOBAIS)
# ==============================================================================
def renderizar_sidebar(
    df: pd.DataFrame,
) -> Tuple[str, str, str, str, pd.DataFrame, pd.DataFrame]:
    """
    Renderiza o menu lateral com navegação entre as 3 páginas principais
    e os filtros globais de Ano, Faixa Etária e Cidade.
    """
    with st.sidebar:
        st.markdown("## Juventude Interrompida")
        st.caption(
            "Observatório Cívico de Mortalidade por Causas Externas em Pernambuco (SIM/DATASUS)"
        )
        st.markdown("---")

        pagina_selecionada = st.radio(
            "Navegação do Painel",
            options=[
                "Visão Geral e Mapa",
                "Análises Demográficas e Tendências",
                "Fluxo da Vulnerabilidade",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### Filtros Globais")

        anos_disponiveis = sorted(df["ano"].unique().tolist(), reverse=True)
        opcoes_ano = ["Todos"] + [str(ano) for ano in anos_disponiveis]
        ano_selecionado = st.selectbox(
            "Ano de Referência:",
            options=opcoes_ano,
            index=0,
            help="Filtra os indicadores e gráficos por ano.",
        )

        ordem_faixa = ["Jovem (15 a 29 anos)", "30+ anos", "0 a 14 anos"]
        faixas_disp = [f for f in ordem_faixa if f in df["faixa_etaria"].unique()] + [
            f for f in df["faixa_etaria"].unique() if f not in ordem_faixa
        ]
        opcoes_faixa = ["Todas"] + faixas_disp
        faixa_selecionada = st.selectbox(
            "Faixa Etária:",
            options=opcoes_faixa,
            index=0,
            help="Filtra os dados por faixa etária da vítima.",
        )

        cidades_disp = sorted(df["municipio_nome"].unique().tolist())
        opcoes_cidade = ["Todas"] + cidades_disp
        cidade_selecionada = st.selectbox(
            "Cidade (Município):",
            options=opcoes_cidade,
            index=0,
            help="Filtra os dados para um município específico de Pernambuco.",
        )

        df_filtrado = df.copy()
        if ano_selecionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado["ano"] == int(ano_selecionado)]
            label_periodo = f"Ano de {ano_selecionado}"
        else:
            label_periodo = f"{min(anos_disponiveis)}–{max(anos_disponiveis)} (Todos)"

        if cidade_selecionada != "Todas":
            df_filtrado = df_filtrado[df_filtrado["municipio_nome"] == cidade_selecionada]

        df_contexto_etario = df_filtrado.copy()

        if faixa_selecionada != "Todas":
            df_filtrado = df_filtrado[df_filtrado["faixa_etaria"] == faixa_selecionada]

        st.info(
            f"**Período:** {label_periodo}\n\n"
            f"**Faixa Etária:** {faixa_selecionada}\n\n"
            f"**Cidade:** {cidade_selecionada}\n\n"
            f"**Registros filtrados:** `{len(df_filtrado):,}` óbitos\n\n"
            f"**Municípios no recorte:** `{df_filtrado['id_municipio'].nunique()}`".replace(",", ".")
        )

        st.markdown("---")
        st.caption(
            "**Fonte dos Dados:** Sistema de Informações sobre Mortalidade (SIM/DATASUS). "
            "Malha geográfica vetorial oficial do IBGE (184 municípios continentais de PE)."
        )

    return (
        pagina_selecionada,
        ano_selecionado,
        faixa_selecionada,
        cidade_selecionada,
        df_filtrado,
        df_contexto_etario,
    )


# ==============================================================================
# 6. PÁGINA 1: VISÃO GERAL E MAPA COROPLÉTICO DE PERNAMBUCO (APENAS UM MAPA)
# ==============================================================================
def renderizar_pagina_mapa(
    df_filtrado: pd.DataFrame,
    df_contexto_etario: pd.DataFrame,
    ano_selecionado: str,
    faixa_selecionada: str,
    cidade_selecionada: str,
) -> None:
    """
    Renderiza a Página 1 contendo os KPIs Sociais no topo e APENAS UM Mapa
    Coroplético do contorno de Pernambuco em largura total.
    """
    st.markdown(
        """
        <div class="civic-header">
            <h1>Visão Geral e Distribuição Territorial (Pernambuco)</h1>
            <p>Contorno territorial de Pernambuco dividido por municípios e colorido conforme o volume de óbitos (IBGE / SIM-DATASUS).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_filtrado.empty:
        st.warning("Não há registros para a combinação de filtros selecionada.")
        return

    total_obitos = len(df_filtrado)

    if faixa_selecionada == "Todas":
        obitos_jovens = int((df_filtrado["faixa_etaria"] == "Jovem (15 a 29 anos)").sum())
        perc_jovens = (obitos_jovens / total_obitos * 100) if total_obitos > 0 else 0.0
        label_kpi_etario = "% de Óbitos de Jovens (15–29 anos)"
        delta_kpi_etario = f"{obitos_jovens:,} jovens vitimados".replace(",", ".")
    else:
        total_ctx = len(df_contexto_etario)
        perc_jovens = (total_obitos / total_ctx * 100) if total_ctx > 0 else 0.0
        label_kpi_etario = f"% na Faixa ({faixa_selecionada})"
        delta_kpi_etario = f"{total_obitos:,} de {total_ctx:,} óbitos no período".replace(",", ".")

    contagem_violencia = df_filtrado["tipo_violencia"].value_counts()
    violencia_predominante = contagem_violencia.index[0]
    qtd_violencia_pred = int(contagem_violencia.iloc[0])
    perc_violencia_pred = (qtd_violencia_pred / total_obitos * 100) if total_obitos > 0 else 0.0

    contagem_municipio = df_filtrado["municipio_nome"].value_counts()
    municipio_critico = contagem_municipio.index[0]
    qtd_municipio_critico = int(contagem_municipio.iloc[0])
    perc_municipio_critico = (qtd_municipio_critico / total_obitos * 100) if total_obitos > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total de Óbitos no Período",
            value=f"{total_obitos:,}".replace(",", "."),
            delta=f"Filtro: {'2018 a 2022' if ano_selecionado == 'Todos' else ano_selecionado}",
            delta_color="off",
        )

    with col2:
        st.metric(
            label=label_kpi_etario,
            value=f"{perc_jovens:.1f}%".replace(".", ","),
            delta=delta_kpi_etario,
            delta_color="off",
        )

    with col3:
        st.metric(
            label="Tipo de Violência Predominante",
            value=violencia_predominante,
            delta=f"{perc_violencia_pred:.1f}% do total ({qtd_violencia_pred:,})".replace(",", "."),
            delta_color="off",
        )

    with col4:
        st.metric(
            label="Município com Maior Volume",
            value=municipio_critico,
            delta=f"{qtd_municipio_critico:,} óbitos ({perc_municipio_critico:.1f}%)".replace(",", "."),
            delta_color="off",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    fig_mapa = criar_mapa_coropletico_pe(
        df_filtrado=df_filtrado,
        cidade_selecionada=cidade_selecionada,
    )

    st.plotly_chart(fig_mapa, use_container_width=True)


# ==============================================================================
# 7. PÁGINA 2: ANÁLISES DEMOGRÁFICAS E TENDÊNCIAS ("QUEM" E "QUANDO")
# ==============================================================================
def renderizar_pagina_demografica(
    df_completo: pd.DataFrame,
    df_filtrado: pd.DataFrame,
    ano_selecionado: str,
    faixa_selecionada: str,
    cidade_selecionada: str,
) -> None:
    """
    Renderiza a Página 2 com foco em 'Quem' e 'Quando':
    1. Gráfico de Linha Temporal (Evolução de óbitos por ano e tipo_violencia).
    2. Gráfico de Barras Empilhadas (Raça vs Tipo, filtrado para Jovens 15-29 anos ou faixa ativa).
    3. Gráfico de Rosca / Donut (Proporção de óbitos por sexo_desc).
    """
    st.markdown(
        """
        <div class="civic-header">
            <h1>Análises Demográficas e Tendências Temporais</h1>
            <p>Investigando <b>"Quando"</b> ocorrem as mortes e <b>"Quem"</b> são as populações mais atingidas pela violência em Pernambuco.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_filtrado.empty:
        st.warning("Não há registros para a combinação de filtros selecionada.")
        return

    st.subheader("1. Evolução Temporal dos Óbitos por Tipo de Violência")

    col_ctrl1, col_ctrl2 = st.columns([1.6, 1.4])
    with col_ctrl1:
        recorte_linhas = st.radio(
            "Recorte das Categorias de Violência:",
            options=[
                "Foco Comparativo: Agressão vs Acidente de Trânsito",
                "Todas as Causas Externas (4 categorias)",
            ],
            index=0,
            horizontal=True,
        )

    usar_serie_completa = False
    if ano_selecionado != "Todos":
        with col_ctrl2:
            usar_serie_completa = st.checkbox(
                f"Exibir série histórica completa (2018–2022) destacando {ano_selecionado}",
                value=True,
                help="Como um único ano está selecionado no filtro global, manter a série completa facilita a análise da evolução temporal.",
            )

    if ano_selecionado != "Todos" and usar_serie_completa:
        df_base_temporal = df_completo.copy()
        if faixa_selecionada != "Todas":
            df_base_temporal = df_base_temporal[df_base_temporal["faixa_etaria"] == faixa_selecionada]
        if cidade_selecionada != "Todas":
            df_base_temporal = df_base_temporal[df_base_temporal["municipio_nome"] == cidade_selecionada]
    else:
        df_base_temporal = df_filtrado.copy()

    if "Agressão vs Acidente de Trânsito" in recorte_linhas:
        df_base_temporal = df_base_temporal[
            df_base_temporal["tipo_violencia"].isin(["Agressão", "Acidente de Trânsito"])
        ]

    df_temporal = (
        df_base_temporal.groupby(["ano", "tipo_violencia"], as_index=False)
        .size()
        .rename(columns={"size": "total_obitos"})
        .sort_values("ano")
    )

    fig_linha = px.line(
        df_temporal,
        x="ano",
        y="total_obitos",
        color="tipo_violencia",
        color_discrete_map=CORES_VIOLENCIA,
        markers=True,
        labels={
            "ano": "Ano do Óbito",
            "total_obitos": "Número de Óbitos",
            "tipo_violencia": "Tipo de Violência",
        },
        template="plotly_white",
        height=400,
    )
    fig_linha.update_traces(
        line=dict(width=3.2),
        marker=dict(size=9),
        hovertemplate="<b>%{fullData.name}</b><br>Ano: %{x}<br>Óbitos: %{y:,}<extra></extra>",
    )
    fig_linha.update_layout(
        xaxis=dict(dtick=1, title="Ano"),
        yaxis=dict(title="Total de Óbitos", gridcolor="#F1F5F9"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title_text="Tipo de Violência:",
        ),
        margin={"r": 20, "t": 45, "l": 20, "b": 20},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )

    if ano_selecionado != "Todos" and usar_serie_completa:
        ano_int = int(ano_selecionado)
        fig_linha.add_vline(
            x=ano_int,
            line_width=2,
            line_dash="dash",
            line_color="#C81E1E",
            annotation_text=f"Ano Filtrado: {ano_int}",
            annotation_position="top right",
        )

    st.plotly_chart(fig_linha, use_container_width=True)

    st.markdown("---")

    col_raca, col_sexo = st.columns([1.45, 1])

    with col_raca:
        if faixa_selecionada == "Todas":
            st.subheader("2. Raça/Cor vs Tipo de Violência (Jovens 15–29 anos)")
            st.caption(
                "Filtro interno aplicado automaticamente: **apenas faixa etária `Jovem (15 a 29 anos)`**."
            )
            df_raca_alvo = df_filtrado[
                df_filtrado["faixa_etaria"] == "Jovem (15 a 29 anos)"
            ].copy()
        else:
            st.subheader(f"2. Raça/Cor vs Tipo de Violência ({faixa_selecionada})")
            st.caption(
                f"Exibindo dados para a faixa etária selecionada no filtro global: **`{faixa_selecionada}`**."
            )
            df_raca_alvo = df_filtrado.copy()

        if df_raca_alvo.empty:
            st.info("Sem registros para o recorte selecionado neste gráfico.")
        else:
            ordem_raca = (
                df_raca_alvo["raca_desc"]
                .value_counts()
                .index.tolist()
            )
            df_raca_tipo = (
                df_raca_alvo.groupby(["raca_desc", "tipo_violencia"], as_index=False)
                .size()
                .rename(columns={"size": "total_obitos"})
            )

            fig_barras = px.bar(
                df_raca_tipo,
                x="raca_desc",
                y="total_obitos",
                color="tipo_violencia",
                barmode="stack",
                category_orders={
                    "raca_desc": ordem_raca,
                    "tipo_violencia": [
                        "Agressão",
                        "Intervenção Policial",
                        "Acidente de Trânsito",
                        "Outras Causas Externas",
                    ],
                },
                color_discrete_map=CORES_VIOLENCIA,
                labels={
                    "raca_desc": "Raça / Cor (IBGE/DATASUS)",
                    "total_obitos": "Número de Óbitos",
                    "tipo_violencia": "Tipo de Violência",
                },
                template="plotly_white",
                height=430,
            )
            fig_barras.update_traces(
                hovertemplate="<b>Raça/Cor:</b> %{x}<br><b>Causa:</b> %{fullData.name}<br><b>Óbitos:</b> %{y:,}<extra></extra>"
            )
            fig_barras.update_layout(
                xaxis_title="Raça / Cor",
                yaxis_title="Total de Óbitos",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    title_text="",
                ),
                margin={"r": 15, "t": 45, "l": 15, "b": 15},
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
            )
            st.plotly_chart(fig_barras, use_container_width=True)

            total_recorte = len(df_raca_alvo)
            negros_recorte = int(df_raca_alvo["raca_desc"].isin(["Parda", "Preta"]).sum())
            perc_negros = (negros_recorte / total_recorte * 100) if total_recorte > 0 else 0.0
            st.caption(
                f"**Evidência Demográfica:** Pessoas **Pretas e Pardas** representam "
                f"**{perc_negros:.1f}%** ({negros_recorte:,} de {total_recorte:,}) das mortes "
                f"neste recorte.".replace(",", ".")
            )

    with col_sexo:
        st.subheader("3. Proporção de Óbitos por Sexo")
        st.caption(
            "Distribuição proporcional do total de óbitos segundo a variável `sexo_desc`."
        )

        df_sexo = (
            df_filtrado.groupby("sexo_desc", as_index=False)
            .size()
            .rename(columns={"size": "total_obitos"})
            .sort_values("total_obitos", ascending=False)
        )

        fig_donut = px.pie(
            df_sexo,
            names="sexo_desc",
            values="total_obitos",
            hole=0.58,
            color="sexo_desc",
            color_discrete_map=CORES_SEXO,
            labels={
                "sexo_desc": "Sexo",
                "total_obitos": "Total de Óbitos",
            },
            template="plotly_white",
            height=430,
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Óbitos: %{value:,}<br>Proporção: %{percent}<extra></extra>",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        fig_donut.update_layout(
            annotations=[
                dict(
                    text=f"<b>{len(df_filtrado):,}</b><br>Óbitos".replace(",", "."),
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    font_color="#0F172A",
                    showarrow=False,
                )
            ],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,
                xanchor="center",
                x=0.5,
                title_text="",
            ),
            margin={"r": 15, "t": 35, "l": 15, "b": 35},
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_donut, use_container_width=True)


# ==============================================================================
# 8. PÁGINA 3: FLUXO DA VULNERABILIDADE (DIAGRAMA DE SANKEY)
# ==============================================================================
def construir_dados_sankey(
    df: pd.DataFrame, destacar_fluxos_criticos: bool = False
) -> Tuple[List[str], List[str], List[int], List[int], List[int], List[str]]:
    """
    Agrupa e conta as ocorrências no DataFrame Pandas para estruturar os nós e
    conexões do Diagrama de Sankey na ordem:
    `faixa_etaria` -> `raca_desc` -> `tipo_violencia`.
    """
    ordem_faixa = ["Jovem (15 a 29 anos)", "30+ anos", "0 a 14 anos"]
    faixas_presentes = [f for f in ordem_faixa if f in df["faixa_etaria"].unique()] + [
        f for f in df["faixa_etaria"].unique() if f not in ordem_faixa
    ]

    racas_presentes = df["raca_desc"].value_counts().index.tolist()

    ordem_violencia = [
        "Agressão",
        "Intervenção Policial",
        "Acidente de Trânsito",
        "Outras Causas Externas",
    ]
    violencias_presentes = [
        v for v in ordem_violencia if v in df["tipo_violencia"].unique()
    ] + [v for v in df["tipo_violencia"].unique() if v not in ordem_violencia]

    nos_id: List[str] = (
        [f"FE::{x}" for x in faixas_presentes]
        + [f"RC::{x}" for x in racas_presentes]
        + [f"TV::{x}" for x in violencias_presentes]
    )
    mapa_indice: Dict[str, int] = {chave: idx for idx, chave in enumerate(nos_id)}

    total_geral = len(df)
    contagem_fe = df["faixa_etaria"].value_counts().to_dict()
    contagem_rc = df["raca_desc"].value_counts().to_dict()
    contagem_tv = df["tipo_violencia"].value_counts().to_dict()

    labels_nos: List[str] = []
    cores_nos: List[str] = []

    for chave in nos_id:
        camada, nome = chave.split("::", 1)
        if camada == "FE":
            qtd = contagem_fe.get(nome, 0)
            perc = (qtd / total_geral * 100) if total_geral else 0
            labels_nos.append(f"{nome} ({qtd:,} | {perc:.1f}%)".replace(",", "."))
            if nome == "Jovem (15 a 29 anos)":
                cores_nos.append("#1E293B")
            else:
                cores_nos.append("#475569")
        elif camada == "RC":
            qtd = contagem_rc.get(nome, 0)
            perc = (qtd / total_geral * 100) if total_geral else 0
            labels_nos.append(f"Raça: {nome} ({qtd:,} | {perc:.1f}%)".replace(",", "."))
            cores_nos.append("#334155")
        else:
            qtd = contagem_tv.get(nome, 0)
            perc = (qtd / total_geral * 100) if total_geral else 0
            labels_nos.append(f"{nome} ({qtd:,} | {perc:.1f}%)".replace(",", "."))
            if nome == "Agressão":
                cores_nos.append("#B91C1C")
            elif nome == "Intervenção Policial":
                cores_nos.append("#EF4444")
            elif nome == "Acidente de Trânsito":
                cores_nos.append("#2563EB")
            else:
                cores_nos.append("#64748B")

    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []
    cores_links: List[str] = []

    fluxo_1 = (
        df.groupby(["faixa_etaria", "raca_desc"], as_index=False)
        .size()
        .rename(columns={"size": "valor"})
    )
    for _, row in fluxo_1.iterrows():
        if row["valor"] > 0:
            sources.append(mapa_indice[f"FE::{row['faixa_etaria']}"])
            targets.append(mapa_indice[f"RC::{row['raca_desc']}"])
            values.append(int(row["valor"]))
            cores_links.append("rgba(148, 163, 184, 0.42)")

    fluxo_2 = (
        df.groupby(["raca_desc", "tipo_violencia"], as_index=False)
        .size()
        .rename(columns={"size": "valor"})
    )
    for _, row in fluxo_2.iterrows():
        if row["valor"] > 0:
            sources.append(mapa_indice[f"RC::{row['raca_desc']}"])
            targets.append(mapa_indice[f"TV::{row['tipo_violencia']}"])
            values.append(int(row["valor"]))
            if destacar_fluxos_criticos and row["tipo_violencia"] in (
                "Agressão",
                "Intervenção Policial",
            ):
                cores_links.append("rgba(220, 38, 38, 0.30)")
            else:
                cores_links.append("rgba(148, 163, 184, 0.42)")

    return labels_nos, cores_nos, sources, targets, values, cores_links


def renderizar_pagina_sankey(df_filtrado: pd.DataFrame, ano_selecionado: str) -> None:
    """
    Renderiza a Página 3 inteiramente dedicada ao Diagrama de Sankey com
    `plotly.graph_objects`, demonstrando o fluxo `faixa_etaria` -> `raca_desc` -> `tipo_violencia`.
    """
    st.markdown(
        """
        <div class="civic-header">
            <h1>Fluxo da Vulnerabilidade</h1>
            <p>Trajetória multidimensional das vidas perdidas: <b>Faixa Etária → Raça/Cor → Tipo de Violência</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_filtrado.empty:
        st.warning("Não há registros para a combinação de filtros selecionada.")
        return

    col_info, col_toggle = st.columns([2.4, 1.2])
    with col_info:
        st.markdown(
            "Os nós de **Agressão** e **Intervenção Policial** estão destacados em **tons de vermelho/alerta**, "
            "enquanto as conexões utilizam **cinza translúcido** para evidenciar a espessura dos fluxos demográficos."
        )
    with col_toggle:
        destacar_criticos = st.checkbox(
            "Realçar conexões finais de Agressão / Intervenção Policial",
            value=False,
            help="Por padrão, todas as conexões são cinza translúcido. Marque para aplicar tom avermelhado translúcido nos links que chegam em Agressão e Intervenção Policial.",
        )

    labels_nos, cores_nos, sources, targets, values, cores_links = (
        construir_dados_sankey(df_filtrado, destacar_fluxos_criticos=destacar_criticos)
    )

    fig_sankey = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=",d",
                valuesuffix=" óbitos",
                node=dict(
                    pad=22,
                    thickness=24,
                    line=dict(color="#FFFFFF", width=1.5),
                    label=labels_nos,
                    color=cores_nos,
                    hovertemplate="<b>%{label}</b><br>Volume total no nó: %{value:,}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=cores_links,
                    hovertemplate=(
                        "<b>Fluxo de Vulnerabilidade</b><br>"
                        "Origem: %{source.label}<br>"
                        "Destino: %{target.label}<br>"
                        "<b>Vidas perdidas: %{value:,}</b><extra></extra>"
                    ),
                ),
            )
        ]
    )

    fig_sankey.update_layout(
        title=dict(
            text=(
                f"<b>Fluxo Demográfico de Mortalidade por Causas Externas em Pernambuco "
                f"({'2018–2022' if ano_selecionado == 'Todos' else ano_selecionado})</b>"
            ),
            font=dict(size=16, color="#0F172A"),
        ),
        font=dict(size=13, color="#1E293B", family="sans-serif"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=620,
        margin=dict(l=20, r=20, t=60, b=30),
    )

    st.plotly_chart(fig_sankey, use_container_width=True)

    trilha_principal = (
        df_filtrado.groupby(
            ["faixa_etaria", "raca_desc", "tipo_violencia"], as_index=False
        )
        .size()
        .sort_values("size", ascending=False)
    )
    if not trilha_principal.empty:
        top_row = trilha_principal.iloc[0]
        perc_trilha = top_row["size"] / len(df_filtrado) * 100
        st.info(
            f"**Principal corredor de vulnerabilidade identificado:** "
            f"`{top_row['faixa_etaria']}` → `Raça {top_row['raca_desc']}` → `{top_row['tipo_violencia']}`, "
            f"concentrando **{top_row['size']:,} óbitos** (**{perc_trilha:.1f}%** de todas as mortes do recorte).".replace(
                ",", "."
            )
        )


# ==============================================================================
# 9. FUNÇÃO PRINCIPAL (ENTRYPOINT)
# ==============================================================================
def main() -> None:
    """Ponto de entrada principal da aplicação Streamlit."""
    configurar_pagina()

    try:
        df_completo = carregar_dados(DATA_PATH)
    except Exception as erro:
        st.error(f"Erro ao carregar a base de dados: {erro}")
        st.stop()

    (
        pagina_selecionada,
        ano_selecionado,
        faixa_selecionada,
        cidade_selecionada,
        df_filtrado,
        df_contexto_etario,
    ) = renderizar_sidebar(df_completo)

    if pagina_selecionada == "Visão Geral e Mapa":
        renderizar_pagina_mapa(
            df_filtrado,
            df_contexto_etario,
            ano_selecionado,
            faixa_selecionada,
            cidade_selecionada,
        )
    elif pagina_selecionada == "Análises Demográficas e Tendências":
        renderizar_pagina_demografica(
            df_completo,
            df_filtrado,
            ano_selecionado,
            faixa_selecionada,
            cidade_selecionada,
        )
    elif pagina_selecionada == "Fluxo da Vulnerabilidade":
        renderizar_pagina_sankey(df_filtrado, ano_selecionado)


if __name__ == "__main__":
    main()
