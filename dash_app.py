"""
Dashboard Interativo Civic Tech — Juventude Interrompida (Versão Plotly Dash)
=============================================================================
Versão nativa em Plotly Dash pronta para hospedagem direta no Plotly Cloud
(https://cloud.plotly.com), Dash Enterprise ou Render/Gunicorn.

Execução local: python dash_app.py (acesse http://localhost:8050)
Servidor WSGI (Produção/Plotly Cloud): dash_app:server
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

DATA_PATH = Path(__file__).parent / "sim_violencia_jovem_pe.csv"
GEOJSON_PATH = Path(__file__).parent / "pe_municipios.geojson"
CONTORNO_UF_PATH = Path(__file__).parent / "pe_contorno_uf.json"

PE_STATE_CENTER = {"lat": -8.3500, "lon": -37.2000}

CORES_VIOLENCIA: Dict[str, str] = {
    "Agressão": "#C81E1E",
    "Intervenção Policial": "#EA580C",
    "Acidente de Trânsito": "#2563EB",
    "Outras Causas Externas": "#64748B",
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


@lru_cache(maxsize=1)
def carregar_malha_pe() -> Dict[str, Any]:
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def carregar_contorno_uf() -> List[Any]:
    if CONTORNO_UF_PATH.exists():
        with open(CONTORNO_UF_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _extrair_aneis_poligono(geometria: Dict[str, Any]) -> List[List[List[float]]]:
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
    geojson_pe = carregar_malha_pe()
    contorno_uf = carregar_contorno_uf()
    cores_classes = CORES_CLASSES_MAPA

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

    for idx_classe in range(5):
        xs = xs_por_classe[idx_classe]
        ys = ys_por_classe[idx_classe]
        if not xs:
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

    fig.update_layout(
        title=dict(
            text="<b>Mapa Coroplético de Mortalidade por Causas Externas em Pernambuco</b>",
            font=dict(size=16, color="#0F172A"),
            x=0.01,
        ),
        xaxis=dict(
            range=[-41.45, -34.60],
            tickvals=[-41, -40, -39, -38, -37, -36, -35],
            ticktext=["41°W", "40°W", "39°W", "38°W", "37°W", "36°W", "35°W"],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            title="",
        ),
        yaxis=dict(
            range=[-9.60, -7.15],
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


def construir_dados_sankey(
    df: pd.DataFrame, destacar_fluxos_criticos: bool = False
) -> Tuple[List[str], List[str], List[int], List[int], List[int], List[str]]:
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
            cores_nos.append("#1E293B" if nome == "Jovem (15 a 29 anos)" else "#475569")
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


def carregar_dados_dash(caminho_csv: Path) -> pd.DataFrame:
    """Carrega e higieniza os dados do SIM/DATASUS sem dependência do runtime Streamlit."""
    df = pd.read_csv(caminho_csv, encoding="utf-8")
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


DF_COMPLETO = carregar_dados_dash(DATA_PATH)

ANOS_DISPONIVEIS = sorted(DF_COMPLETO["ano"].unique().tolist(), reverse=True)
OPCOES_ANO = [{"label": "Todos (2018–2022)", "value": "Todos"}] + [
    {"label": str(ano), "value": str(ano)} for ano in ANOS_DISPONIVEIS
]

ORDEM_FAIXA = ["Jovem (15 a 29 anos)", "30+ anos", "0 a 14 anos"]
FAIXAS_DISPONIVEIS = [f for f in ORDEM_FAIXA if f in DF_COMPLETO["faixa_etaria"].unique()] + [
    f for f in DF_COMPLETO["faixa_etaria"].unique() if f not in ORDEM_FAIXA
]
OPCOES_FAIXA = [{"label": "Todas as Faixas Etárias", "value": "Todas"}] + [
    {"label": faixa, "value": faixa} for faixa in FAIXAS_DISPONIVEIS
]

CIDADES_DISPONIVEIS = sorted(DF_COMPLETO["municipio_nome"].unique().tolist())
OPCOES_CIDADE = [{"label": "Todas as Cidades (PE)", "value": "Todas"}] + [
    {"label": cidade, "value": cidade} for cidade in CIDADES_DISPONIVEIS
]

ESTILO_BOTAO_INATIVO = {
    "width": "100%",
    "textAlign": "left",
    "padding": "10px 14px",
    "marginBottom": "8px",
    "borderRadius": "8px",
    "border": "1px solid #CBD5E1",
    "backgroundColor": "#FFFFFF",
    "color": "#1E293B",
    "fontWeight": "600",
    "fontSize": "0.88rem",
    "cursor": "pointer",
    "transition": "all 0.15s ease",
}

ESTILO_BOTAO_ATIVO = {
    **ESTILO_BOTAO_INATIVO,
    "backgroundColor": "#0F172A",
    "border": "1px solid #0F172A",
    "color": "#FFFFFF",
    "fontWeight": "700",
    "boxShadow": "0 2px 6px rgba(15, 23, 42, 0.25)",
}

app = Dash(
    __name__,
    title="Juventude Interrompida | Observatório PE (Plotly)",
    suppress_callback_exceptions=True,
)
# Variável WSGI exigida pelo Plotly Cloud / Gunicorn
server = app.server

app.index_string = """<!DOCTYPE html>
<html lang="pt-BR">
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                box-sizing: border-box;
            }
            :root {
                --Dash-Fill-Interactive-Strong: #0F172A !important;
                --Dash-Fill-Interactive-Weak: #1E293B !important;
                --Dash-Stroke-Strong: #0F172A !important;
                --Dash-Text-Primary: #0F172A !important;
            }
            /* Hover nos botões de navegação */
            #btn-pag1:hover, #btn-pag2:hover, #btn-pag3:hover {
                border-color: #0F172A !important;
                box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.15) !important;
            }
            /* Hover e foco nos filtros Dropdown (Dash 2 e Dash 4) */
            .dash-dropdown .Select-control:hover,
            .Select-control:hover,
            .dash-dropdown button:hover,
            .dash-dropdown [role="combobox"]:hover {
                border-color: #0F172A !important;
            }
            .Select.is-focused > .Select-control,
            .Select.is-open > .Select-control,
            .dash-dropdown button:focus,
            .dash-dropdown [role="combobox"]:focus,
            .dash-dropdown [data-state="open"] {
                border-color: #0F172A !important;
                box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.2) !important;
                outline: none !important;
            }
            /* Opções na lista do filtro: Hover e Seleção em Preto */
            .VirtualizedSelectOption:hover,
            .VirtualizedSelectFocusedOption,
            .Select-option.is-focused,
            .dash-dropdown-option:hover,
            [role="listbox"] [role="option"]:hover,
            [role="listbox"] [role="option"][data-highlighted] {
                background-color: #1E293B !important;
                color: #FFFFFF !important;
                cursor: pointer !important;
            }
            .VirtualizedSelectSelectedOption,
            .Select-option.is-selected,
            .dash-dropdown-option[aria-selected="true"],
            [role="listbox"] [role="option"][aria-selected="true"],
            [role="listbox"] [role="option"][data-state="checked"] {
                background-color: #0F172A !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
            }
            /* Layout Desktop e Responsividade Mobile/Tablet */
            .app-shell {
                display: flex;
                flex-direction: row;
                min-height: 100vh;
                background-color: #FFFFFF;
                color: #1E293B;
                font-family: Inter, system-ui, -apple-system, sans-serif;
            }
            .app-sidebar {
                width: 300px;
                background-color: #F8FAFC;
                border-right: 1px solid #E2E8F0;
                padding: 24px 20px;
                flex-shrink: 0;
            }
            .app-main {
                flex: 1;
                min-width: 0;
                padding: 28px 34px;
                overflow-x: hidden;
            }
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .charts-grid {
                display: grid;
                grid-template-columns: 1.4fr 1fr;
                gap: 20px;
                margin-top: 16px;
            }
            .page-title {
                margin-top: 0;
                font-size: 1.65rem;
                color: #0F172A;
            }
            @media (max-width: 1100px) {
                .kpi-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
                .charts-grid {
                    grid-template-columns: 1fr;
                }
            }
            @media (max-width: 768px) {
                .app-shell {
                    flex-direction: column;
                }
                .app-sidebar {
                    width: 100% !important;
                    border-right: none !important;
                    border-bottom: 1px solid #E2E8F0 !important;
                    padding: 16px !important;
                }
                .app-main {
                    width: 100% !important;
                    padding: 16px 12px !important;
                }
                .kpi-grid {
                    grid-template-columns: repeat(2, 1fr);
                    gap: 10px;
                    margin-bottom: 16px;
                }
                .page-title {
                    font-size: 1.25rem !important;
                }
            }
            @media (max-width: 480px) {
                .kpi-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body style="margin: 0;">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def criar_kpi_card(titulo: str, valor: str, subtitulo: str) -> html.Div:
    return html.Div(
        className="kpi-card",
        style={
            "backgroundColor": "#F8FAFC",
            "border": "1px solid #E2E8F0",
            "borderRadius": "10px",
            "padding": "16px 20px",
            "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.04)",
        },
        children=[
            html.Div(
                titulo,
                style={"color": "#475569", "fontSize": "0.88rem", "fontWeight": "600"},
            ),
            html.Div(
                valor,
                style={
                    "color": "#0F172A",
                    "fontSize": "1.55rem",
                    "fontWeight": "800",
                    "margin": "6px 0",
                    "wordBreak": "break-word",
                },
            ),
            html.Div(
                subtitulo,
                style={"color": "#64748B", "fontSize": "0.82rem"},
            ),
        ],
    )


app.layout = html.Div(
    className="app-shell",
    children=[
        dcc.Store(id="pagina-ativa", data="pag1"),
        # Barra Lateral (Sidebar)
        html.Aside(
            className="app-sidebar",
            children=[
                html.H2(
                    "Juventude Interrompida",
                    style={"margin": "0 0 6px 0", "fontSize": "1.25rem", "color": "#0F172A"},
                ),
                html.P(
                    "Observatório Cívico de Mortalidade por Causas Externas em Pernambuco (SIM/DATASUS)",
                    style={"fontSize": "0.82rem", "color": "#64748B", "marginBottom": "20px"},
                ),
                html.Hr(style={"borderColor": "#E2E8F0", "margin": "16px 0"}),
                html.Label(
                    "Navegação do Painel",
                    style={"fontWeight": "700", "fontSize": "0.9rem", "display": "block", "marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Button(
                            "Visão Geral e Mapa",
                            id="btn-pag1",
                            n_clicks=0,
                            style=ESTILO_BOTAO_ATIVO,
                        ),
                        html.Button(
                            "Análises Demográficas e Tendências",
                            id="btn-pag2",
                            n_clicks=0,
                            style=ESTILO_BOTAO_INATIVO,
                        ),
                        html.Button(
                            "Fluxo da Vulnerabilidade",
                            id="btn-pag3",
                            n_clicks=0,
                            style=ESTILO_BOTAO_INATIVO,
                        ),
                    ]
                ),
                html.Hr(style={"borderColor": "#E2E8F0", "margin": "20px 0"}),
                html.Div(
                    "Filtros Globais",
                    style={"fontWeight": "800", "fontSize": "0.95rem", "marginBottom": "12px", "color": "#0F172A"},
                ),
                html.Label(
                    "Ano de Referência",
                    style={"fontWeight": "600", "fontSize": "0.85rem", "display": "block", "marginBottom": "6px"},
                ),
                dcc.Dropdown(
                    id="filtro-ano",
                    options=OPCOES_ANO,
                    value="Todos",
                    clearable=False,
                    style={"fontSize": "0.9rem", "marginBottom": "14px"},
                ),
                html.Label(
                    "Faixa Etária",
                    style={"fontWeight": "600", "fontSize": "0.85rem", "display": "block", "marginBottom": "6px"},
                ),
                dcc.Dropdown(
                    id="filtro-faixa",
                    options=OPCOES_FAIXA,
                    value="Todas",
                    clearable=False,
                    style={"fontSize": "0.9rem", "marginBottom": "14px"},
                ),
                html.Label(
                    "Cidade (Município)",
                    style={"fontWeight": "600", "fontSize": "0.85rem", "display": "block", "marginBottom": "6px"},
                ),
                dcc.Dropdown(
                    id="filtro-cidade",
                    options=OPCOES_CIDADE,
                    value="Todas",
                    clearable=False,
                    placeholder="Selecione ou digite uma cidade...",
                    style={"fontSize": "0.9rem", "marginBottom": "14px"},
                ),
                html.Div(
                    id="resumo-sidebar",
                    style={
                        "marginTop": "16px",
                        "padding": "12px",
                        "backgroundColor": "#FFFFFF",
                        "border": "1px solid #E2E8F0",
                        "borderRadius": "8px",
                        "fontSize": "0.82rem",
                        "color": "#334155",
                        "lineHeight": "1.5",
                    },
                ),
            ],
        ),
        # Conteúdo Principal
        html.Main(
            id="conteudo-pagina",
            className="app-main",
        ),
    ],
)


@app.callback(
    Output("pagina-ativa", "data"),
    Output("btn-pag1", "style"),
    Output("btn-pag2", "style"),
    Output("btn-pag3", "style"),
    Output("resumo-sidebar", "children"),
    Output("conteudo-pagina", "children"),
    Input("btn-pag1", "n_clicks"),
    Input("btn-pag2", "n_clicks"),
    Input("btn-pag3", "n_clicks"),
    Input("filtro-ano", "value"),
    Input("filtro-faixa", "value"),
    Input("filtro-cidade", "value"),
    State("pagina-ativa", "data"),
)
def atualizar_dashboard(
    _n1: int,
    _n2: int,
    _n3: int,
    ano_selecionado: str,
    faixa_selecionada: str,
    cidade_selecionada: str,
    pagina_atual: str,
):
    trigger = ctx.triggered_id
    if trigger == "btn-pag1":
        pagina = "pag1"
    elif trigger == "btn-pag2":
        pagina = "pag2"
    elif trigger == "btn-pag3":
        pagina = "pag3"
    else:
        pagina = pagina_atual or "pag1"

    estilo_1 = ESTILO_BOTAO_ATIVO if pagina == "pag1" else ESTILO_BOTAO_INATIVO
    estilo_2 = ESTILO_BOTAO_ATIVO if pagina == "pag2" else ESTILO_BOTAO_INATIVO
    estilo_3 = ESTILO_BOTAO_ATIVO if pagina == "pag3" else ESTILO_BOTAO_INATIVO

    df_filtrado = DF_COMPLETO.copy()

    if ano_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado["ano"] == int(ano_selecionado)]
        periodo_txt = f"Ano {ano_selecionado}"
    else:
        periodo_txt = "2018–2022 (Todos)"

    if cidade_selecionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado["municipio_nome"] == cidade_selecionada]

    df_contexto_etario = df_filtrado.copy()

    if faixa_selecionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado["faixa_etaria"] == faixa_selecionada]

    resumo = [
        html.Div([html.B("Período: "), periodo_txt]),
        html.Div([html.B("Faixa Etária: "), faixa_selecionada]),
        html.Div([html.B("Cidade: "), cidade_selecionada]),
        html.Div([html.B("Registros: "), f"{len(df_filtrado):,}".replace(",", "."), " óbitos"]),
        html.Div([html.B("Municípios: "), str(df_filtrado["id_municipio"].nunique())]),
    ]

    if df_filtrado.empty:
        return (
            pagina,
            estilo_1,
            estilo_2,
            estilo_3,
            resumo,
            html.Div(
                style={
                    "padding": "24px",
                    "border": "1px solid #E2E8F0",
                    "borderRadius": "10px",
                    "backgroundColor": "#F8FAFC",
                },
                children=[
                    html.H3("Nenhum registro encontrado para a combinação de filtros selecionada."),
                    html.P("Altere os filtros de Ano, Faixa Etária ou Cidade na barra lateral."),
                ],
            ),
        )

    if pagina == "pag1":
        total = len(df_filtrado)
        if faixa_selecionada == "Todas":
            jovens = int((df_filtrado["faixa_etaria"] == "Jovem (15 a 29 anos)").sum())
            perc_jovens = (jovens / total * 100) if total else 0.0
            sub_jovens = f"{jovens:,} jovens vitimados".replace(",", ".")
        else:
            total_ctx = len(df_contexto_etario)
            perc_jovens = (total / total_ctx * 100) if total_ctx else 0.0
            sub_jovens = f"Participação de '{faixa_selecionada}' no recorte"

        vc_v = df_filtrado["tipo_violencia"].value_counts()
        v_top, v_qtd = vc_v.index[0], int(vc_v.iloc[0])
        vc_m = df_filtrado["municipio_nome"].value_counts()
        m_top, m_qtd = vc_m.index[0], int(vc_m.iloc[0])

        fig_mapa = criar_mapa_coropletico_pe(
            df_filtrado=df_filtrado,
            cidade_selecionada=cidade_selecionada,
        )

        return (
            pagina,
            estilo_1,
            estilo_2,
            estilo_3,
            resumo,
            html.Div(
                [
                    html.H1("Visão Geral e Distribuição Territorial (Pernambuco)", className="page-title"),
                    html.Div(
                        className="kpi-grid",
                        children=[
                            criar_kpi_card(
                                "Total de Óbitos no Período",
                                f"{total:,}".replace(",", "."),
                                f"Filtro: {periodo_txt}",
                            ),
                            criar_kpi_card(
                                "% de Óbitos de Jovens (15–29)" if faixa_selecionada == "Todas" else f"% na Faixa ({faixa_selecionada})",
                                f"{perc_jovens:.1f}%".replace(".", ","),
                                sub_jovens,
                            ),
                            criar_kpi_card(
                                "Violência Predominante",
                                v_top,
                                f"{v_qtd:,} casos ({v_qtd/total*100:.1f}%)".replace(",", "."),
                            ),
                            criar_kpi_card(
                                "Município em Destaque",
                                m_top,
                                f"{m_qtd:,} óbitos ({m_qtd/total*100:.1f}%)".replace(",", "."),
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "border": "1px solid #E2E8F0",
                            "borderRadius": "10px",
                            "padding": "8px",
                            "backgroundColor": "#FFFFFF",
                        },
                        children=[dcc.Graph(figure=fig_mapa, config={"responsive": True})],
                    ),
                ]
            ),
        )

    if pagina == "pag2":
        df_base_temp = DF_COMPLETO.copy()
        if faixa_selecionada != "Todas":
            df_base_temp = df_base_temp[df_base_temp["faixa_etaria"] == faixa_selecionada]
        if cidade_selecionada != "Todas":
            df_base_temp = df_base_temp[df_base_temp["municipio_nome"] == cidade_selecionada]

        df_temp = (
            df_base_temp.groupby(["ano", "tipo_violencia"], as_index=False)
            .size()
            .rename(columns={"size": "total_obitos"})
            .sort_values("ano")
        )
        fig_linha = px.line(
            df_temp,
            x="ano",
            y="total_obitos",
            color="tipo_violencia",
            color_discrete_map=CORES_VIOLENCIA,
            markers=True,
            title="Evolução Temporal de Óbitos por Tipo de Violência (2018–2022)",
            labels={
                "ano": "Ano do Óbito",
                "total_obitos": "Número de Óbitos",
                "tipo_violencia": "Tipo de Violência",
            },
            template="plotly_white",
            height=370,
        )
        fig_linha.update_layout(
            xaxis=dict(dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, title_text=""),
            margin={"r": 15, "t": 50, "l": 15, "b": 70},
        )

        if faixa_selecionada == "Todas":
            df_raca_base = df_filtrado[df_filtrado["faixa_etaria"] == "Jovem (15 a 29 anos)"]
            titulo_barras = "Raça/Cor vs Tipo de Violência — Filtro Interno: Jovem (15 a 29 anos)"
        else:
            df_raca_base = df_filtrado
            titulo_barras = f"Raça/Cor vs Tipo de Violência — Faixa: {faixa_selecionada}"

        df_raca = (
            df_raca_base.groupby(["raca_desc", "tipo_violencia"], as_index=False)
            .size()
            .rename(columns={"size": "total_obitos"})
        )
        fig_barras = px.bar(
            df_raca,
            x="raca_desc",
            y="total_obitos",
            color="tipo_violencia",
            barmode="stack",
            color_discrete_map=CORES_VIOLENCIA,
            title=titulo_barras,
            labels={
                "raca_desc": "Raça / Cor",
                "total_obitos": "Número de Óbitos",
                "tipo_violencia": "Tipo de Violência",
            },
            template="plotly_white",
            height=410,
        )
        fig_barras.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, title_text=""),
            margin={"r": 15, "t": 50, "l": 15, "b": 70},
        )

        df_sexo = (
            df_filtrado.groupby("sexo_desc", as_index=False)
            .size()
            .rename(columns={"size": "total_obitos"})
        )
        fig_donut = px.pie(
            df_sexo,
            names="sexo_desc",
            values="total_obitos",
            hole=0.58,
            color="sexo_desc",
            color_discrete_map=CORES_SEXO,
            title="Proporção de Óbitos por Sexo",
            template="plotly_white",
            height=410,
        )
        fig_donut.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, title_text=""),
            margin={"r": 15, "t": 50, "l": 15, "b": 50},
        )

        return (
            pagina,
            estilo_1,
            estilo_2,
            estilo_3,
            resumo,
            html.Div(
                [
                    html.H1("Análises Demográficas e Tendências Temporais", className="page-title"),
                    dcc.Graph(figure=fig_linha, config={"responsive": True}),
                    html.Div(
                        className="charts-grid",
                        children=[
                            dcc.Graph(figure=fig_barras, config={"responsive": True}),
                            dcc.Graph(figure=fig_donut, config={"responsive": True}),
                        ],
                    ),
                ]
            ),
        )

    # Página 3: Fluxo da Vulnerabilidade
    labels_nos, cores_nos, sources, targets, values, cores_links = construir_dados_sankey(df_filtrado)
    fig_sankey = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(pad=22, thickness=24, label=labels_nos, color=cores_nos),
                link=dict(source=sources, target=targets, value=values, color=cores_links),
            )
        ]
    )
    fig_sankey.update_layout(
        title=f"Fluxo da Vulnerabilidade: Faixa Etária → Raça/Cor → Tipo de Violência ({periodo_txt})",
        template="plotly_white",
        height=600,
        margin={"r": 15, "t": 55, "l": 15, "b": 25},
    )
    return (
        pagina,
        estilo_1,
        estilo_2,
        estilo_3,
        resumo,
        html.Div(
            [
                html.H1("Fluxo da Vulnerabilidade", className="page-title"),
                dcc.Graph(figure=fig_sankey, config={"responsive": True}),
            ]
        ),
    )


if __name__ == "__main__":
    app.run(debug=False, port=8050)
