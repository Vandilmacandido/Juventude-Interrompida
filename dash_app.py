"""
Dashboard Interativo Civic Tech — Juventude Interrompida (Versão Plotly Dash)
=============================================================================
Versão nativa em Plotly Dash pronta para hospedagem direta no Plotly Cloud
(https://cloud.plotly.com), Dash Enterprise ou Render/Gunicorn.

Execução local: python dash_app.py (acesse http://localhost:8050)
Servidor WSGI (Produção/Plotly Cloud): dash_app:server
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

from app import (
    CORES_SEXO,
    CORES_VIOLENCIA,
    DATA_PATH,
    MUNICIPIOS_PE_COORDS,
    PE_STATE_CENTER,
    construir_dados_sankey,
    criar_mapa_coropletico_pe,
)


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
    "backgroundColor": "#2563EB",
    "border": "1px solid #2563EB",
    "color": "#FFFFFF",
    "fontWeight": "700",
    "boxShadow": "0 2px 6px rgba(37, 99, 235, 0.25)",
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
                --Dash-Fill-Interactive-Strong: #2563EB !important;
                --Dash-Fill-Interactive-Weak: #DBEAFE !important;
                --Dash-Stroke-Strong: #2563EB !important;
                --Dash-Text-Primary: #0F172A !important;
            }
            /* Hover nos botões de navegação */
            #btn-pag1:hover, #btn-pag2:hover, #btn-pag3:hover {
                border-color: #2563EB !important;
                box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
            }
            /* Hover e foco nos filtros Dropdown (Dash 2 e Dash 4) */
            .dash-dropdown .Select-control:hover,
            .Select-control:hover,
            .dash-dropdown button:hover,
            .dash-dropdown [role="combobox"]:hover {
                border-color: #2563EB !important;
            }
            .Select.is-focused > .Select-control,
            .Select.is-open > .Select-control,
            .dash-dropdown button:focus,
            .dash-dropdown [role="combobox"]:focus,
            .dash-dropdown [data-state="open"] {
                border-color: #2563EB !important;
                box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
                outline: none !important;
            }
            /* Opções na lista do filtro: Hover e Seleção em Azul */
            .VirtualizedSelectOption:hover,
            .VirtualizedSelectFocusedOption,
            .Select-option.is-focused,
            .dash-dropdown-option:hover,
            [role="listbox"] [role="option"]:hover,
            [role="listbox"] [role="option"][data-highlighted] {
                background-color: #DBEAFE !important;
                color: #1E3A8A !important;
                cursor: pointer !important;
            }
            .VirtualizedSelectSelectedOption,
            .Select-option.is-selected,
            .dash-dropdown-option[aria-selected="true"],
            [role="listbox"] [role="option"][aria-selected="true"],
            [role="listbox"] [role="option"][data-state="checked"] {
                background-color: #2563EB !important;
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
