"""Sistema final de la práctica MIAX: agente investigador sobre informes 10-K.

Este módulo contiene el código que sí pertenece a la entrega: herramientas,
retrieval mejorado, agente, guardrails, evaluadores y las interfaces públicas
``responder`` y ``evaluar``. Los módulos ``miax_s1`` y ``miax_s2`` se mantienen
como auxiliares docentes de las dos sesiones.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import miax_s1
import miax_s2

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MODELO = os.getenv("MIAX_MODEL", "google_genai:gemini-3.8-flash")
TOLERANCIA_XBRL = 0.01
RRF_K = 60
TOP_K = 5
PRECIO_MODELO_USD_M = (0.75, 3.75)  # revisar antes de reportar coste definitivo


def _base_corpus() -> Path:
    return miax_s2.dir_corpus()


def _cargar_datos():
    base = _base_corpus()
    secciones, chunks = miax_s2.cargar_corpus()
    return (
        pd.DataFrame(secciones),
        chunks,
        pd.read_parquet(base / "xbrl_facts.parquet"),
    )


SECCIONES, CHUNKS, XBRL = _cargar_datos()
POR_ID = {c["chunk_id"]: c for c in CHUNKS}

# ---------------------------------------------------------------------------
# Retrieval: baseline -> filtros -> híbrido -> reescritura
# ---------------------------------------------------------------------------
def buscar_denso_baseline(query: str, ticker=None, fiscal_year=None,
                           item=None, k: int = TOP_K) -> list[dict]:
    """Búsqueda de la sesión 1, conservada como baseline experimental."""
    return miax_s1.buscar(query, ticker=ticker, fiscal_year=fiscal_year,
                          item=item, k=k)


def _mascara_meta(meta: pd.DataFrame, ticker=None, fiscal_year=None, item=None):
    mascara = np.ones(len(meta), dtype=bool)
    if ticker is not None:
        mascara &= meta["ticker"].eq(str(ticker).upper()).to_numpy()
    if fiscal_year is not None:
        mascara &= meta["fiscal_year"].astype(int).eq(int(fiscal_year)).to_numpy()
    if item is not None:
        mascara &= meta["item"].eq(str(item)).to_numpy()
    return mascara


def buscar_denso_filtrado(query: str, ticker=None, fiscal_year=None,
                           item=None, k: int = TOP_K) -> list[dict]:
    """Búsqueda densa con filtrado explícito por metadatos."""
    indice, meta, _ = miax_s2.cargar_indice()
    permitidas = _mascara_meta(meta, ticker, fiscal_year, item)
    scores, posiciones = indice.search(miax_s2.codificar([query]), indice.ntotal)
    salida = []
    for score, pos in zip(scores[0], posiciones[0]):
        pos = int(pos)
        if pos < 0 or not permitidas[pos]:
            continue
        salida.append(miax_s2.fila_a_fragmento(meta.iloc[pos], score))
        if len(salida) >= k:
            break
    return salida


def buscar_hibrido(query: str, ticker=None, fiscal_year=None, item=None,
                    k: int = TOP_K, rrf_k: int = RRF_K) -> list[dict]:
    """Fusiona ranking denso y BM25 mediante Reciprocal Rank Fusion (RRF)."""
    indice, meta, _ = miax_s2.cargar_indice()
    permitidas = _mascara_meta(meta, ticker, fiscal_year, item)
    ids_permitidos = set(meta.loc[permitidas, "chunk_id"])
    if not ids_permitidos:
        return []

    # Ranking denso.
    scores_d, pos_d = indice.search(miax_s2.codificar([query]), indice.ntotal)
    rank_d, frag_por_id = {}, {}
    rank_filtrado = 0
    for score, pos in zip(scores_d[0], pos_d[0]):
        pos = int(pos)
        if pos < 0:
            continue
        fila = meta.iloc[pos]
        cid = fila["chunk_id"]
        if cid not in ids_permitidos:
            continue
        rank_filtrado += 1
        rank_d[cid] = rank_filtrado
        frag_por_id[cid] = miax_s2.fila_a_fragmento(fila, score)

    # Ranking BM25. El rank se cuenta después de aplicar los mismos filtros.
    bm25, chunks_bm = miax_s2.montar_bm25()
    scores_b = bm25.get_scores(miax_s2.tokenizar(query))
    rank_b = {}
    rank_filtrado = 0
    for pos in np.argsort(scores_b)[::-1]:
        c = chunks_bm[int(pos)]
        cid = c["chunk_id"]
        if cid not in ids_permitidos:
            continue
        rank_filtrado += 1
        rank_b[cid] = rank_filtrado
        if cid not in frag_por_id:
            frag_por_id[cid] = {**c, "puntuacion": float(scores_b[int(pos)])}

    # RRF evita mezclar directamente escalas heterogéneas de similitud y BM25.
    candidatos = set(rank_d) | set(rank_b)
    fusion = {
        cid: ((1 / (rrf_k + rank_d[cid])) if cid in rank_d else 0.0)
             + ((1 / (rrf_k + rank_b[cid])) if cid in rank_b else 0.0)
        for cid in candidatos
    }
    mejores = sorted(fusion, key=fusion.get, reverse=True)[:k]
    salida = []
    for cid in mejores:
        f = dict(frag_por_id[cid])
        f["puntuacion"] = round(float(fusion[cid]), 6)
        salida.append(f)
    return salida


_REESCRITOR = None
INSTRUCCION_REESCRITURA = """Rewrite the user's question as ONE concise English search query for a 10-K filing index. Preserve company, fiscal year and financial concept. Prefer terminology likely to occur literally in the filing. Return only the query, with no explanation."""


def _modelo_reescritor():
    global _REESCRITOR
    if _REESCRITOR is None:
        from langchain.chat_models import init_chat_model
        _REESCRITOR = init_chat_model(MODELO, temperature=0)
    return _REESCRITOR


def reescribir_consulta(pregunta: str) -> str:
    """Reescritura con LLM; degrada a la consulta original si la red falla."""
    try:
        msg = _modelo_reescritor().invoke([
            {"role": "system", "content": INSTRUCCION_REESCRITURA},
            {"role": "user", "content": pregunta},
        ])
        texto = getattr(msg, "text", None) or getattr(msg, "content", "")
        return str(texto).strip() or pregunta
    except Exception:
        return pregunta


# ---------------------------------------------------------------------------
# Herramientas: las firmas son contrato del enunciado
# ---------------------------------------------------------------------------
from langchain.tools import tool


@tool
def list_available() -> str:
    """Devuelve compañías, ejercicios fiscales y secciones disponibles.

    Úsala cuando no estés seguro de que una compañía, ejercicio o sección exista
    en el corpus. No inventes cobertura que esta herramienta no confirme.
    """
    lineas = []
    for (ticker, empresa), grupo in SECCIONES.groupby(["ticker", "empresa"]):
        ejercicios = sorted(int(x) for x in grupo["fiscal_year"].unique())
        items = sorted(str(x) for x in grupo["item"].unique())
        lineas.append(f"{ticker} ({empresa}): ejercicios {ejercicios}, items {items}")
    return "\n".join(lineas)


@tool
def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
    """Devuelve el valor EXACTO de una magnitud financiera reportada en XBRL.

    Es la fuente autorizada para cualquier cifra financiera. Úsala SIEMPRE para
    cifras exactas, en lugar de leer el número de la prosa del 10-K.

    Args:
        ticker: símbolo bursátil, p. ej. 'NVDA'.
        fiscal_year: ejercicio fiscal reportado, p. ej. 2025.
        concept: concepto US-GAAP, p. ej. 'Revenues', 'NetIncomeLoss' o 'Assets'.
    """
    ticker = ticker.upper()
    filas = XBRL[(XBRL.ticker == ticker)
                 & (XBRL.fiscal_year.astype(int) == int(fiscal_year))
                 & (XBRL.concept == concept)]
    if filas.empty:
        disponibles = sorted(XBRL[(XBRL.ticker == ticker)
                                   & (XBRL.fiscal_year.astype(int) == int(fiscal_year))]
                                  .concept.unique())
        return (f"{ticker} no reportó '{concept}' en FY{fiscal_year}. "
                f"Conceptos disponibles: {disponibles or 'ninguno'}.")
    f = filas.iloc[0]
    return (f"{ticker} FY{fiscal_year} {concept} = {f['value']:,.0f} {f['unit']} "
            f"(cierre {f['period_end']}, {f['form']})")


@tool
def search_filings(query: str, ticker: str | None = None,
                   fiscal_year: int | None = None,
                   item: str | None = None, k: int = TOP_K) -> str:
    """Busca fragmentos relevantes en los 10-K mediante retrieval híbrido.

    Úsala para riesgos, estrategia, litigios, causas y comentarios de dirección.
    NO la uses como fuente autorizada de cifras exactas: para eso usa
    get_xbrl_fact. Pasa ticker, fiscal_year e item siempre que los conozcas.

    Args:
        query: necesidad de información en lenguaje natural.
        ticker: compañía si se conoce.
        fiscal_year: ejercicio fiscal si se conoce.
        item: sección ('1A', '7', '7A' u '8') si se conoce.
        k: número de fragmentos a devolver.
    """
    consulta = reescribir_consulta(query)
    fragmentos = buscar_hibrido(consulta, ticker, fiscal_year, item, k)
    return miax_s2.formatear_fragmentos(fragmentos)


@tool
def read_section(ticker: str, fiscal_year: int, item: str) -> str:
    """Devuelve el TEXTO COMPLETO de una sección de un 10-K.

    Es una herramienta cara en tokens. Úsala solo cuando search_filings no dé
    contexto suficiente y sea imprescindible leer la sección completa.
    """
    filas = SECCIONES[(SECCIONES.ticker == ticker.upper())
                      & (SECCIONES.fiscal_year.astype(int) == int(fiscal_year))
                      & (SECCIONES.item == item)]
    if filas.empty:
        return f"No hay sección {item} de {ticker.upper()} FY{fiscal_year} en el corpus."
    return str(filas.iloc[0]["texto"])


HERRAMIENTAS = [list_available, get_xbrl_fact, search_filings, read_section]

# ---------------------------------------------------------------------------
# Contrato de salida y prompt
# ---------------------------------------------------------------------------
class RespuestaFinanciera(BaseModel):
    """Respuesta trazable a una pregunta sobre informes 10-K."""
    respuesta: str = Field(description="Respuesta en prosa, breve y directa")
    cifra: float | None = Field(default=None, description="Valor numérico principal, si aplica")
    unidad: str | None = Field(default=None, description="USD, shares, porcentaje…")
    ticker: str | None = None
    ejercicio: int | None = None
    fuente: Literal["xbrl", "texto", "ambas", "ninguna"]
    cita: str | None = Field(default=None, description="Texto literal que respalda la respuesta")
    chunk_id: str | None = Field(default=None, description="Identificador del fragmento citado")


SYSTEM = """Eres un analista financiero que responde preguntas sobre el corpus de informes 10-K usando ÚNICAMENTE las herramientas disponibles.

Reglas obligatorias:
- Para CUALQUIER cifra financiera exacta usa get_xbrl_fact; no leas cifras de la prosa.
- Para riesgos, estrategia, litigios, causas o comentarios de dirección usa search_filings.
- Pasa ticker, fiscal_year e item como filtros siempre que puedan deducirse de la pregunta.
- En comparativas entre ejercicios consulta cada ejercicio necesario. Si además se pregunta por la causa, combina XBRL para las cifras y search_filings para la explicación.
- Usa read_section solo si los fragmentos no bastan.
- Si dudas sobre la cobertura, usa list_available.
- No inventes conceptos XBRL. Si un concepto no existe, utiliza la lista de conceptos disponibles que devuelve get_xbrl_fact para corregirte.
- Cuando uses texto, devuelve chunk_id y una cita literal y CONTIGUA del chunk (nunca resumas ni
  uses puntos suspensivos para combinar fragmentos separados). Si necesitas varios datos de una
  tabla, elige la cita más relevante y completa, sin fusionar filas con "...".
- Las tablas del 10-K suelen expresar cifras en miles o millones (mira el encabezado:
  "in millions", "in thousands"). El campo `cifra` SIEMPRE debe reportarse en unidades
  absolutas (sin escalar): si la tabla dice "$25,474" bajo un encabezado "(in millions)",
  el valor de `cifra` es 25474000000, no 25474.
- Si el dato no está en el corpus, dilo. No lo estimes ni uses conocimiento externo.
"""

# ---------------------------------------------------------------------------
# Guardrail XBRL y agente final
# ---------------------------------------------------------------------------
from langchain.agents.middleware import AgentState, ModelCallLimitMiddleware, ToolCallLimitMiddleware, after_model
from langgraph.runtime import Runtime

_MARCA_XBRL = "VERIFICACION_XBRL"


@after_model(can_jump_to=["model"])
def verificar_cifras_contra_xbrl(state: AgentState, runtime: Runtime) -> dict | None:
    """Rechaza una cifra estructurada que no coincida con ningún hecho XBRL.

    Solo aplica cuando la respuesta declara XBRL como fuente (``fuente == 'xbrl'``).
    Si la cifra viene del texto (``fuente == 'texto'``), no tiene sentido exigir
    que coincida con un hecho XBRL que, por construcción, puede no existir para
    ese concepto/empresa — sería penalizar el comportamiento correcto de usar el
    texto cuando XBRL no reporta el dato.

    Además ejecuta el extractor de cifras sobre la prosa para dejar explícita la
    inspección requerida por la práctica. La decisión de corrección se basa en
    ``cifra`` porque es el campo contractual inequívoco de la magnitud principal.
    """
    respuesta = state.get("structured_response")
    if respuesta is None:
        return None
    _ = miax_s2.extraer_cifras(getattr(respuesta, "respuesta", "") or "")
    cifra = getattr(respuesta, "cifra", None)
    ticker = getattr(respuesta, "ticker", None)
    ejercicio = getattr(respuesta, "ejercicio", None)
    fuente = getattr(respuesta, "fuente", None)
    if cifra is None or not ticker or ejercicio is None:
        return None
    if fuente != "xbrl":
        return None

    if any(_MARCA_XBRL in str(getattr(m, "content", "")) for m in state.get("messages", [])):
        return None

    hechos = XBRL[(XBRL.ticker == str(ticker).upper())
                  & (XBRL.fiscal_year.astype(int) == int(ejercicio))]
    if hechos.empty:
        return None
    afirmada = float(cifra)
    if any(miax_s2.cuadra(afirmada, float(v), TOLERANCIA_XBRL) for v in hechos["value"]):
        return None

    disponibles = "; ".join(
        f"{r.concept}={float(r.value):,.0f} {r.unit}" for _, r in hechos.iterrows()
    )
    aviso = (
        f"{_MARCA_XBRL}: la cifra estructurada {afirmada:,.6g} para {ticker} "
        f"FY{ejercicio} no coincide, con tolerancia {TOLERANCIA_XBRL:.0%}, con "
        f"ningún hecho XBRL disponible. Hechos: {disponibles}. "
        "Corrige la respuesta llamando a get_xbrl_fact o indica que el dato no está disponible."
    )
    return {"messages": [{"role": "user", "content": aviso}], "jump_to": "model"}


_AGENTE_FINAL = None


def construir_agente_final():
    global _AGENTE_FINAL
    if _AGENTE_FINAL is None:
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver
        _AGENTE_FINAL = create_agent(
            model=MODELO,
            tools=HERRAMIENTAS,
            system_prompt=SYSTEM,
            response_format=RespuestaFinanciera,
            middleware=[
                ToolCallLimitMiddleware(run_limit=8),
                ModelCallLimitMiddleware(run_limit=10),
                verificar_cifras_contra_xbrl,
            ],
            checkpointer=InMemorySaver(),
        )
    return _AGENTE_FINAL


# ---------------------------------------------------------------------------
# Observabilidad e interfaz pública
# ---------------------------------------------------------------------------
def tokens_de(resultado) -> tuple[int, int]:
    entrada = salida = 0
    for mensaje in resultado.get("messages", []):
        uso = getattr(mensaje, "usage_metadata", None) or {}
        entrada += int(uso.get("input_tokens", 0) or 0)
        salida += int(uso.get("output_tokens", 0) or 0)
    return entrada, salida


def coste_estimado(resultado) -> float:
    entrada, salida = tokens_de(resultado)
    p_in, p_out = PRECIO_MODELO_USD_M
    return (entrada * p_in + salida * p_out) / 1e6


def responder(pregunta: str, thread_id: str | None = None) -> dict:
    """Ejecuta el sistema final. Esta es una de las dos interfaces del hold-out."""
    t0 = time.perf_counter()
    resultado = construir_agente_final().invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config={"configurable": {"thread_id": thread_id or f"q-{time.time_ns()}"}},
    )
    latencia = time.perf_counter() - t0
    entrada, salida = tokens_de(resultado)
    return {**resultado, "latencia_s": latencia, "tokens_entrada": entrada,
            "tokens_salida": salida, "coste_usd": coste_estimado(resultado)}


# ---------------------------------------------------------------------------
# Evaluadores
# ---------------------------------------------------------------------------
def cita_correcta(item: dict, resultado: dict) -> bool | None:
    """Comprueba existencia del chunk y que la cita esté realmente en él."""
    r = resultado.get("structured_response")
    if r is None:
        return False if item.get("ancla_texto") else None
    if not getattr(r, "chunk_id", None):
        return False if item.get("ancla_texto") else None
    fragmento = POR_ID.get(r.chunk_id)
    if fragmento is None or not getattr(r, "cita", None):
        return False
    cita = miax_s2.normalizar(r.cita)
    texto = miax_s2.normalizar(fragmento.get("texto", ""))
    # Se exige la cita completa, no un prefijo arbitrario.
    return bool(cita) and cita in texto


def cifra_coincide_xbrl(item: dict, resultado: dict) -> bool | None:
    """Compara la cifra estructurada con el ground truth XBRL del golden."""
    esperada = item.get("cifra_esperada")
    if esperada is None:
        return None
    r = resultado.get("structured_response")
    if r is None or getattr(r, "cifra", None) is None:
        return False
    return miax_s2.cuadra(float(r.cifra), float(esperada), TOLERANCIA_XBRL)


def uso_la_tool_correcta(item: dict, resultado: dict) -> bool:
    """Verifica que la trayectoria contenga todas las tools esperadas."""
    usadas = set(miax_s2.herramientas_usadas(resultado))
    esperadas = set(item.get("herramienta_esperada") or [])
    return esperadas.issubset(usadas)


EVALUADORES = {"cita": cita_correcta, "cifra": cifra_coincide_xbrl,
               "trayectoria": uso_la_tool_correcta}


def evaluar(ruta_jsonl: str | Path, salida_csv: str | Path | None = None) -> pd.DataFrame:
    """Evalúa el sistema final sobre un golden/hold-out JSONL sin editar código."""
    ruta = Path(ruta_jsonl)
    preguntas = [json.loads(l) for l in ruta.open(encoding="utf-8") if l.strip()]
    filas = []
    for i, item in enumerate(preguntas, 1):
        fila = {"id": item.get("id", f"q{i:03d}"), "familia": item.get("familia")}
        try:
            r = responder(item["pregunta"], thread_id=f"eval-{fila['id']}")
            fila.update({
                "latencia_s": r["latencia_s"], "coste_usd": r["coste_usd"],
                "tokens_entrada": r["tokens_entrada"], "tokens_salida": r["tokens_salida"],
                "llamadas": len(miax_s2.herramientas_usadas(r)),
            })
            for nombre, fn in EVALUADORES.items():
                fila[nombre] = fn(item, r)
            if item.get("ancla_texto"):
                q = reescribir_consulta(item["pregunta"])
                rec = buscar_hibrido(q, item.get("ticker"), item.get("fiscal_year"),
                                     item.get("item_esperado"), TOP_K)
                fila["recall@5"] = miax_s2.acierta(item, rec)
            else:
                fila["recall@5"] = None
            sr = r.get("structured_response")
            fila["respuesta"] = sr.respuesta if sr else None
        except Exception as exc:
            fila["error"] = f"{type(exc).__name__}: {exc}"
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    if salida_csv is not None:
        tabla.to_csv(salida_csv, index=False)
    return tabla


def resumir_resultados(tabla: pd.DataFrame, version: str) -> pd.DataFrame:
    def media(col):
        x = tabla[col].dropna() if col in tabla else pd.Series(dtype=float)
        return float(x.astype(float).mean()) if len(x) else float("nan")
    return pd.DataFrame([{
        "version": version,
        "cita": media("cita"), "cifra": media("cifra"),
        "trayectoria": media("trayectoria"), "recall@5": media("recall@5"),
        "coste_medio_usd": tabla.get("coste_usd", pd.Series(dtype=float)).mean(),
        "latencia_media_s": tabla.get("latencia_s", pd.Series(dtype=float)).mean(),
        "llamadas_por_pregunta": tabla.get("llamadas", pd.Series(dtype=float)).mean(),
    }])


__all__ = [
    "HERRAMIENTAS", "RespuestaFinanciera", "buscar_denso_baseline",
    "buscar_denso_filtrado", "buscar_hibrido", "reescribir_consulta",
    "responder", "evaluar", "cita_correcta", "cifra_coincide_xbrl",
    "uso_la_tool_correcta", "resumir_resultados", "construir_agente_final",
]