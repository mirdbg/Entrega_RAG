"""Módulo auxiliar de la sesión 1 — lo que hoy es caja negra.

Se reparte junto a `S1_Herramientas_y_Bucle_*.ipynb`. Contiene dos cosas, y
las dos están aquí por el mismo motivo: **no son el contenido de la sesión 1**.

1. `buscar()` — la búsqueda densa sobre el índice FAISS que hay debajo de la
   herramienta `search_filings`. Es caja negra a propósito hasta el día 17,
   que es cuando se abre y se cuestiona (troceado, embeddings, `IndexFlatIP`,
   top-k, y los tres arreglos que se miden con `recall@k`).

2. `demo_apertura()` — la demo del minuto uno. Reproduce una ejecución del
   agente terminado, grabada en `demo_traza.json`, para enseñar el destino
   antes de construirlo. **No depende ni de la red ni de que el corpus esté
   montado**: en el minuto uno de clase todavía no se ha ejecutado la celda de
   instalación ni la de setup.

## Lo único que hay que saber de `buscar()`

El prefijo de consulta de BGE. Los modelos BGE piden un prefijo en la
**consulta** y no en los fragmentos indexados. Omitirlo no da ningún error:
simplemente recupera peor. Está documentado en `indice/MANIFEST.md` y es la
clase de fallo silencioso que la sesión 2 enseña a detectar.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes del índice. Tienen que coincidir con `indice/MANIFEST.md`.
# ---------------------------------------------------------------------------
MODELO_EMBEDDINGS = "BAAI/bge-small-en-v1.5"
PREFIJO_CONSULTA_BGE = (
    "Represent this sentence for searching relevant passages: "
)

# Dónde puede estar el corpus ya descomprimido, en el orden en que se mira.
# `corpus/` es donde lo deja la celda de setup; el resto son los sitios donde
# aparece cuando el notebook se ejecuta fuera de Colab.
CANDIDATOS_CORPUS = [
    Path("corpus"),
    Path("/content/corpus"),
]

RUTA_TRAZA_DEMO = Path(__file__).resolve().parent / "demo_traza.json"


class CorpusNoEncontrado(RuntimeError):
    """El corpus no está montado. Se lanza con instrucciones, no a secas."""


def dir_corpus() -> Path:
    """La carpeta del corpus descomprimido, o un error que dice qué hacer."""
    for candidato in CANDIDATOS_CORPUS:
        if (candidato / "chunks.jsonl").is_file():
            return candidato
    raise CorpusNoEncontrado(
        "No encuentro el corpus. Ejecuta la celda de setup (§1) antes de "
        f"esta. Buscado en: {[str(c) for c in CANDIDATOS_CORPUS]}"
    )


# ---------------------------------------------------------------------------
# Búsqueda densa — el cuerpo de `search_filings`
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _indice():
    """Índice, metadatos y codificador. Se cargan una sola vez.

    El modelo de embeddings tarda unos segundos en cargar la primera vez y
    baja unos 130 MB si no está en la caché de Colab. Por eso va aquí dentro
    y no en el import: importar el módulo tiene que ser instantáneo.
    """
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    base = dir_corpus() / "indice"
    indice = faiss.read_index(str(base / "corpus.faiss"))
    meta = pd.read_parquet(base / "chunks_meta.parquet")

    if indice.ntotal != len(meta):
        raise RuntimeError(
            f"El índice tiene {indice.ntotal} vectores y los metadatos "
            f"{len(meta)} filas. Están desalineados: vuelve a descomprimir "
            "los dos ZIP en la misma carpeta."
        )

    codificador = SentenceTransformer(MODELO_EMBEDDINGS)
    return indice, meta, codificador


def buscar(
    query: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = 5,
) -> list[dict]:
    """Los `k` fragmentos más parecidos a `query`, con sus metadatos.

    Cada resultado es un dict con `chunk_id`, `ticker`, `fiscal_year`, `item`,
    `texto`, `n_tokens`, `contiene_tabla` y `puntuacion` (similitud coseno,
    entre -1 y 1: los vectores están normalizados y el índice usa producto
    interno).

    Los filtros se aplican **después** de la búsqueda, sobre el orden que
    devuelve el índice. Con 1.749 vectores eso es instantáneo y no cambia el
    resultado; con un corpus grande habría que filtrar antes, y esa es una de
    las conversaciones del día 17.
    """
    indice, meta, codificador = _indice()

    vector = codificador.encode(
        [PREFIJO_CONSULTA_BGE + query],   # el prefijo, SOLO en la consulta
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    puntuaciones, posiciones = indice.search(vector, indice.ntotal)

    resultados: list[dict] = []
    for puntuacion, posicion in zip(puntuaciones[0], posiciones[0]):
        fila = meta.iloc[int(posicion)]
        # Ojo: `fila.item` devuelve el método `Series.item`, no la columna.
        # Con corchetes siempre.
        if ticker and fila["ticker"] != ticker:
            continue
        if fiscal_year and int(fila["fiscal_year"]) != int(fiscal_year):
            continue
        if item and fila["item"] != item:
            continue
        resultados.append(
            {
                "chunk_id": fila["chunk_id"],
                "ticker": fila["ticker"],
                "fiscal_year": int(fila["fiscal_year"]),
                "item": fila["item"],
                "texto": fila["texto"],
                "n_tokens": int(fila["n_tokens"]),
                "contiene_tabla": bool(fila["contiene_tabla"]),
                "puntuacion": round(float(puntuacion), 4),
            }
        )
        if len(resultados) >= k:
            break
    return resultados


def formatear_fragmentos(fragmentos: list[dict]) -> str:
    """Los fragmentos, en el texto que ve el modelo.

    Cada uno lleva su `chunk_id` delante: es lo que permite que el agente cite
    y que el evaluador `cita_correcta` compruebe la cita.
    """
    if not fragmentos:
        return ("Sin resultados para esa consulta con esos filtros. "
                "Prueba a quitar algún filtro o a reformular la búsqueda.")
    partes = []
    for f in fragmentos:
        partes.append(
            f"[{f['chunk_id']}] {f['ticker']} FY{f['fiscal_year']} "
            f"Item {f['item']} (similitud {f['puntuacion']:.3f})\n{f['texto']}"
        )
    return "\n\n---\n\n".join(partes)


# ---------------------------------------------------------------------------
# La demo del minuto uno
# ---------------------------------------------------------------------------
def _imprimir_paso(n: int, paso: dict) -> None:
    if paso["tipo"] == "herramienta":
        args = ", ".join(f"{k}={v!r}" for k, v in paso["argumentos"].items())
        print(f"  {n}. {paso['herramienta']}({args})")
        utiles = [x for x in paso["resultado"].splitlines() if x.strip()]
        for linea in utiles[:2]:
            print(f"       -> {linea[:100]}")
    elif paso["tipo"] == "razonamiento":
        print(f"  {n}. (el modelo decide) {paso['texto']}")


def demo_apertura(agente=None, pregunta: str | None = None) -> dict:
    """Enseña el agente terminado antes de construirlo.

    Sin argumentos reproduce la ejecución grabada en `demo_traza.json`, que es
    lo que hace falta en el minuto uno: ahí todavía no hay ni LangChain
    instalado ni corpus montado.

    Si se le pasa un `agente` ya construido —al final de la sesión, con el
    agente de §6— ejecuta la pregunta de verdad y, si algo falla, cae en la
    grabación en vez de romper la clase.
    """
    grabacion = json.loads(RUTA_TRAZA_DEMO.read_text(encoding="utf-8"))
    pregunta = pregunta or grabacion["pregunta"]

    if agente is not None:
        try:
            resultado = agente.invoke(
                {"messages": [{"role": "user", "content": pregunta}]},
                config={"configurable": {"thread_id": "demo-apertura"}},
            )
            print(f"PREGUNTA: {pregunta}\n")
            print("(ejecución en vivo)\n")
            print(resultado["structured_response"].respuesta)
            return resultado
        except Exception as e:                      # la clase no se para
            print(f"No se pudo ejecutar en vivo ({type(e).__name__}: {e}).")
            print("Reproduzco la ejecución grabada.\n")

    print(f"PREGUNTA: {pregunta}\n")
    print("TRAYECTORIA")
    for i, paso in enumerate(grabacion["trayectoria"], 1):
        _imprimir_paso(i, paso)
    print("\nRESPUESTA")
    print(grabacion["respuesta"]["respuesta"])
    print(f"\nfuente: {grabacion['respuesta']['fuente']}"
          f" · cita: {grabacion['respuesta']['chunk_id']}")

    meta = grabacion["metadatos"]
    latencia = (f" · {meta['latencia_s']:.1f} s"
                if meta.get("latencia_s") else "")
    print(f"\n[grabación del {meta['fecha']} · modelo {meta['modelo']}"
          f" · {meta['llamadas_herramienta']} llamadas a herramienta"
          f"{latencia}]")
    if meta.get("origen") != "ejecucion_real":
        print("[AVISO: traza PROVISIONAL. Las salidas de herramienta son "
              "reales, la prosa final es de ejemplo. Regenérala con "
              "`python generar_traza_demo.py` en cuanto haya clave.]")
    return grabacion
