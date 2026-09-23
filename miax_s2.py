"""Módulo auxiliar de la sesión 2 — lo que se da hecho.

Se reparte junto a `S2_Robustez_y_Evaluacion_*.ipynb`. Aquí está **todo lo
que no es el contenido de la sesión**, por tres motivos distintos:

1. **El baseline de repuesto** (`baseline()`). El día 17 empieza ejecutando el
   agente de la semana anterior. Quien no lo traiga corriendo se quedaría
   fuera la sesión entera, así que hay uno montado: las cuatro herramientas
   del día 10 y `create_agent`. No es un atajo —quien lo use empieza el día
   sin conocer su propio código— pero es mejor que perder la clase.

2. **La fontanería del índice** (`cargar_indice`, `codificar`). Cargar FAISS y
   el codificador no enseña nada; lo que enseña es qué se hace con ellos, y
   eso está en el notebook, a la vista.

3. **Las primitivas que hay que dar para que la medición sea comparable**
   (`acierta`, `recall_en_k`, `montar_bm25`, `extraer_cifras`). Si cada grupo
   escribe su propia métrica, la tabla del informe deja de comparar nada.

## Lo que NO está aquí, y es deliberado

`buscar_denso()` no está. El día 10 era caja negra en `miax_s1.buscar`; hoy se
abre, y se abre escribiéndola en el notebook. Un `import` la volvería a cerrar.
"""

from __future__ import annotations

import functools
import json
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dónde está el corpus. Mismo orden que en `miax_s1`.
# ---------------------------------------------------------------------------
MODELO_EMBEDDINGS = "BAAI/bge-small-en-v1.5"
PREFIJO_CONSULTA_BGE = (
    "Represent this sentence for searching relevant passages: "
)

CANDIDATOS_CORPUS = [
    Path("corpus"),
    Path("/content/corpus")]


class CorpusNoEncontrado(RuntimeError):
    """El corpus no está montado. Se lanza con instrucciones, no a secas."""


def dir_corpus() -> Path:
    for candidato in CANDIDATOS_CORPUS:
        if (candidato / "chunks.jsonl").is_file():
            return candidato
    raise CorpusNoEncontrado(
        "No encuentro el corpus. Ejecuta la celda de setup (§0) antes de "
        f"esta. Buscado en: {[str(c) for c in CANDIDATOS_CORPUS]}"
    )


def _leer_jsonl(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


@functools.lru_cache(maxsize=1)
def cargar_corpus() -> tuple[list[dict], list[dict]]:
    """`(secciones, chunks)` tal y como salen del corpus."""
    base = dir_corpus()
    return (_leer_jsonl(base / "secciones.jsonl"),
            _leer_jsonl(base / "chunks.jsonl"))


# ---------------------------------------------------------------------------
# El índice: lo que el día 10 estaba dentro de la caja
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def cargar_indice():
    """`(indice, meta, codificador)`. Tarda unos segundos la primera vez.

    `meta` está alineado por posición con el índice: la fila *i* describe el
    vector *i*. Si eso se rompe, el retrieval devuelve el texto equivocado sin
    dar ningún error — por eso la comprobación de abajo no es decorativa.
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
            f"{len(meta)} filas. Vuelve a descomprimir los dos ZIP en la "
            f"misma carpeta."
        )
    return indice, meta, SentenceTransformer(MODELO_EMBEDDINGS)


def codificar(textos: list[str], es_consulta: bool = True):
    """Vectores normalizados, con el prefijo de BGE si son consultas.

    El prefijo va en la CONSULTA y no en los fragmentos indexados. Omitirlo no
    da ningún error: simplemente recupera peor. Es el fallo silencioso del que
    va media sesión.
    """
    _, _, codificador = cargar_indice()
    if es_consulta:
        textos = [PREFIJO_CONSULTA_BGE + t for t in textos]
    return codificador.encode(
        textos, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")


def fila_a_fragmento(fila, puntuacion: float) -> dict:
    """Una fila de `chunks_meta` en el dict que manejan las herramientas."""
    return {
        "chunk_id": fila["chunk_id"],
        "ticker": fila["ticker"],
        "fiscal_year": int(fila["fiscal_year"]),
        "item": fila["item"],
        "texto": fila["texto"],
        "n_tokens": int(fila["n_tokens"]),
        "contiene_tabla": bool(fila["contiene_tabla"]),
        "inicio_car": int(fila["inicio_car"]),
        "fin_car": int(fila["fin_car"]),
        "puntuacion": round(float(puntuacion), 4),
    }


def formatear_fragmentos(fragmentos: list[dict]) -> str:
    """Los fragmentos, en el texto que ve el modelo."""
    if not fragmentos:
        return ("Sin resultados para esa consulta con esos filtros. "
                "Prueba a quitar algún filtro o a reformular la búsqueda.")
    return "\n\n---\n\n".join(
        f"[{f['chunk_id']}] {f['ticker']} FY{f['fiscal_year']} "
        f"Item {f['item']} (similitud {f['puntuacion']:.3f})\n{f['texto']}"
        for f in fragmentos
    )


# ---------------------------------------------------------------------------
# BM25: se da montado. El ejercicio es la fusión, no el índice léxico.
# ---------------------------------------------------------------------------
_PALABRAS = re.compile(r"[a-z0-9$%.]+")


def tokenizar(texto: str) -> list[str]:
    """Minúsculas y palabras. Conserva `$`, `%` y los puntos decimales.

    No es un detalle: el caso en el que BM25 gana al embedding son los
    tickers, los años y las cifras. Un tokenizador que se coma el `$` y los
    decimales tira justo la señal que se venía a buscar.
    """
    return _PALABRAS.findall(texto.lower())


@functools.lru_cache(maxsize=1)
def montar_bm25():
    """`(bm25, chunks)` con los 1.749 fragmentos del troceado base."""
    from rank_bm25 import BM25Okapi

    _, chunks = cargar_corpus()
    return BM25Okapi([tokenizar(c["texto"]) for c in chunks]), chunks


# ---------------------------------------------------------------------------
# La métrica: recall@k anclado al texto, no al chunk_id
# ---------------------------------------------------------------------------
# Es una copia de `source/data_source/anclas.py`, reducida a lo que hace falta
# en clase. Se copia y no se importa porque `anclas.py` no se reparte: vive en
# el pipeline de preparación del corpus, que los alumnos no tienen.
_ESPACIOS = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    return _ESPACIOS.sub(" ", texto).strip().lower()


def acierta(item_golden: dict, recuperados: list[dict]) -> bool:
    """¿Alguno de los fragmentos recuperados contiene el ancla entera?

    Es la primitiva de `recall@k`, y **no depende del troceado**: funciona
    igual con el troceado base y con el que escribáis vosotros. Por eso la
    verdad del golden set es una frase del informe y no un `chunk_id`: en
    cuanto cambiéis la ventana o el solape, todos los `chunk_id` son otros.

    Un fragmento del ejercicio equivocado NO cuenta, aunque contenga el
    ancla. Los 10-K repiten factores de riesgo palabra por palabra de un año
    para otro: sin esta comprobación, recuperar el FY2024 puntuaría como si
    hubierais encontrado el FY2025.
    """
    ancla = item_golden.get("ancla_texto")
    if not ancla:
        return False
    objetivo = normalizar(ancla)
    inicio, fin = item_golden.get("ancla_inicio"), item_golden.get("ancla_fin")

    for fragmento in recuperados:
        # Si el fragmento no trae metadatos de documento no hay con qué
        # comprobarlo, y se acepta la coincidencia de texto: es lo que
        # mantiene la promesa de no exigirle nada a vuestro troceador.
        sin_metadatos = (fragmento.get("ticker") is None
                         and fragmento.get("fiscal_year") is None)
        mismo_documento = (
            fragmento.get("ticker") == item_golden.get("ticker")
            and int(fragmento.get("fiscal_year", -1))
            == int(item_golden.get("fiscal_year", -2))
        )
        if not (sin_metadatos or mismo_documento):
            continue
        # 1. Por tramo, si el fragmento lo trae. Exacto.
        if (inicio is not None
                and fragmento.get("inicio_car") is not None
                and fragmento.get("item") == item_golden.get("item_esperado")
                and fragmento["inicio_car"] <= inicio
                and fragmento["fin_car"] >= fin):
            return True
        # 2. Por texto normalizado. No le exige nada a vuestro troceador.
        if objetivo in normalizar(fragmento.get("texto", "")):
            return True
    return False


def recall_en_k(golden: list[dict], recuperados_por_id: dict) -> float:
    """recall@k sobre los ítems del golden set que llevan ancla."""
    con_ancla = [g for g in golden if g.get("ancla_texto")]
    if not con_ancla:
        return 0.0
    return sum(acierta(g, recuperados_por_id.get(g["id"], []))
               for g in con_ancla) / len(con_ancla)


def posicion_del_ancla(item_golden: dict, ordenados: list[dict]) -> int | None:
    """En qué puesto aparece el primer fragmento que contiene el ancla.

    `recall@5` dice sí o no; esto dice por cuánto. Un ancla en el puesto 7 y
    otra en el 1.400 fallan las dos, y no son el mismo problema.
    """
    for posicion, fragmento in enumerate(ordenados, 1):
        if acierta(item_golden, [fragmento]):
            return posicion
    return None


# ---------------------------------------------------------------------------
# Las cinco preguntas duras del diagnóstico en frío
# ---------------------------------------------------------------------------
# Una por cada fila de la tabla de fallos, y las cinco salen del golden set
# oficial: así el diagnóstico de las 0:00 y la evaluación de las 2:00 miden lo
# mismo y la mejora se puede enseñar de punta a punta.
PREGUNTAS_DURAS: list[dict] = [
    {
        "id": "of-006",
        "fallo": "No encontró nada relevante",
        "por_que": "La respuesta está en el Item 7A, que son 37 fragmentos "
                   "de 1.749. Nadie escribe «7A» en su pregunta, y sin "
                   "filtro esos 37 compiten con todo lo demás.",
    },
    {
        "id": "of-002",
        "fallo": "Recuperó lo que no era",
        "por_que": "Microsoft repite sus factores de riesgo casi enteros "
                   "entre FY2024 y FY2025. Sin filtrar por ejercicio, lo "
                   "más parecido a la pregunta puede ser el año que no es.",
    },
    {
        "id": "of-020",
        "fallo": "No supo comparar dos ejercicios",
        "por_que": "Hay que consultar dos cifras y leer la explicación de "
                   "la diferencia. Una sola pasada de recuperación no lo "
                   "hace, y aquí además el beneficio neto BAJA mientras el "
                   "negocio crece.",
    },
    {
        "id": "of-012",
        "fallo": "Se inventó la cifra",
        "por_que": "Alphabet etiqueta `Revenues` y no "
                   "`RevenueFromContractWithCustomerExcludingAssessedTax` "
                   "en FY2025. El modelo generaliza el concepto de otra "
                   "compañía, no lo encuentra, y rellena con lo que le "
                   "suena.",
    },
    {
        "id": "of-019",
        "fallo": "Usó la herramienta equivocada",
        "por_que": "El beneficio bruto de Apple está en el XBRL y también "
                   "comentado en el MD&A. Si lo lee de la prosa acierta hoy "
                   "y falla el día que la tabla venga partida.",
    },
]


# ---------------------------------------------------------------------------
# Reescrituras de respaldo — para que el arreglo 3 se pueda medir sin clave
# ---------------------------------------------------------------------------
# El arreglo 3 reescribe la consulta con el LLM. Si OpenRouter no responde,
# la comparación de los cuatro recall@k se quedaría coja justo en la fila que
# más mueve la aguja, así que aquí está la reescritura grabada de las trece
# preguntas con ancla. Es el mismo criterio que `demo_traza.json` en la
# sesión 1: la clase no se para porque falle una red.
#
# Son traducciones al inglés con los términos del informe, que es exactamente
# lo que hay que pedirle al modelo que haga.
REESCRITURAS_RESPALDO: dict[str, str] = {
    "of-001": "competition in the China market limited by export controls",
    "of-002": "generative AI models in internal systems create new attack "
              "surfaces for adversaries",
    "of-003": "legal bases for data transfers from the European Union to the "
              "United States",
    "of-004": "new tariffs announced on imports to the United States in 2025",
    "of-005": "DOJ and state Attorneys General antitrust lawsuit on Search "
              "advertising practices",
    "of-006": "foreign exchange risk International segment percentage of "
              "consolidated revenues",
    "of-014": "Microsoft Cloud revenue growth highlights fiscal year 2025 "
              "compared with 2024",
    "of-015": "increased demand for Data Center systems accelerated computing "
              "generative AI",
    "of-016": "research and development expense increase employee "
              "compensation infrastructure AI initiatives",
    "of-017": "operating income expected first quarter guidance",
    "of-018": "capital expenditures spent on technical infrastructure 2024 "
              "and 2025",
    "of-019": "products gross margin percentage decreased tariff costs mix of "
              "products",
    "of-020": "provision for income taxes increased effective tax rate",
}


def preguntas_duras(golden: list[dict]) -> list[dict]:
    """Las cinco, ya emparejadas con su ítem del golden set."""
    por_id = {g["id"]: g for g in golden}
    faltan = [d["id"] for d in PREGUNTAS_DURAS if d["id"] not in por_id]
    if faltan:
        raise KeyError(
            f"El golden set cargado no trae {faltan}. ¿Estáis usando "
            f"`golden_set_ejemplo.jsonl` en lugar del oficial?"
        )
    return [{**d, **por_id[d["id"]]} for d in PREGUNTAS_DURAS]


# ---------------------------------------------------------------------------
# El extractor de cifras del ejercicio de middleware
# ---------------------------------------------------------------------------
_MULTIPLICADOR = {
    "billion": 1e9, "billones": 1e9, "mil millones": 1e9,
    "million": 1e6, "millones": 1e6, "millón": 1e6,
    "thousand": 1e3, "miles": 1e3,
}
_CIFRA = re.compile(
    r"(\d[\d.,]*)\s*(billion|billones|mil millones|million|millones|millón|"
    r"thousand|miles)?",
    re.IGNORECASE,
)


def extraer_cifras(texto: str) -> list[float]:
    """Los números que afirma un texto, en unidades absolutas.

    Se da hecho porque parsear «281.724 millones de dólares» en español y en
    inglés es un rato de expresiones regulares y no es la lección. La lección
    es qué se hace con el número una vez extraído.

    Asume la convención española —punto de millares, coma decimal— cuando el
    número la usa de forma inequívoca; si no, la inglesa.
    """
    encontrados: list[float] = []
    for crudo, sufijo in _CIFRA.findall(texto):
        limpio = crudo.rstrip(".,")
        if not any(c.isdigit() for c in limpio):
            continue
        if "," in limpio and "." in limpio:
            # El separador que va más a la derecha es el decimal.
            if limpio.rfind(",") > limpio.rfind("."):
                limpio = limpio.replace(".", "").replace(",", ".")
            else:
                limpio = limpio.replace(",", "")
        elif "," in limpio:
            entera, _, decimal = limpio.rpartition(",")
            limpio = (f"{entera.replace(',', '')}.{decimal}"
                      if len(decimal) != 3 else limpio.replace(",", ""))
        elif "." in limpio:
            entera, _, decimal = limpio.rpartition(".")
            if len(decimal) == 3 and entera:
                limpio = limpio.replace(".", "")
        try:
            valor = float(limpio)
        except ValueError:
            continue
        encontrados.append(valor * _MULTIPLICADOR.get(sufijo.lower(), 1.0)
                           if sufijo else valor)
    return encontrados


def cuadra(afirmada: float, real: float, tolerancia: float = 0.01) -> bool:
    """¿La cifra afirmada coincide con la del XBRL, con tolerancia relativa?

    La tolerancia existe porque redondear a «281.700 millones» no es
    inventarse un número. Inventárselo es decir 250.000.
    """
    if real == 0:
        return afirmada == 0
    return abs(afirmada - real) / abs(real) <= tolerancia


# ---------------------------------------------------------------------------
# Ver la trayectoria
# ---------------------------------------------------------------------------
def pretty_trace(resultado, max_chars: int = 220) -> None:
    """Qué herramientas se llamaron, con qué argumentos y qué devolvieron.

    La misma de la sesión 1. Sin trayectoria no se distingue una respuesta
    correcta de una respuesta correcta por casualidad, que es justo lo que
    comprueba el evaluador `uso_la_tool_correcta`.
    """
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    paso = 0
    for mensaje in mensajes:
        for llamada in getattr(mensaje, "tool_calls", None) or []:
            paso += 1
            args = ", ".join(f"{k}={v!r}" for k, v in llamada["args"].items())
            print(f"  {paso}. {llamada['name']}({args})")
        if type(mensaje).__name__ == "ToolMessage":
            contenido = str(mensaje.content).replace("\n", " ")
            print(f"       -> {contenido[:max_chars]}"
                  f"{'…' if len(contenido) > max_chars else ''}")
    if isinstance(resultado, dict) and resultado.get("structured_response"):
        r = resultado["structured_response"]
        print(f"\n  respuesta: {r.respuesta}")
        print(f"  fuente: {r.fuente} · cita: {r.chunk_id}")


def herramientas_usadas(resultado) -> list[str]:
    """Los nombres de las herramientas que aparecen en la trayectoria."""
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    return [llamada["name"]
            for mensaje in mensajes
            for llamada in (getattr(mensaje, "tool_calls", None) or [])]


# ---------------------------------------------------------------------------
# Coste y latencia: dos columnas de la tabla, no una nota al pie
# ---------------------------------------------------------------------------
# USD por millón de tokens (entrada, salida). Consultado el 2/09/2026 en
# https://openrouter.ai/api/v1/models. REVISAR LA VÍSPERA.
PRECIOS_OPENROUTER = {
    "google/gemini-3.5-flash-lite":  (0.30,  2.50),
    "google/gemini-3.8-flash":       (0.75,  3.75),
    "anthropic/claude-opus-5":       (5.00, 25.00),
    "anthropic/claude-fable-5.1":   (10.00, 50.00),
}


def tokens_de(resultado) -> tuple[int, int]:
    """(entrada, salida) sumando el uso reportado por todos los mensajes."""
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    entrada = salida = 0
    for mensaje in mensajes:
        uso = getattr(mensaje, "usage_metadata", None) or {}
        entrada += uso.get("input_tokens", 0) or 0
        salida += uso.get("output_tokens", 0) or 0
    return entrada, salida


def coste_de(resultado, modelo: str) -> float:
    """Coste en USD de una invocación, según `PRECIOS_OPENROUTER`."""
    nombre = modelo.split(":", 1)[-1]
    if nombre not in PRECIOS_OPENROUTER:
        return 0.0
    precio_entrada, precio_salida = PRECIOS_OPENROUTER[nombre]
    entrada, salida = tokens_de(resultado)
    return (entrada * precio_entrada + salida * precio_salida) / 1e6


def cronometrar(funcion, *args, **kwargs) -> tuple:
    """`(resultado, segundos)`."""
    comienzo = time.perf_counter()
    resultado = funcion(*args, **kwargs)
    return resultado, time.perf_counter() - comienzo


# ---------------------------------------------------------------------------
# El baseline de repuesto — las cuatro herramientas del día 10
# ---------------------------------------------------------------------------
def construir_herramientas() -> list:
    """Las cuatro herramientas de la sesión 1, tal y como quedaron.

    Están aquí para quien no traiga las suyas. Si traéis las vuestras, usad
    las vuestras: el día 24 se defiende vuestro código, no este.
    """
    import pandas as pd
    from langchain.tools import tool

    base = dir_corpus()
    secciones, _ = cargar_corpus()
    xbrl = pd.read_parquet(base / "xbrl_facts.parquet")

    @tool
    def list_available() -> str:
        """Devuelve qué compañías, ejercicios y secciones hay en el corpus.

        Úsala SIEMPRE antes de responder si no estás seguro de que la
        compañía o el ejercicio por los que te preguntan existen. El corpus
        es limitado: si algo no está aquí, no está.
        """
        por_ticker: dict[str, dict] = {}
        for s in secciones:
            entrada = por_ticker.setdefault(
                s["ticker"], {"empresa": s["empresa"], "ejercicios": set(),
                              "items": set()})
            entrada["ejercicios"].add(s["fiscal_year"])
            entrada["items"].add(s["item"])
        return "\n".join(
            f"{t} ({d['empresa']}): ejercicios "
            f"{sorted(d['ejercicios'])}, items {sorted(d['items'])}"
            for t, d in sorted(por_ticker.items())
        )

    @tool
    def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
        """Devuelve el valor EXACTO de una magnitud financiera tal y como la
        compañía la reportó en XBRL.

        Es la fuente autorizada para cualquier cifra. Úsala SIEMPRE en lugar
        de leer un número del texto del informe.

        Args:
            ticker: Símbolo bursátil, p. ej. 'NVDA'.
            fiscal_year: Ejercicio fiscal reportado, p. ej. 2024.
            concept: Concepto en taxonomía US-GAAP, p. ej. 'Revenues',
                'NetIncomeLoss', 'Assets', 'OperatingIncomeLoss'.

        Devuelve el valor con su unidad y fecha de cierre, o un aviso
        explícito si la compañía no reportó ese concepto en ese ejercicio.
        """
        fila = xbrl[(xbrl.ticker == ticker)
                    & (xbrl.fiscal_year == int(fiscal_year))
                    & (xbrl.concept == concept)]
        if fila.empty:
            hay = sorted(xbrl[(xbrl.ticker == ticker)
                              & (xbrl.fiscal_year == int(fiscal_year))]
                         .concept.unique())
            return (f"{ticker} no reportó '{concept}' en FY{fiscal_year}. "
                    f"Conceptos disponibles: {hay or 'ninguno'}.")
        f = fila.iloc[0]
        return (f"{ticker} FY{fiscal_year} {concept} = {f['value']:,.0f} "
                f"{f['unit']} (cierre {f['period_end']}, {f['form']})")

    @tool
    def search_filings(query: str, ticker: str | None = None,
                       fiscal_year: int | None = None,
                       item: str | None = None, k: int = 5) -> str:
        """Busca fragmentos de texto relevantes en los informes 10-K.

        Úsala para preguntas cualitativas: riesgos, estrategia, litigios,
        comentarios de la dirección. NO la uses para obtener cifras: para eso
        está get_xbrl_fact.

        Args:
            query: Qué buscar, en lenguaje natural y en inglés.
            ticker: Filtra por compañía si la pregunta la menciona.
            fiscal_year: Filtra por ejercicio si la pregunta lo menciona.
            item: Filtra por sección: '1A' riesgos, '7' MD&A,
                '7A' riesgo de mercado, '8' estados financieros.
            k: Número de fragmentos a devolver.

        Devuelve k fragmentos, cada uno con su chunk_id para poder citarlo.
        """
        return formatear_fragmentos(
            buscar_con_filtros(query, ticker, fiscal_year, item, k)
        )

    @tool
    def read_section(ticker: str, fiscal_year: int, item: str) -> str:
        """Devuelve el TEXTO COMPLETO de una sección de un 10-K.

        Es una herramienta CARA: puede devolver decenas de miles de tokens.
        Úsala solo cuando search_filings devuelva fragmentos insuficientes y
        necesites el contexto entero de una sección concreta.
        """
        for s in secciones:
            if (s["ticker"] == ticker and s["fiscal_year"] == int(fiscal_year)
                    and s["item"] == item):
                return s["texto"]
        return (f"No hay sección {item} de {ticker} FY{fiscal_year} en el "
                f"corpus. Usa list_available para ver qué hay.")

    return [list_available, get_xbrl_fact, search_filings, read_section]


def buscar_con_filtros(query: str, ticker=None, fiscal_year=None,
                       item=None, k: int = 5) -> list[dict]:
    """Búsqueda densa con filtro de metadatos. El arreglo 1, ya montado.

    Está aquí para que `construir_herramientas()` funcione sin depender de lo
    que escribáis en el notebook. En clase lo escribís vosotros: comparar lo
    vuestro con esto es parte del ejercicio.
    """
    indice, meta, _ = cargar_indice()
    mascara = meta.index
    if ticker:
        mascara = mascara.intersection(meta.index[meta["ticker"] == ticker])
    if fiscal_year:
        mascara = mascara.intersection(
            meta.index[meta["fiscal_year"].astype(int) == int(fiscal_year)])
    if item:
        mascara = mascara.intersection(meta.index[meta["item"] == item])

    puntuaciones, posiciones = indice.search(codificar([query]), indice.ntotal)
    permitidas = set(int(i) for i in mascara)
    salida = []
    for puntuacion, posicion in zip(puntuaciones[0], posiciones[0]):
        if int(posicion) not in permitidas:
            continue
        salida.append(fila_a_fragmento(meta.iloc[int(posicion)], puntuacion))
        if len(salida) >= k:
            break
    return salida


SYSTEM = """Eres un analista financiero que responde preguntas sobre informes
10-K usando ÚNICAMENTE las herramientas disponibles.

Reglas:
- Para cualquier CIFRA, usa get_xbrl_fact. Nunca leas un número de la prosa.
- Para riesgos, estrategia o comentarios de la dirección, usa search_filings.
- Si no sabes si una compañía o un ejercicio están en el corpus, empieza por
  list_available.
- El corpus está en inglés: escribe las consultas de búsqueda en inglés.
- Cita el chunk_id del fragmento en el que te apoyes.
- Si el dato no está en el corpus, dilo. No lo estimes.
"""


def esquema_respuesta():
    """`RespuestaFinanciera`, el contrato §7 del enunciado."""
    from typing import Literal

    from pydantic import BaseModel, Field

    class RespuestaFinanciera(BaseModel):
        """Respuesta trazable a una pregunta sobre informes 10-K."""

        respuesta: str = Field(
            description="Respuesta en prosa, breve y directa")
        cifra: float | None = Field(
            default=None, description="Valor numérico, si la pregunta pide uno")
        unidad: str | None = Field(
            default=None, description="USD, shares, porcentaje…")
        ticker: str | None = None
        ejercicio: int | None = None
        fuente: Literal["xbrl", "texto", "ambas", "ninguna"] = Field(
            description="De dónde sale el dato. 'ninguna' si no está en el "
                        "corpus")
        cita: str | None = Field(
            default=None,
            description="Texto literal del informe que respalda la respuesta")
        chunk_id: str | None = Field(
            default=None,
            description="Identificador del fragmento citado, para verificar")

    return RespuestaFinanciera


def baseline(modelo: str = "openrouter:google/gemini-3.8-flash",
             middleware: list | None = None):
    """El agente del día 10, montado. Para quien no traiga el suyo.

    Devuelve el agente de `create_agent` con las cuatro herramientas, el
    esquema de respuesta y `InMemorySaver`. Si traéis el vuestro, usad el
    vuestro.
    """
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    return create_agent(
        model=modelo,
        tools=construir_herramientas(),
        system_prompt=SYSTEM,
        response_format=esquema_respuesta(),
        middleware=middleware or [],
        checkpointer=InMemorySaver(),
    )
