"""Terminología del artículo para el texto visible en español: 'conjunto(s) de evaluación' (no 'estrato') y 'establecido/a' (no 'congelado/a').
Solo cambia palabras completas en español; no toca identificadores (p. ej., bloques_estratos, stratum, CONFIGURACION_CONGELADA)."""
import io, re, tokenize
_R = [(r"(?<![\w_<])Estratos(?![\w_])", "Conjuntos de evaluación"), (r"(?<![\w_<])estratos(?![\w_])", "conjuntos de evaluación"),
      (r"(?<![\w_<])Estrato(?![\w_])", "Conjunto de evaluación"), (r"(?<![\w_<])estrato(?![\w_])", "conjunto de evaluación"),
      (r"(?<![\w_])congelados(?![\w_])", "establecidos"), (r"(?<![\w_])congeladas(?![\w_])", "establecidas"),
      (r"(?<![\w_])congelado(?![\w_])", "establecido"), (r"(?<![\w_])congelada(?![\w_])", "establecida")]
def texto(s: str) -> str:
    for a, b in _R: s = re.sub(a, b, s)
    return s
def codigo(src: str) -> str:
    """Aplica texto() solo a cadenas y comentarios de código Python (no a identificadores)."""
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    lines = src.splitlines(keepends=True); offs = [0]
    for l in lines: offs.append(offs[-1] + len(l))
    out, last = [], 0
    kinds = {tokenize.STRING, tokenize.COMMENT} | ({tokenize.FSTRING_MIDDLE} if hasattr(tokenize, "FSTRING_MIDDLE") else set())
    for t in toks:
        if t.type in kinds:
            a = offs[t.start[0] - 1] + t.start[1]; b = offs[t.end[0] - 1] + t.end[1]
            out.append(src[last:a]); out.append(texto(src[a:b])); last = b
    out.append(src[last:]); return "".join(out)
