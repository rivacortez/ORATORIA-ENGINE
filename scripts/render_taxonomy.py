"""Render the annotation manual from the taxonomy in code.

The taxonomy has exactly one source of truth: ``domain.shared.taxonomy``. That
module is what the allowlist checks against, so if a Markdown manual were
maintained by hand it would be the copy that drifts - and it is the copy the
annotators read. An annotator working from a stale definition produces
disagreement that looks like a hard boundary case and is actually a
documentation bug, which is the most expensive kind to find: it surfaces as a
low inter-annotator agreement score weeks later.

So the manual is generated, and ``tests/unit/test_taxonomy_doc_is_current.py``
fails when the checked-in file no longer matches. Regenerate with:

    uv run python scripts/render_taxonomy.py

That also makes this module the only place a scope qualification can be written.
The manual publishes both halves of the taxonomy and only the speech half is
being validated, so it has to say so - and a correction typed into the Markdown
survives exactly until the next regeneration, which is the worst possible
lifetime for a warning: long enough to be reviewed and approved, short enough to
be gone before anyone reads it. `tests/corpus/test_scope.py` sweeps the rendered
file for the phrasings that must not appear in it.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

from evidence_engine.domain.shared.taxonomy import (
    PROHIBITED_CONCEPTS,
    SPEECH_EVENT_DEFINITIONS,
    TAXONOMY_VERSION,
    VISUAL_EVENT_DEFINITIONS,
    EventDefinition,
    ProsodicIndicator,
    SpeechEventType,
    VisualEventType,
)
from evidence_engine.domain.speech_events.prosody import INDICATOR_UNITS

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "taxonomy" / "ANNOTATION_MANUAL.md"

#: Class counts, taken from the enums rather than spelled out. A manual that
#: states a number the taxonomy is free to change is a manual that starts lying
#: on the day somebody adds a class, and this is the copy the annotators read.
#: Short names because they are interpolated mid-sentence and the surrounding
#: prose is wrapped for the rendered file, not for this one.
_SPEECH = len(SpeechEventType)
_VISUAL = len(VisualEventType)
_ALL = _SPEECH + _VISUAL

# The quotation in the scope section below has to stay on a single line.
# `tests/corpus/test_scope.py` strips quoted spans before sweeping a document
# for closure phrasings that must not appear, and its regex does not cross a
# newline - so a quotation wrapped across two lines reads to the sweep as the
# assertion it is warning against, and the file that carries the warning fails
# the check the warning exists to satisfy.
HEADER = f"""<!--
GENERATED FILE - do not edit by hand.

Source: src/evidence_engine/domain/shared/taxonomy.py
Regenerate: uv run python scripts/render_taxonomy.py
Guarded by: tests/unit/test_taxonomy_doc_is_current.py
-->

# Manual de anotacion - Taxonomia de eventos observables v{TAXONOMY_VERSION}

Este documento es el entregable central de la Fase 0 y publica las {_ALL} clases
de la taxonomia. **Solo una de sus dos mitades se somete a validacion en la
Fase 0.5.** El criterio de salida de esa fase es que **dos anotadores apliquen
de forma consistente las {_SPEECH} clases del habla** sobre una muestra piloto, y
que las categorias no resueltas queden documentadas.

## Alcance: cual mitad se valida y cual no

| Mitad | Estado |
|---|---|
| {_SPEECH} clases del habla | Se valida en la Fase 0.5. Sin ejecutar todavia. |
| {_VISUAL} clases visuales | Publicadas y sin validar. Entran asi a la Fase 5. |

Los pilotos de la Fase 0.5 son solo de audio: ninguno le muestra un fotograma
de video a un anotador, asi que nadie ha medido si dos personas aplican igual
las clases visuales. Se publican igual porque la Fase 5 las va a necesitar, y
publicadas sin validar no quiere decir provisionales: quiere decir que no
existe ninguna medicion de consistencia sobre ellas y que ni siquiera esta
decidido si son eventos. La iluminacion insuficiente es una condicion continua,
y dos personas marcandola sobre tramos que se solapan estan delimitando, no
detectando.

La mitad del habla tampoco esta medida aun. Su piloto esta escrito y esperando
a dos anotadores entrenados; lo que la separa de la otra mitad es que tiene
piloto, no que ya tenga resultado.

Ninguna frase de este proyecto afirma que "la taxonomia esta validada": toda
afirmacion de consistencia nombra la mitad de la que habla. El recorte
completo, con sus tres motivos, esta en `docs/corpus/PILOT_PROTOCOL.md`.

## Reglas que gobiernan toda la taxonomia

1. **Solo lo observable.** Cada clase nombra algo que un revisor podria senalar
   en la grabacion: un sonido que ocurrio, una direccion hacia la que apuntaba
   la cabeza, un tramo sin voz. Ninguna clase nombra un estado interno.
2. **La definicion manda sobre el detector.** Estas definiciones estan escritas
   para una persona con una forma de onda y un video delante, no para un
   modelo. El detector aproxima la definicion; nunca al reves.
3. **La duda se anota como duda.** Cuando el contexto no decide, el rol
   contextual es `uncertain`. Un anotador que adivina para no dejar el campo
   vacio introduce ruido que despues se mide como error del modelo.
4. **Las versiones publicadas son inmutables.** Cambiar este archivo cambia
   `TAXONOMY_VERSION`. Los resultados historicos siguen resolviendose contra la
   version bajo la que se produjeron.

## Prohibiciones absolutas

Ningun campo, etiqueta o comentario de anotacion puede nombrar emocion,
ansiedad, nerviosismo, estres, personalidad, engano ni diagnostico clinico.
No es una preferencia de estilo: el sistema no tiene instrumento que mida
ninguna de esas cosas, asi que cualquier etiqueta de ese tipo seria una
invencion. Los terminos vetados estan en `PROHIBITED_CONCEPTS` y hay una prueba
que recorre todo el dominio buscandolos.

Ejemplo de la diferencia:

| Se anota | No se anota |
|---|---|
| "la mirada estuvo fuera de camara 6 s" | "el expositor evitaba el contacto visual" |
| "pausa llena de 780 ms" | "el expositor dudaba" |
| "balanceo lateral de 0,8 Hz durante 12 s" | "el expositor estaba nervioso" |
"""


#: Sits between the visual heading and the first visual class, because that is
#: where a reader who scrolled past the scope section arrives. The header's
#: table says the same thing; repeating it here is deliberate - this manual is
#: read in sections, and the section that publishes nine unmeasured classes is
#: the one that has to carry the warning on its own.
VISUAL_SCOPE_WARNING = f"""\
**Estas {_VISUAL} clases estan publicadas y sin validar.** Aparecen completas
porque la Fase 5 las va a necesitar, no porque alguien haya medido si dos
anotadores las aplican igual: los pilotos de la Fase 0.5 son solo de audio. Una
anotacion visual hecha antes de que la Fase 5 corra su propio piloto es una
anotacion cuya consistencia nadie puede citar."""


def _class_section(
    title: str,
    definitions: Mapping[object, EventDefinition],
    preamble: str = "",
) -> str:
    lines = [f"\n## {title}\n"]
    if preamble:
        lines.append(f"\n{preamble}\n")
    for event_type, spec in definitions.items():
        identifier = getattr(event_type, "value", str(event_type))
        lines.append(f"\n### `{identifier}` - {spec.display_name_es}\n")
        lines.append(
            f"**Prioridad:** {spec.priority.value} · "
            f"**Modalidad:** {spec.modality.value} · "
            f"**Base de deteccion:** {spec.basis.value}\n"
        )
        lines.append(f"\n{spec.definition_es}\n")
        lines.append(f"\n**Ejemplo positivo:** {spec.positive_example_es}\n")
        lines.append("\n**No se anota como esta clase:**\n")
        lines.extend(f"\n- {example}" for example in spec.negative_examples_es)
        lines.append("\n")
    return "".join(lines)


def render() -> str:
    parts = [
        HEADER,
        _class_section(
            "Clases del habla (FR-012)",
            {t: SPEECH_EVENT_DEFINITIONS[t] for t in SpeechEventType},
        ),
        "\n## Roles contextuales (FR-013)\n",
        "\nSe asignan **solo** a las clases lexicas: `lexical_filler`, "
        "`repetition`, `false_start` y `self_repair`. Una pausa llena no tiene "
        "palabra a la que asignarle un rol, y forzarle uno invitaria al "
        "clasificador a inventarlo.\n",
        "\n| Rol | Cuando se aplica |\n|---|---|\n",
        "| `filler` | Hesitacion. No aporta contenido proposicional. |\n",
        "| `semantic` | La expresion tiene significado lexico en esa posicion. |\n",
        "| `discourse_marker` | Estructura el discurso: reformula, cierra, cede el turno. |\n",
        "| `uncertain` | El contexto no lo decide. Se reporta asi; no se adivina. |\n",
        "\n**Solo el rol `filler` cuenta como defecto.** La expresion cruda se "
        "conserva siempre, incluso cuando el rol resulta ser `semantic` "
        "(FR-017): sin esos casos no se puede calcular la precision por clase "
        "que exige la NFR-003.\n",
        _class_section(
            "Clases visuales (FR-019 a FR-022)",
            {t: VISUAL_EVENT_DEFINITIONS[t] for t in VisualEventType},
            preamble=VISUAL_SCOPE_WARNING,
        ),
        "\n## Indicadores prosodicos (FR-018)\n",
        "\nCantidades medibles, cada una con su unidad fija. El analizador de "
        "prosodia produce esto y nada mas; §5 le prohibe inferir emociones.\n",
        "\n| Indicador | Unidad |\n|---|---|\n",
    ]
    parts.extend(
        f"| `{indicator.value}` | {INDICATOR_UNITS[indicator]} |\n"
        for indicator in ProsodicIndicator
    )
    parts.append("\n## Terminos vetados\n\n")
    parts.append(", ".join(f"`{term}`" for term in sorted(PROHIBITED_CONCEPTS)))
    parts.append("\n")
    return "".join(parts)


def main() -> int:
    content = render()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
