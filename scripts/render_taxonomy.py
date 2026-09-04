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

HEADER = f"""<!--
GENERATED FILE - do not edit by hand.

Source: src/evidence_engine/domain/shared/taxonomy.py
Regenerate: uv run python scripts/render_taxonomy.py
Guarded by: tests/unit/test_taxonomy_doc_is_current.py
-->

# Manual de anotacion - Taxonomia de eventos observables v{TAXONOMY_VERSION}

Este documento es el entregable central de la Fase 0. El criterio de salida de
esa fase es que **dos anotadores puedan aplicar esta taxonomia de forma
consistente** sobre una muestra piloto, y que las categorias no resueltas
queden documentadas.

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


def _class_section(title: str, definitions: Mapping[object, EventDefinition]) -> str:
    lines = [f"\n## {title}\n"]
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
