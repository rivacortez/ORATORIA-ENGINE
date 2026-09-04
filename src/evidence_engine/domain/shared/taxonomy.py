"""The published observable taxonomy, version 1.0.0.

This module is the Phase 0 deliverable expressed as code. It is an allowlist:
an event type that is not defined here cannot be constructed, serialized or
published. A denylist was rejected because it only blocks the labels somebody
remembered to forbid, and NFR-024 has to hold against labels nobody has
thought of yet.

Two rules shape every entry.

*Observable only.* Each class names something a reviewer could point at in the
recording - a sound that occurred, a direction a head was facing, a stretch
with no voice in it. None of them name an internal state. "The gaze was off
camera for 60% of the window" is in; "the speaker was nervous" is out, and is
out for scientific reasons before ethical ones: the engine has no instrument
that measures nervousness, so any such label would be an invention.

*Definitions before detectors.* The Spanish definitions here are the ones two
annotators must be able to apply consistently to a pilot sample before Phase 0
can close. They are written for a human with a waveform and a video, not for a
model, and the detector's job is to approximate them - never the reverse.

Changing this file changes ``TAXONOMY_VERSION``. Published versions are
immutable (US-008): historical results keep resolving against the version they
were produced under, so a redefinition never silently rewrites past evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from evidence_engine.domain.shared.errors import ProhibitedLabel
from evidence_engine.domain.shared.provenance import Modality, SemanticVersion

TAXONOMY_VERSION = SemanticVersion(1, 0, 0)


class EventPriority(StrEnum):
    """Delivery priority from §9. P0 classes gate the phase exit criteria."""

    P0 = "P0"
    P1 = "P1"


class DetectionBasis(StrEnum):
    """What kind of evidence a class is decided from.

    Recorded because it determines how a class can fail and therefore how it
    must be evaluated. NFR-003 requires ambiguous lexical fillers to be scored
    separately from acoustic filled pauses precisely because the two fail in
    different ways: the acoustic one confuses a sound, the lexical one confuses
    a meaning, and averaging them into a single F1 hides both.
    """

    #: Decided from the signal alone - duration, energy, spectral stability.
    ACOUSTIC = "acoustic"
    #: Decided from the recognized word string.
    LEXICAL = "lexical"
    #: Requires both: the word, and the acoustics or context around it.
    LEXICAL_CONTEXTUAL = "lexical_contextual"
    #: Decided from the absence of signal over time.
    TEMPORAL = "temporal"
    #: Decided from landmark geometry across frames.
    GEOMETRIC = "geometric"


class SpeechEventType(StrEnum):
    """The verbatim disfluency classes required by FR-012."""

    FILLED_PAUSE = "filled_pause"
    LEXICAL_FILLER = "lexical_filler"
    REPETITION = "repetition"
    FALSE_START = "false_start"
    SELF_REPAIR = "self_repair"
    CUT_OFF = "cut_off"
    PROLONGATION = "prolongation"
    SILENT_PAUSE = "silent_pause"
    UNINTELLIGIBLE = "unintelligible"


class ContextualRole(StrEnum):
    """The role a potentially ambiguous expression plays (FR-013).

    This is the distinction that stops the engine from counting every "o sea"
    as a defect. In Peruvian Spanish - as in most varieties - the same string
    is sometimes a hesitation, sometimes a genuine reformulation marker, and
    sometimes carries lexical meaning outright ("este" as a demonstrative).
    Collapsing the three inflates the filler count on exactly the speakers who
    use discourse markers well.

    ``UNCERTAIN`` is a first-class outcome, not a failure. §17 prescribes it as
    the mitigation for ambiguous fillers, and an honest "we could not tell"
    costs a consumer far less than a confident wrong call.
    """

    #: Hesitation. Contributes no propositional content.
    FILLER = "filler"
    #: Carries lexical meaning in this position.
    SEMANTIC = "semantic"
    #: Structures the discourse (reformulation, closing, turn management).
    DISCOURSE_MARKER = "discourse_marker"
    #: The context does not decide it. Reported as such, never guessed.
    UNCERTAIN = "uncertain"


class VisualEventType(StrEnum):
    """Observable visual classes required by FR-019 through FR-022."""

    GAZE_TOWARD_CAMERA = "gaze_toward_camera"
    GAZE_AWAY_FROM_CAMERA = "gaze_away_from_camera"
    FACE_VISIBILITY_LOSS = "face_visibility_loss"
    INSUFFICIENT_LIGHTING = "insufficient_lighting"
    INSUFFICIENT_FRAMING = "insufficient_framing"
    POSTURE_DEVIATION = "posture_deviation"
    REPETITIVE_TORSO_MOVEMENT = "repetitive_torso_movement"
    SELF_TOUCH = "self_touch"
    REPETITIVE_HAND_MOVEMENT = "repetitive_hand_movement"


class ProsodicIndicator(StrEnum):
    """The measurable vocal indicators of FR-018.

    Measurable is the operative word. Each is a quantity with a unit and an
    instrument. §5 forbids the prosody analyzer from inferring emotions, and
    this list is what it is allowed to produce instead.
    """

    MEAN_INTENSITY_DB = "mean_intensity_db"
    PITCH_MEAN_HZ = "pitch_mean_hz"
    PITCH_STABILITY = "pitch_stability"
    SPEAKING_RATE_WPM = "speaking_rate_wpm"
    ARTICULATION_RATE_SPS = "articulation_rate_sps"
    MEAN_PAUSE_DURATION_MS = "mean_pause_duration_ms"
    VOICED_RATIO = "voiced_ratio"


@dataclass(frozen=True, slots=True)
class EventDefinition:
    """One published class: what it is, how it is decided, what it is not.

    ``negative_examples_es`` is not documentation padding. The Phase 0 exit
    criterion is inter-annotator agreement, and agreement is won or lost on the
    boundary cases, not on the clear ones. Every class therefore ships with the
    nearby thing it must not be confused with.
    """

    identifier: str
    modality: Modality
    priority: EventPriority
    basis: DetectionBasis
    display_name_es: str
    definition_es: str
    positive_example_es: str
    negative_examples_es: tuple[str, ...]


def _speech(
    event_type: SpeechEventType,
    priority: EventPriority,
    basis: DetectionBasis,
    display_name_es: str,
    definition_es: str,
    positive_example_es: str,
    *negative_examples_es: str,
) -> tuple[SpeechEventType, EventDefinition]:
    return event_type, EventDefinition(
        identifier=event_type.value,
        modality=Modality.AUDIO,
        priority=priority,
        basis=basis,
        display_name_es=display_name_es,
        definition_es=definition_es,
        positive_example_es=positive_example_es,
        negative_examples_es=negative_examples_es,
    )


def _visual(
    event_type: VisualEventType,
    priority: EventPriority,
    display_name_es: str,
    definition_es: str,
    positive_example_es: str,
    *negative_examples_es: str,
) -> tuple[VisualEventType, EventDefinition]:
    return event_type, EventDefinition(
        identifier=event_type.value,
        modality=Modality.VIDEO,
        priority=priority,
        basis=DetectionBasis.GEOMETRIC,
        display_name_es=display_name_es,
        definition_es=definition_es,
        positive_example_es=positive_example_es,
        negative_examples_es=negative_examples_es,
    )


SPEECH_EVENT_DEFINITIONS: Mapping[SpeechEventType, EventDefinition] = MappingProxyType(
    dict(
        [
            _speech(
                SpeechEventType.FILLED_PAUSE,
                EventPriority.P0,
                DetectionBasis.ACOUSTIC,
                "Pausa llena",
                "Vocalizacion sostenida sin contenido lexico que ocupa el lugar de una "
                "pausa. Se decide por el sonido y su duracion, no por como el "
                "reconocedor la haya escrito: el mismo sonido sale transcrito como "
                "'eh', 'he', 'e' o 'y' segun el contexto.",
                "«El objetivo del proyecto eeeh consiste en...»",
                "Una conjuncion 'y' realmente pronunciada y breve entre dos sintagmas.",
                "El alargamiento de la vocal final de una palabra con contenido: eso es "
                "una prolongacion, no una pausa llena.",
            ),
            _speech(
                SpeechEventType.LEXICAL_FILLER,
                EventPriority.P0,
                DetectionBasis.LEXICAL_CONTEXTUAL,
                "Muletilla lexica",
                "Palabra o locucion existente en el idioma usada como apoyo de "
                "hesitacion en vez de por su significado. Solo cuenta como muletilla "
                "cuando el rol contextual es 'filler'; si aporta significado o "
                "estructura el discurso se anota con su rol correspondiente y NO se "
                "cuenta como defecto.",
                "«Entonces este... pasamos al siguiente punto.» ('este' sin referente)",
                "«Este resultado contradice la hipotesis.» ('este' es demostrativo: rol "
                "semantico).",
                "«O sea, lo que quiero decir es otra cosa.» reformulando de verdad: rol "
                "marcador del discurso.",
            ),
            _speech(
                SpeechEventType.REPETITION,
                EventPriority.P0,
                DetectionBasis.LEXICAL_CONTEXTUAL,
                "Repeticion",
                "Una o mas palabras contiguas repetidas sin que la repeticion aporte "
                "contenido nuevo ni enfasis retorico deliberado.",
                "«Vamos a a analizar los datos.»",
                "Repeticion con funcion enfatica marcada por prosodia: «Es muy, muy importante.»",
                "Repeticion separada por una reformulacion: eso es autocorreccion.",
            ),
            _speech(
                SpeechEventType.FALSE_START,
                EventPriority.P0,
                DetectionBasis.LEXICAL_CONTEXTUAL,
                "Falso inicio",
                "Un fragmento de enunciado que se abandona antes de completarse y se "
                "sustituye por otro que empieza de nuevo, sin corregir el anterior.",
                "«Los resultados mues... En el capitulo tres explico la metodologia.»",
                "Una pausa larga en mitad de una oracion que despues SI se completa: "
                "eso es pausa silenciosa.",
            ),
            _speech(
                SpeechEventType.SELF_REPAIR,
                EventPriority.P0,
                DetectionBasis.LEXICAL_CONTEXTUAL,
                "Autocorreccion",
                "El hablante interrumpe lo que dice y lo reemplaza por una version "
                "corregida del mismo contenido. Se distingue del falso inicio en que "
                "aqui SI hay relacion entre lo abandonado y lo que sigue.",
                "«La muestra fue de treinta... perdon, de trescientos participantes.»",
                "Un cambio de tema sin relacion con lo anterior: eso es falso inicio.",
            ),
            _speech(
                SpeechEventType.CUT_OFF,
                EventPriority.P0,
                DetectionBasis.ACOUSTIC,
                "Truncamiento",
                "Una palabra interrumpida antes de terminarse, tipicamente con corte "
                "glotal audible.",
                "«El resulta— el hallazgo principal es...»",
                "Una palabra corta pronunciada completa que suena parecida a un fragmento.",
            ),
            _speech(
                SpeechEventType.PROLONGATION,
                EventPriority.P0,
                DetectionBasis.ACOUSTIC,
                "Alargamiento",
                "Estiramiento anomalo de un segmento dentro de una palabra con "
                "contenido, por encima de la duracion tipica de ese segmento para el "
                "hablante.",
                "«La metodologiiiia que empleamos...»",
                "Alargamiento con funcion enfatica o de enumeracion.",
                "Una vocalizacion aislada sin palabra que la sostenga: eso es pausa llena.",
            ),
            _speech(
                SpeechEventType.SILENT_PAUSE,
                EventPriority.P0,
                DetectionBasis.TEMPORAL,
                "Pausa silenciosa",
                "Intervalo sin voz por encima del umbral configurado. El umbral es "
                "versionado y configurable (FR-015): no es una constante del idioma, "
                "asi que cambiarlo cambia la version de configuracion, no el codigo.",
                "Silencio de 1,8 s entre dos oraciones con el umbral en 700 ms.",
                "La pausa respiratoria breve por debajo del umbral.",
                "El silencio anterior al inicio del habla y el posterior a su final.",
            ),
            _speech(
                SpeechEventType.UNINTELLIGIBLE,
                EventPriority.P0,
                DetectionBasis.ACOUSTIC,
                "Fragmento ininteligible",
                "Tramo con voz cuyo contenido no puede determinarse. Se marca; nunca se "
                "reconstruye. US-002 lo exige de forma explicita: los fragmentos "
                "inciertos se marcan en vez de inventarse.",
                "Un tramo de 900 ms tapado por ruido de la sala.",
                "Un tramo que el anotador SI entiende aunque el modelo falle: eso es un "
                "error del modelo, no un fragmento ininteligible.",
            ),
        ]
    )
)


VISUAL_EVENT_DEFINITIONS: Mapping[VisualEventType, EventDefinition] = MappingProxyType(
    dict(
        [
            _visual(
                VisualEventType.GAZE_TOWARD_CAMERA,
                EventPriority.P0,
                "Mirada hacia la camara",
                "La direccion estimada de la mirada cae dentro del cono calibrado que "
                "apunta a la camara. Es una ESTIMACION a partir de geometria facial: "
                "§5 prohibe presentarla como seguimiento ocular de grado hardware.",
                "El rostro y la mirada orientados al lente durante 4 s seguidos.",
                "Cabeza orientada al frente con la mirada claramente desviada: la "
                "orientacion de cabeza no es mirada.",
            ),
            _visual(
                VisualEventType.GAZE_AWAY_FROM_CAMERA,
                EventPriority.P0,
                "Mirada fuera de la camara",
                "La direccion estimada de la mirada cae fuera del cono calibrado, con "
                "la direccion de salida registrada cuando puede estimarse.",
                "Mirada dirigida hacia abajo, a los apuntes, durante 6 s.",
                "Perdida de deteccion del rostro: eso es perdida de visibilidad, no "
                "mirada desviada.",
            ),
            _visual(
                VisualEventType.FACE_VISIBILITY_LOSS,
                EventPriority.P0,
                "Perdida de visibilidad del rostro",
                "El rostro deja de ser detectable con la confianza minima requerida. "
                "Es una condicion de calidad, y su consecuencia es que los "
                "indicadores dependientes se marcan como no disponibles con motivo, "
                "nunca en cero.",
                "El hablante se gira por completo hacia la pantalla proyectada.",
                "Rostro visible pero mal iluminado: eso es iluminacion insuficiente.",
            ),
            _visual(
                VisualEventType.INSUFFICIENT_LIGHTING,
                EventPriority.P0,
                "Iluminacion insuficiente",
                "La luminancia de la region facial cae por debajo del umbral que hace "
                "fiables las estimaciones geometricas.",
                "Contraluz de ventana que deja el rostro en sombra.",
                "Escena oscura con el rostro correctamente iluminado.",
            ),
            _visual(
                VisualEventType.INSUFFICIENT_FRAMING,
                EventPriority.P0,
                "Encuadre insuficiente",
                "El encuadre no contiene la region corporal que un indicador necesita: "
                "sin hombros no hay postura, sin manos no hay gesto.",
                "Plano cerradisimo que corta el encuadre por encima de los hombros.",
                "Plano amplio y correcto con el hablante lejos pero completo.",
            ),
            _visual(
                VisualEventType.POSTURE_DEVIATION,
                EventPriority.P0,
                "Desviacion postural",
                "Desviacion de la pose respecto de la postura de referencia capturada "
                "en la calibracion del propio hablante. Es relativa a su linea base, "
                "no a una postura ideal universal.",
                "Inclinacion sostenida del torso 20 grados respecto de la calibracion.",
                "Un cambio de postura unico y estable: eso es una nueva linea base, no "
                "una desviacion sostenida.",
            ),
            _visual(
                VisualEventType.REPETITIVE_TORSO_MOVEMENT,
                EventPriority.P0,
                "Movimiento repetitivo del torso",
                "Oscilacion del torso con periodicidad detectable sostenida en el "
                "tiempo. Se reporta el movimiento y su frecuencia; §5 prohibe "
                "etiquetarlo como nerviosismo.",
                "Balanceo lateral regular de ~0,8 Hz durante 12 s.",
                "Un desplazamiento unico para acercarse a la pizarra.",
            ),
            _visual(
                VisualEventType.SELF_TOUCH,
                EventPriority.P1,
                "Autocontacto visible",
                "Contacto sostenido de la mano con el rostro, el cabello, el cuello o "
                "la ropa. FR-022 lo condiciona: solo se emite cuando el encuadre y la "
                "confianza superan sus umbrales; en caso contrario no se emite nada.",
                "Mano apoyada en la nuca durante 3 s con manos y rostro visibles.",
                "Mano que pasa por delante del rostro sin contacto.",
                "Gesto ilustrativo cerca de la cara.",
            ),
            _visual(
                VisualEventType.REPETITIVE_HAND_MOVEMENT,
                EventPriority.P1,
                "Movimiento repetitivo de manos",
                "Movimiento de manos con periodicidad detectable y sin funcion "
                "ilustrativa aparente en el discurso.",
                "Clic repetido de un boligrafo durante 8 s.",
                "Gesticulacion variada acompanando la explicacion.",
            ),
        ]
    )
)


#: Concepts the service must never emit under any name (NFR-024, FR-023).
#: Used by the negative tests that walk every published schema and label. The
#: allowlist above is the real defence; this set is the tripwire that catches a
#: prohibited concept sneaking in through a field name or an enum member.
PROHIBITED_CONCEPTS: frozenset[str] = frozenset(
    {
        "emotion",
        "emotional",
        "mood",
        "affect",
        "sentiment",
        "anxiety",
        "anxious",
        "nervous",
        "nervousness",
        "stress",
        "stressed",
        "fear",
        "confidence_level",
        "insecurity",
        "insecure",
        "personality",
        "trait",
        "deception",
        "deceptive",
        "lying",
        "truthfulness",
        "diagnosis",
        "diagnostic",
        "disorder",
        "stutter",
        "stuttering",
        "pathology",
        "clinical",
        "symptom",
    }
)


def require_known_speech_event(identifier: str) -> SpeechEventType:
    """Resolve a speech event identifier against the allowlist, or refuse."""
    try:
        return SpeechEventType(identifier)
    except ValueError as exc:
        raise ProhibitedLabel(
            f"'{identifier}' is not in speech taxonomy {TAXONOMY_VERSION}; "
            "only published observable classes may be emitted"
        ) from exc


def require_known_visual_event(identifier: str) -> VisualEventType:
    """Resolve a visual event identifier against the allowlist, or refuse."""
    try:
        return VisualEventType(identifier)
    except ValueError as exc:
        raise ProhibitedLabel(
            f"'{identifier}' is not in visual taxonomy {TAXONOMY_VERSION}; "
            "only published observable classes may be emitted"
        ) from exc


def definition_of(event_type: SpeechEventType | VisualEventType) -> EventDefinition:
    """The published definition behind a class, for manuals and result docs."""
    if isinstance(event_type, SpeechEventType):
        return SPEECH_EVENT_DEFINITIONS[event_type]
    return VISUAL_EVENT_DEFINITIONS[event_type]


def p0_speech_events() -> frozenset[SpeechEventType]:
    """The classes NFR-002 holds to a macro-F1 target."""
    return frozenset(
        event
        for event, spec in SPEECH_EVENT_DEFINITIONS.items()
        if spec.priority is EventPriority.P0
    )
