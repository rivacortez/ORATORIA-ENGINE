<!--
GENERATED FILE - do not edit by hand.

Source: src/evidence_engine/domain/shared/taxonomy.py
Regenerate: uv run python scripts/render_taxonomy.py
Guarded by: tests/unit/test_taxonomy_doc_is_current.py
-->

# Manual de anotacion - Taxonomia de eventos observables v1.0.0

Este documento es el entregable central de la Fase 0 y publica las 18 clases
de la taxonomia. **Solo una de sus dos mitades se somete a validacion en la
Fase 0.5.** El criterio de salida de esa fase es que **dos anotadores apliquen
de forma consistente las 9 clases del habla** sobre una muestra piloto, y
que las categorias no resueltas queden documentadas.

## Alcance: cual mitad se valida y cual no

| Mitad | Estado |
|---|---|
| 9 clases del habla | Se valida en la Fase 0.5. Sin ejecutar todavia. |
| 9 clases visuales | Publicadas y sin validar. Entran asi a la Fase 5. |

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

## Clases del habla (FR-012)

### `filled_pause` - Pausa llena
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** acoustic

Vocalizacion sostenida sin contenido lexico que ocupa el lugar de una pausa. Se decide por el sonido y su duracion, no por como el reconocedor la haya escrito: el mismo sonido sale transcrito como 'eh', 'he', 'e' o 'y' segun el contexto.

**Ejemplo positivo:** «El objetivo del proyecto eeeh consiste en...»

**No se anota como esta clase:**

- Una conjuncion 'y' realmente pronunciada y breve entre dos sintagmas.
- El alargamiento de la vocal final de una palabra con contenido: eso es una prolongacion, no una pausa llena.

### `lexical_filler` - Muletilla lexica
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** lexical_contextual

Palabra o locucion existente en el idioma usada como apoyo de hesitacion en vez de por su significado. Solo cuenta como muletilla cuando el rol contextual es 'filler'; si aporta significado o estructura el discurso se anota con su rol correspondiente y NO se cuenta como defecto.

**Ejemplo positivo:** «Entonces este... pasamos al siguiente punto.» ('este' sin referente)

**No se anota como esta clase:**

- «Este resultado contradice la hipotesis.» ('este' es demostrativo: rol semantico).
- «O sea, lo que quiero decir es otra cosa.» reformulando de verdad: rol marcador del discurso.

### `repetition` - Repeticion
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** lexical_contextual

Una o mas palabras contiguas repetidas sin que la repeticion aporte contenido nuevo ni enfasis retorico deliberado.

**Ejemplo positivo:** «Vamos a a analizar los datos.»

**No se anota como esta clase:**

- Repeticion con funcion enfatica marcada por prosodia: «Es muy, muy importante.»
- Repeticion separada por una reformulacion: eso es autocorreccion.

### `false_start` - Falso inicio
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** lexical_contextual

Un fragmento de enunciado que se abandona antes de completarse y se sustituye por otro que empieza de nuevo, sin corregir el anterior.

**Ejemplo positivo:** «Los resultados mues... En el capitulo tres explico la metodologia.»

**No se anota como esta clase:**

- Una pausa larga en mitad de una oracion que despues SI se completa: eso es pausa silenciosa.

### `self_repair` - Autocorreccion
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** lexical_contextual

El hablante interrumpe lo que dice y lo reemplaza por una version corregida del mismo contenido. Se distingue del falso inicio en que aqui SI hay relacion entre lo abandonado y lo que sigue.

**Ejemplo positivo:** «La muestra fue de treinta... perdon, de trescientos participantes.»

**No se anota como esta clase:**

- Un cambio de tema sin relacion con lo anterior: eso es falso inicio.

### `cut_off` - Truncamiento
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** acoustic

Una palabra interrumpida antes de terminarse, tipicamente con corte glotal audible.

**Ejemplo positivo:** «El resulta— el hallazgo principal es...»

**No se anota como esta clase:**

- Una palabra corta pronunciada completa que suena parecida a un fragmento.

### `prolongation` - Alargamiento
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** acoustic

Estiramiento anomalo de un segmento dentro de una palabra con contenido, por encima de la duracion tipica de ese segmento para el hablante.

**Ejemplo positivo:** «La metodologiiiia que empleamos...»

**No se anota como esta clase:**

- Alargamiento con funcion enfatica o de enumeracion.
- Una vocalizacion aislada sin palabra que la sostenga: eso es pausa llena.

### `silent_pause` - Pausa silenciosa
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** temporal

Intervalo sin voz por encima del umbral configurado. El umbral es versionado y configurable (FR-015): no es una constante del idioma, asi que cambiarlo cambia la version de configuracion, no el codigo.

**Ejemplo positivo:** Silencio de 1,8 s entre dos oraciones con el umbral en 700 ms.

**No se anota como esta clase:**

- La pausa respiratoria breve por debajo del umbral.
- El silencio anterior al inicio del habla y el posterior a su final.

### `unintelligible` - Fragmento ininteligible
**Prioridad:** P0 · **Modalidad:** audio · **Base de deteccion:** acoustic

Tramo con voz cuyo contenido no puede determinarse. Se marca; nunca se reconstruye. US-002 lo exige de forma explicita: los fragmentos inciertos se marcan en vez de inventarse.

**Ejemplo positivo:** Un tramo de 900 ms tapado por ruido de la sala.

**No se anota como esta clase:**

- Un tramo que el anotador SI entiende aunque el modelo falle: eso es un error del modelo, no un fragmento ininteligible.

## Roles contextuales (FR-013)

Se asignan **solo** a las clases lexicas: `lexical_filler`, `repetition`, `false_start` y `self_repair`. Una pausa llena no tiene palabra a la que asignarle un rol, y forzarle uno invitaria al clasificador a inventarlo.

| Rol | Cuando se aplica |
|---|---|
| `filler` | Hesitacion. No aporta contenido proposicional. |
| `semantic` | La expresion tiene significado lexico en esa posicion. |
| `discourse_marker` | Estructura el discurso: reformula, cierra, cede el turno. |
| `uncertain` | El contexto no lo decide. Se reporta asi; no se adivina. |

**Solo el rol `filler` cuenta como defecto.** La expresion cruda se conserva siempre, incluso cuando el rol resulta ser `semantic` (FR-017): sin esos casos no se puede calcular la precision por clase que exige la NFR-003.

## Clases visuales (FR-019 a FR-022)

**Estas 9 clases estan publicadas y sin validar.** Aparecen completas
porque la Fase 5 las va a necesitar, no porque alguien haya medido si dos
anotadores las aplican igual: los pilotos de la Fase 0.5 son solo de audio. Una
anotacion visual hecha antes de que la Fase 5 corra su propio piloto es una
anotacion cuya consistencia nadie puede citar.

### `gaze_toward_camera` - Mirada hacia la camara
**Prioridad:** P0 · **Modalidad:** video · **Base de deteccion:** geometric

La direccion estimada de la mirada cae dentro del cono calibrado que apunta a la camara. Es una ESTIMACION a partir de geometria facial: §5 prohibe presentarla como seguimiento ocular de grado hardware.

**Ejemplo positivo:** El rostro y la mirada orientados al lente durante 4 s seguidos.

**No se anota como esta clase:**

- Cabeza orientada al frente con la mirada claramente desviada: la orientacion de cabeza no es mirada.

### `gaze_away_from_camera` - Mirada fuera de la camara
**Prioridad:** P0 · **Modalidad:** video · **Base de deteccion:** geometric

La direccion estimada de la mirada cae fuera del cono calibrado, con la direccion de salida registrada cuando puede estimarse.

**Ejemplo positivo:** Mirada dirigida hacia abajo, a los apuntes, durante 6 s.

**No se anota como esta clase:**

- Perdida de deteccion del rostro: eso es perdida de visibilidad, no mirada desviada.

### `face_visibility_loss` - Perdida de visibilidad del rostro
**Prioridad:** P0 · **Modalidad:** video · **Base de deteccion:** geometric

El rostro deja de ser detectable con la confianza minima requerida. Es una condicion de calidad, y su consecuencia es que los indicadores dependientes se marcan como no disponibles con motivo, nunca en cero.

**Ejemplo positivo:** El hablante se gira por completo hacia la pantalla proyectada.

**No se anota como esta clase:**

- Rostro visible pero mal iluminado: eso es iluminacion insuficiente.

### `insufficient_lighting` - Iluminacion insuficiente
**Prioridad:** P0 · **Modalidad:** video · **Base de deteccion:** geometric

La luminancia de la region facial cae por debajo del umbral que hace fiables las estimaciones geometricas.

**Ejemplo positivo:** Contraluz de ventana que deja el rostro en sombra.

**No se anota como esta clase:**

- Escena oscura con el rostro correctamente iluminado.

### `insufficient_framing` - Encuadre insuficiente
**Prioridad:** P0 · **Modalidad:** video · **Base de deteccion:** geometric

El encuadre no contiene la region corporal que un indicador necesita: sin hombros no hay postura, sin manos no hay gesto.

**Ejemplo positivo:** Plano cerradisimo que corta el encuadre por encima de los hombros.

**No se anota como esta clase:**

- Plano amplio y correcto con el hablante lejos pero completo.

### `posture_deviation` - Desviacion postural
**Prioridad:** P0 · **Modalidad:** video · **Base de deteccion:** geometric

Desviacion de la pose respecto de la postura de referencia capturada en la calibracion del propio hablante. Es relativa a su linea base, no a una postura ideal universal.

**Ejemplo positivo:** Inclinacion sostenida del torso 20 grados respecto de la calibracion.

**No se anota como esta clase:**

- Un cambio de postura unico y estable: eso es una nueva linea base, no una desviacion sostenida.

### `repetitive_torso_movement` - Movimiento repetitivo del torso
**Prioridad:** P0 · **Modalidad:** video · **Base de deteccion:** geometric

Oscilacion del torso con periodicidad detectable sostenida en el tiempo. Se reporta el movimiento y su frecuencia; §5 prohibe etiquetarlo como nerviosismo.

**Ejemplo positivo:** Balanceo lateral regular de ~0,8 Hz durante 12 s.

**No se anota como esta clase:**

- Un desplazamiento unico para acercarse a la pizarra.

### `self_touch` - Autocontacto visible
**Prioridad:** P1 · **Modalidad:** video · **Base de deteccion:** geometric

Contacto sostenido de la mano con el rostro, el cabello, el cuello o la ropa. FR-022 lo condiciona: solo se emite cuando el encuadre y la confianza superan sus umbrales; en caso contrario no se emite nada.

**Ejemplo positivo:** Mano apoyada en la nuca durante 3 s con manos y rostro visibles.

**No se anota como esta clase:**

- Mano que pasa por delante del rostro sin contacto.
- Gesto ilustrativo cerca de la cara.

### `repetitive_hand_movement` - Movimiento repetitivo de manos
**Prioridad:** P1 · **Modalidad:** video · **Base de deteccion:** geometric

Movimiento de manos con periodicidad detectable y sin funcion ilustrativa aparente en el discurso.

**Ejemplo positivo:** Clic repetido de un boligrafo durante 8 s.

**No se anota como esta clase:**

- Gesticulacion variada acompanando la explicacion.

## Indicadores prosodicos (FR-018)

Cantidades medibles, cada una con su unidad fija. El analizador de prosodia produce esto y nada mas; §5 le prohibe inferir emociones.

| Indicador | Unidad |
|---|---|
| `mean_intensity_db` | dBFS |
| `pitch_mean_hz` | Hz |
| `pitch_stability` | ratio |
| `speaking_rate_wpm` | wpm |
| `articulation_rate_sps` | syllables_per_second |
| `mean_pause_duration_ms` | ms |
| `voiced_ratio` | ratio |

## Terminos vetados

`affect`, `anxiety`, `anxious`, `clinical`, `confidence_level`, `deception`, `deceptive`, `diagnosis`, `diagnostic`, `disorder`, `emotion`, `emotional`, `fear`, `insecure`, `insecurity`, `lying`, `mood`, `nervous`, `nervousness`, `pathology`, `personality`, `sentiment`, `stress`, `stressed`, `stutter`, `stuttering`, `symptom`, `trait`, `truthfulness`
