# Consent, retention and deletion policy

**Policy version:** 1.0.0 · **Status:** approved for Phase 0

This is the policy the code enforces, not a description of it. The version
string here is the one recorded in every `ConsentReceipt`; editing this document
without bumping the version would make historical receipts attest to wording
their signatories never saw.

---

## 1. The ordering rule

> Consent is recorded **before** protected media is accepted. (FR-031)

This is the whole design, and it is the one thing that cannot be relaxed for
convenience. A receipt written after the upload describes a decision the
participant had already lost the chance to make. In code, the gate lives on the
session aggregate rather than in a request handler, so the WebSocket path
cannot skip what the REST path checks.

## 2. Retention

**Default: ephemeral.** Raw media is processed and discarded. Nothing needs to
opt out of retention, because nothing opts in by default.

| Category                                         | Default                                      | Longer retention                                                          |
| ------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------- |
| Raw audio and video                              | Deleted once the run that needs it completes | Requires an explicit versioned policy with a positive TTL                 |
| Derived features (spectrograms, landmark series) | Deleted with the raw media                   | Same                                                                      |
| Derived non-identifying aggregates               | Retained when the consent policy allows      | These are what make a result reproducible after the recording is gone     |
| Consent receipts                                 | Retained                                     | Deleting the receipt would destroy the proof that deletion was authorised |
| Audit records                                    | Retained, append-only                        | They prove execution; they never contain the deleted content              |

A policy that claims to retain raw media without stating a TTL is rejected by
the constructor. So is a TTL on a policy that does not retain — both are
symptoms of a caller that has not decided, and an undecided retention policy
becomes an indefinite one.

## 3. Withdrawal

- Withdrawal **requires no justification**. There is no field for one; adding
  one would invite an interface to demand it.
- Withdrawal is recorded on the receipt rather than by deleting it, so the
  deletion can be demonstrated end to end afterwards.
- A withdrawal mid-session stops the next `resume`, not only the next `begin`.
  The resume path is the one an interrupted presentation actually takes.

## 4. Deletion

A validated deletion request removes raw media and directly identifying session
evidence **within 24 hours** and produces an audit record. In practice the
service deletes synchronously; the 24-hour figure is the ceiling that covers
replica and backup propagation.

The order of operations runs from least to most recoverable:

1. **Revoke outstanding signed URLs.** A URL minted a minute before the request
   would otherwise keep working against a cache or a replica after the bytes
   are gone.
2. Delete raw media.
3. Delete protected derived evidence.
4. Mark the session deleted, which also withdraws consent.
5. Write the audit record — **last**, and carrying counts.

The audit record goes last because writing it first would attest to work that
had not happened. It carries counts (`media_objects=1 evidence_records=412`)
because "deletion ran and touched N rows" is provable and "deletion requested"
is not.

**Verification** is a separate read-only call. A verification that deleted as a
side effect of checking would always pass — it would be reporting on its own
actions rather than on the state deletion left behind.

Repeated deletion requests are idempotent. A client retrying after a timeout
must not be told its withdrawal failed.

## 5. What never leaves

- **Raw secrets.** API keys are stored as a peppered SHA-256 hash and are
  returned to the caller exactly once, at issue.
- **Transcript content, in operational logs.** The telemetry port accepts
  scalars only, and the adapter additionally redacts any field whose name
  suggests evidence. The redaction is crude on purpose: it will occasionally
  blank something harmless, which costs a debugging session, while the
  alternative failure mode is a student's transcript in a log aggregator.
- **Cross-tenant data.** Every repository read is scoped by tenant in its
  signature, so the unscoped call cannot be written. Another tenant's session
  returns _not found_, never _forbidden_ — a 403 would confirm it exists.

---

## 6. Participant-facing text (es-PE)

> Este texto es el que ve el participante. Se versiona junto con la politica.

### Que se graba

Durante tu practica se captura **audio** y, si lo autorizas, **video**. El
audio se usa para transcribir literalmente lo que dijiste y para medir pausas,
muletillas y caracteristicas de tu voz. El video se usa para estimar hacia
donde estaba orientada tu mirada, tu postura y si hubo movimientos repetitivos.

### Que NO se hace con tu grabacion

- **No se te diagnostica nada.** El sistema no mide ansiedad, nerviosismo,
  personalidad ni ninguna condicion de salud. No tiene forma de hacerlo.
- **No se te compara con otras personas.** Las mediciones son sobre tu propia
  grabacion y tu propia calibracion.
- **No se guarda tu grabacion por defecto.** Se procesa y se elimina.

### Que se conserva

Se conservan las **mediciones** (por ejemplo: "hubo una pausa de 1,8 s en el
segundo 42") y la transcripcion, no la grabacion. Si en algun momento se
necesitara conservar el audio o el video por mas tiempo, se te pedira
autorizacion explicita y se te dira por cuanto tiempo.

### Tu derecho a retirarte

Puedes pedir que se elimine tu evidencia **en cualquier momento y sin dar
explicaciones**. Al hacerlo:

- se elimina la grabacion y la evidencia identificable;
- se invalidan los enlaces de descarga que existieran;
- recibes una confirmacion;
- queda un registro de que la eliminacion ocurrio, **sin conservar lo que se
  elimino**.

Retirarte no tiene ninguna consecuencia academica.

### Si algo no se pudo medir

Cuando la camara esta tapada, la sala esta muy oscura o se corta la conexion,
el sistema **te dice que ese indicador no se pudo calcular y por que**. Nunca
te reporta un cero en su lugar. Un cero significaria que se midio y salio mal;
"no disponible" significa que no se midio.
