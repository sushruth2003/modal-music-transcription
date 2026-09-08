# YourMT3+ MoE deployment — 6 September 2026

Auto Transcribe uses YourMT3+ MoE on the existing live URL:
https://sushruthb03--transcribe.modal.run/

Modal app `ap-3xSUG8FZ4gLuWHo4joeGuc`, deployment **v26**. Previous deployment:
**v24**, retained for rollback. No model/artifact Volumes or job dictionaries were replaced.

## What changed

- Replaced the production MuScriptor worker with the evaluated YourMT3+ MoE
  checkpoint, FP32, batch size 8, on one L4. It still scales to zero after 30 seconds.
- Pinned author source and checkpoint to HF Space `mimbres/YourMT3`, revision
  `5e66c1ea173a8186e0d20432b841d3180cc015b5`. The 561,544,628-byte checkpoint is verified
  against SHA-256 `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25`
  during download and worker startup. Runtime downloads are disabled.
- Preserved predicted solo-instrument labels instead of the author's demo export
  that merges strings and other instrument families. Singing and chorus have
  separate track names/events; standard MIDI voice patches 53/52 provide playback,
  and `yourmt3_program=100/101` text metadata retains the original class identity.
- Instrument selection now filters predicted tracks. The picker explains this;
  it does not condition the model or relabel predictions.
- MIDI and browser notes retain performance timestamps. Beat detection supplies
  tempo/meter without the former MuScriptor onset correction or MIDI bar padding.
  Overlapping unisons get separate tracks/channels; MIDI ports isolate more than
  15 pitched tracks. Port handling varies between MIDI applications.
- A score request with zero detected notes completes with an empty MIDI and an
  explanation that no PDF can be rendered.

## Validation

69 local tests passed, plus formatting, lint and whitespace checks.

Seven complete pipeline runs passed before deployment: 30-second URMP ensemble,
Slakh band mixture, GuitarSet guitar, MAESTRO piano, 2-second silence, 65-second
synthetic input spanning multiple model batches, and an 8-second MP4 with violin
filtering. The ensemble and filtered-video cases produced valid PDFs. The others
produced valid MIDI and paired note events. Silence produced zero notes.

Four source-stratified spot checks compare the production MIDI exporter with the
frozen architecture-evaluation output. They are regression checks, not a new
independent model benchmark. The model's quality and dataset limitations remain
those documented in the architecture study.

| Recording | Note F1, evaluated → production | Instrument F1, evaluated → production | Duration F1, evaluated → production |
|---|---:|---:|---:|
| URMP 01 Jupiter | 79.646 → 79.646 | 60.177 → 60.177 | 70.796 → 70.796 |
| Slakh Track02032 | 69.776 → 69.776 | 58.296 → 58.296 | 36.054 → 36.054 |
| GuitarSet 04 SS1 | 82.645 → 82.645 | 82.645 → 82.645 | 80.992 → 80.992 |
| MAESTRO piano 01 | 92.877 → 92.877 | 92.877 → 92.877 | 15.890 → 16.164 |

Finer MIDI tick rounding changes one borderline piano duration match. There is no
material quality regression in these spot checks. The four cases are too small
to establish quality on arbitrary recordings; the preceding 32-clip study is the
basis for the model choice.

Deployed HTTP checks passed for home, help, API docs/schema, health, instruments,
all seven saved test results, MIDI/PDF downloads and HTTP audio byte ranges.
Unsupported upload returned 400 before the public hourly quota was reached;
subsequent submission checks correctly returned 429 with a retry interval.
Unknown job returned 404. The explicitly
approved historical synthetic job `24c3ad5f9c48410abdb30aae5864c366` retained identical
audio, MIDI and piano-roll SHA-256 checksums across deployment.

Machine-readable local evidence: `outputs/yourmt3-release-20260906/checks.json`,
`quality-parity.json`, `http-checks.json` and `legacy-before.json`. Real recordings
and generated outputs are excluded from version control.

Browser validation also submitted an MP4 with violin filtering and PDF output
through the public upload form, observed completion, exercised source playback,
seeking, synthesized preview and instrument muting, and clicked MIDI/PDF links.
The saved result reloaded successfully with no browser console errors. The thirty-second synthesized preview advanced to 24 seconds and paused correctly.
The final empty-result screen showed no PDF link and its explanation, with no
browser errors. New live
job `27ef8992aa4e40e79499f2899cd3881c` returned 3 violin notes, a MIDI and a PDF.

The deployed CLI submitted `5ca833bfc169467f8ab8333546530110` with two seconds of
silence and PDF requested. It completed with zero notes, a valid empty MIDI, no
PDF link, and the expected explanation. Both live jobs report YourMT3+ MoE and
the pinned checkpoint. Their evidence is in `live-jobs.json` and `cli-silence.log`.

## Rollback

If production needs rollback:

```bash
uv run modal app rollback ap-3xSUG8FZ4gLuWHo4joeGuc v24
```

This restores the previous worker and web deployment. The previous
MuScriptor weights and job artifacts are intact. Reverting a deployment does not
undo new job records or alter their saved outputs. The new API/artifact contract
was kept compatible with historical jobs.
