# CLAUDIO_AI: instrucciones para Claude Code

- **Idioma y estilo:** hablar en español, directo y conciso. El usuario es Tomas.
- **Antes de hacer nada:** leer el contexto completo en `claudio_v2/CONTEXTO_CLAUDE_CODE.md`, que tiene objetivo, reglas de etiquetado, pipeline, resultados y pendientes.
- **Objetivo:** YOLO nano de una sola clase `componente` para planos unifilares.
- **Prioridad:** recall 100% (0 falsos negativos). Los falsos positivos se aceptan.
- **Evaluación:** siempre con los DXF con texto. Usar `claudio_v2/src/evaluate.py` (2 escalas) y `comparar.py`, con la tabla por umbral de 0.25 a 0.05.
- **Datos externos:** los planos externos y los folios de QElectroTech son SOLO para test.
- **Rastros de `.elmt`:** no dejar rastros en los pesos. Usar `best_limpio.pt` y DXF sin metadata.
- **Entrenamientos en curso:** no cortar entrenamientos que estén corriendo (`log_D.txt`, `log_E.txt`) sin avisar.
- **Borrado:** nada de borrar definitivo; mover a `_para_borrar`.
- **Etiquetas:** mostrar grillas numeradas antes de cambiar etiquetas o entrenar con datos nuevos.
