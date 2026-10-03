# CLAUDIO_AI: instrucciones para Claude Code

- **Idioma y estilo:** hablar en español, directo y conciso. El usuario es Tomas.
- **Antes de hacer nada:** leer `claudio_v2/CONTEXTO.md`. Tiene el objetivo, las reglas, los resultados, lo que está corriendo y el mapa de carpetas. La historia vieja está en `claudio_v2/docs/historial/`: solo si hace falta.
- **Objetivo:** detector de una sola clase `componente` para planos unifilares. Hay que llegar a recall 100% (0 FN) con menos de 350 FP en los 7 planos de test.
- **Evaluación:** siempre sobre los DXF con texto, con `claudio_v2/src/evaluate.py` (2 escalas) y `barrido.py` (tabla por umbral, grupos DES y RES).
- **Datos externos:** los 7 planos de test, los externos y QElectroTech son SOLO test.
- **Rastros de `.elmt`:** no dejar rastros en los pesos. Usar `best_limpio`.
- **Entrenamientos en curso:** no cortar los que están corriendo (ver `claudio_v2/log_cadena_*.txt`) sin avisar.
- **Borrado:** nada de borrar definitivo; mover a `_para_borrar`.
- **Etiquetas:** mostrar grillas numeradas antes de cambiar etiquetas o entrenar con datos nuevos.
