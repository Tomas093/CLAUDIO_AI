# Original User Request

## 2026-09-11T22:45:41Z

Búsqueda y descarga autónoma de planos unifilares eléctricos en formato CAD (DWG y DXF) conforme a la normativa argentina (AEA 90364 / IRAM), combinados con el conjunto existente de esquemas argentinos locales, normalizándolos y evaluándolos exhaustivamente con el detector universal YOLO Nano para garantizar una tasa de acierto (recall) >= 98% sin falsos negativos ni regresiones.

Working directory: c:\Users\Tomas\Documents\LAB3\CLAUDIO_AI
Integrity mode: development

## Planos Pre-Validados en esta Sesión (Norma AEA 90364)
- **Tablero Seccional TSBE (dxf/Tablerotsbe.dxf)**: 15/15 aparatos detectados (**100.0% Recall, 100.0% Precision** a conf 0.50).
- **Tablero de Distribución (dxf/plano3.dxf)**: 25/25 aparatos detectados (**100.0% Recall, 100.0% Precision** a conf 0.50).

## Requirements

### R1. Adquisición y Descarga de Planos Argentinos (Web y Locales)
- Descargar y organizar esquemas unifilares en DWG/DXF provenientes de repositorios abiertos (GitHub, cátedras universitarias UTN/FIUBA, portales de fabricantes Schneider/Siemens Argentina y colegios profesionales).
- Integrar y procesar la totalidad del conjunto de esquemas argentinos locales (Tablerotsbe.dxf, plano.dxf, plano2.dxf, plano3.dxf, plano4.dxf, plano5.dxf y OCJ-DE-IEL-UNI-000-001-O03.dxf).

### R2. Conversión Vectorial y Normalización de Escala
- Para archivos DWG, aplicar conversión a DXF limpio mediante herramientas de línea de comando o utilidades vectoriales.
- Renderizar los planos DXF aplicando ColorPolicy.COLOR_SWAP_BW (preservación de trazos blancos/negros según fondo) y HatchPolicy.NORMAL.
- Normalizar la escala espacial (75 - 100 px/CAD) según las dimensiones de los símbolos modulares y de potencia.

### R3. Inferencia de Componentes con el Detector Universal
- Ejecutar el detector empaquetado (detector_pack/detector_unifilar.py con est_componente_nano.pt) con inferencia por baldosas solapadas (overlap 80%) y NMS centroidal en coordenadas CAD ({min} \approx 0.20$).
- Detectar la gama completa de aparatos: interruptores termomagnéticos (PIA), disyuntores diferenciales (ID), seccionadores bajo carga, descargadores de sobretensión (DPS), grupos electrógenos, transformadores de medida, luces piloto y borneras (ø y sólidas).

### R4. Auditoría Métrica y Generación de Láminas Visuales
- Calcular métricas de Recall, Precisión y conteo de detecciones por plano evaluado.
- Generar láminas PNG en alta resolución con bounding boxes verdes y etiquetas de confianza para cada plano.
- Comprobar que no existan regresiones en los planos patrón (TEST 1, TEST 2, FL-UN-02, TSSS_2, Vyre).

## Acceptance Criteria

### Adquisición y Procesamiento de Planos
- [ ] Incorporación exitosa de al menos 3 nuevos esquemas unifilares descargados de la web + el procesamiento completo de los planos locales de la serie plano1-5 y Tablerotsbe.
- [ ] Renderizado sin fallas ni omisiones gráficas en ningún plano evaluado.

### Calidad de Detección del Modelo
- [ ] Recall >= 98% en cada uno de los esquemas unifilares evaluados.
- [ ] Cero falsas detecciones sistemáticas sobre cables vacíos, flechas de salida o textos de descripción de circuitos.
- [ ] Verificación de cero regresión en los benchmarks históricos.
- [ ] Todas las láminas visuales generadas y guardadas en directorios correspondientes para inspección.
