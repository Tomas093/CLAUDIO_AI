"""Receta de ds21 (28/09): la de ds20 (plan U + fusible DXF 0,15) mas lo nuevo de compose.
La usan train.py (hook N21) y grilla_ds21.py, asi la grilla muestra exactamente lo que se entrena."""
RECETA = dict(P_TABLERO='0.32', P_SPM='0.45', P_FUSIBLE='0.35', P_POLO='0.25', P_DENSA='0.28', P_BARRA='0.10',
              P_TRAFO='0.15', P_CELDAS='0.25', P_COMPUESTO='0', P_TERNA='0.35', P_PULS='0.30', P_PAT='0',
              P_MARCO='0.20', GENS_SIN_PUNTEADA='1', GENS_NOMBRE_SIMPLE='1', PESO_TOMAS='6', P_FUSDXF='0.15',
              FONDO_BLANCO='1', VIS_MIN='0.90', P_LETRA='0.15', P_ROTULO='0', P_ROTULO_NEG='0.30', P_NEGEXTRA='0.40',
              PSEUDO_CONFIRMA='1',
              # 28/09, revision de Tomas (work/revision_tomas): sinteticos solo con simbolos 100% adentro del tile
              # (con 0,90 igual marco recortes de 94-97%); reales/manuales siguen con VIS_MIN 0,90.
              VIS_SINT='1.0',
              # sprites que no son un componente (varios simbolos juntos, pedazos, o no nombrados por Tomas):
              # e00400, e00508, e00234 (juntos/pedazos), e00095, e00336 (grilla E). Mas la lista de siempre.
              EXCLUIR_SIM='p0245.png,p0247.png,p0253.png,p0259.png,p0260.png,p0261.png,p0262.png,p0281.png,'
                          'p0286.png,e00232.png,e00095.png,e00400.png,e00336.png,e00508.png,e00234.png')   # 28/09: pseudo-etiquetas de zips solo si RF4 las ve y no tocan el borde
