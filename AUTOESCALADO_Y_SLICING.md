# Autoescalado y Slicing en la Detección de Símbolos Eléctricos

Documento explicativo sobre cómo encaramos los dos problemas centrales del pipeline:
**escala variable entre planos** y **detección de objetos chicos en imágenes grandes**.

---

## Contexto

El objetivo del proyecto es detectar símbolos eléctricos (interruptores, tomacorrientes,
diferenciales, etc.) en planos DXF usando un modelo de visión por computadora basado en
YOLO. Para que YOLO pueda procesar un DXF, primero hay que convertirlo a una imagen PNG.
Y a partir de ese paso aparecen los problemas que este documento describe.

---

## Por qué hace falta autoescalar

Cuando empezamos a trabajar con DXFs reales nos encontramos con que la salida directa
del rasterizado no era utilizable por el modelo. Detectamos varios problemas que
justifican la necesidad de un autoescalado.

### Los DXFs vienen en unidades distintas

Algunos planos están dibujados en milímetros, otros en metros, otros en pulgadas, y
algunos en unidades arbitrarias sin ninguna referencia explícita. Un mismo símbolo de
interruptor puede medir 1.2 unidades en un DXF, 1200 en otro, y 0.05 en un tercero.

### El bounding box global no sirve como referencia

La idea más simple para escalar es mirar las dimensiones totales del plano y ajustar
para que entre en una imagen de tamaño fijo. Pero esto falla rápido. Un plano puede
tener una entidad perdida o un cajetín exageradamente grande que estira el bounding
box, dejando todo lo importante apretado en una esquina diminuta. Además, dos planos
del mismo proyecto pueden tener tamaños totales muy diferentes aunque los símbolos en
ambos sean del mismo tipo y deban verse igual.

### El modelo espera símbolos a un tamaño concreto

YOLO no es magia: fue entrenado con imágenes donde los símbolos tenían un tamaño
aproximado en píxeles. Si en producción los símbolos salen mucho más chicos, el modelo
no los ve. Si salen mucho más grandes, los reconoce parcialmente o los confunde con
otros. La detección depende fuertemente de que los objetos estén en el rango de tamaño
para el que fue entrenado.

### Configurar manualmente la escala no escala

Una solución parche es decirle al script "para este plano usá escala X". Pero eso
implica conocer cada plano de antemano, requiere intervención manual cada vez, y rompe
cualquier intento de automatización. Si el sistema va a procesar decenas o cientos de
planos, no es viable.

---

## Cómo solucionamos el autoescalado

La idea central fue **dejar que el DXF se autoescale a sí mismo** identificando dentro
del propio archivo una entidad que sirva como referencia de tamaño. Si encontramos algo
que sabemos que mide aproximadamente lo mismo en todos los planos, podemos usar su
tamaño en unidades CAD para calcular cuántos píxeles por unidad CAD necesitamos.

Implementamos una cascada de candidatos a entidad de referencia, en orden de
confiabilidad:

Primero buscamos **bloques INSERT**. Los símbolos eléctricos son casi siempre bloques
reutilizables del DXF, y los dibujantes tienden a usar los mismos bloques entre
proyectos. Calculamos la mediana del lado mayor del bounding box de los INSERTs,
filtrando outliers con el rango intercuartil para descartar bloques de cajetín o
leyendas que distorsionarían la mediana.

Si no hay INSERTs útiles, buscamos **círculos pequeños**. Casi todos los símbolos
eléctricos tienen un círculo característico, y su tamaño suele ser consistente.
Aplicamos el mismo filtrado por IQR para evitar que círculos enormes (como los de las
referencias del cajetín) influyan en la mediana.

Si tampoco hay círculos confiables, recurrimos a las **alturas de los textos
técnicos**. Los textos de anotaciones suelen tener una altura estandarizada en planos
profesionales. Excluimos capas conocidas como cajetines y títulos, y agrupamos las
alturas restantes con clustering por bins logarítmicos para identificar el tipo de
texto más numeroso, que casi siempre corresponde a las anotaciones de los símbolos.

Como último recurso, si nada de lo anterior funciona, caemos en un **fallback al
bounding box global** apuntando a una imagen de unos 8000 píxeles en el lado largo.
Este caso es raro y representa un plano sin entidades estándar reconocibles, donde
cualquier escala razonable es lo mejor que se puede hacer sin intervención humana.

Una vez identificada la entidad de referencia y su tamaño en unidades CAD, calculamos
el factor `px_per_cad` dividiendo el tamaño objetivo en píxeles (que fijamos en 64 px
como sweet spot para YOLOv8) por el tamaño de la referencia. Ese factor se usa
después en el rasterizado para garantizar que los símbolos siempre tengan
aproximadamente el mismo tamaño visual, sin importar las unidades del DXF original.

Sumamos también una salvaguarda: si el factor calculado produce una imagen demasiado
grande para procesar (por ejemplo, un plano de planta arquitectónico completo que
necesitaría 32000 píxeles de lado), reducimos automáticamente la escala hasta entrar
en un techo configurable y logueamos un aviso de que los símbolos quedaron más chicos
que el target ideal.

---

## Por qué hace falta slicing

Una vez resuelto el problema de la escala, queda otro problema independiente: el modelo
de detección no puede procesar imágenes grandes directamente. Igual que con el
autoescalado, hubo varios problemas concretos que motivaron la solución.

### YOLO redimensiona todo a un tamaño fijo

YOLO tiene un input de tamaño fijo, típicamente 640 por 640 píxeles. Cuando le pasamos
una imagen, internamente la redimensiona antes de procesarla. Si la imagen original
mide 16000 por 13000 píxeles, queda reducida a 640 por 512. Los símbolos que medían 60
píxeles ahora miden 2.4 píxeles. La red no tiene resolución suficiente para reconocer
nada a ese tamaño.

### Los símbolos eléctricos son chicos comparados con el plano

Esto es un caso clásico de detección de objetos pequeños en imágenes grandes. La
proporción del símbolo respecto al plano completo puede ser de 1 a 1000 o más en
términos de área. La inferencia directa simplemente no funciona en esa escala.

### No podemos bajar la resolución del plano

Una alternativa ingenua sería rasterizar el plano a una imagen más chica para que
quepa cómodamente en el input de YOLO. Pero eso es exactamente el problema que
queríamos evitar: si reducimos la resolución, perdemos la capacidad de distinguir los
símbolos.

### Procesar el plano completo a alta resolución no es viable directamente

Aún si YOLO no redimensionara la entrada, una imagen de 200 megapíxeles tampoco cabría
en la memoria de la mayoría de las GPUs. Así que ni siquiera "procesar todo a la vez"
sería una opción técnica.

---

## Cómo solucionamos el slicing

La estrategia fue dividir el plano en pedazos del tamaño que YOLO espera, procesar
cada pedazo por separado, y unificar los resultados al final. Esta técnica se conoce
como slicing o tiled inference, y la implementamos con SAHI, una librería diseñada
específicamente para este caso de uso.

El plano renderizado se divide en tiles de 640 por 640 píxeles. Cada tile mantiene la
resolución original del plano, lo que significa que los símbolos dentro de cada tile
se ven a su tamaño natural y son perfectamente detectables por YOLO. La inferencia se
ejecuta tile por tile, y cada detección obtenida en coordenadas locales del tile se
traslada a coordenadas globales del plano completo sumándole el offset del tile.

Para que un símbolo que cae justo en el borde entre dos tiles no quede partido,
introducimos un solapamiento del 20 por ciento entre tiles vecinos. Cualquier símbolo
que caiga en una zona compartida queda completo en al menos uno de los dos tiles, así
que la detección siempre es posible.

El costo del solapamiento es que un mismo símbolo va a aparecer detectado dos veces
cuando esté en una zona compartida. Para resolverlo aplicamos NMS (Non-Max
Suppression): cuando dos cajas se solapan más de un cierto umbral, nos quedamos con la
de mayor confianza y descartamos la otra. Decidimos usar NMS clásico en lugar de la
variante de fusión (Greedy NMM), porque en pruebas iniciales esta última generaba
falsos positivos al combinar detecciones de tiles vecinos en cajas alargadas
artificiales.

Sumamos también un parámetro de zoom configurable. La idea es que el tamaño del slice
puede ajustarse para cubrir más o menos área por tile según las características del
plano. Slices más chicos producen más tiles con más detalle por símbolo (útil para
planos donde los símbolos son particularmente chicos), mientras que slices más grandes
producen menos tiles con menos detalle pero inferencia más rápida. Un parámetro
adicional permite definir la cantidad mínima de tiles por eje, calculando
automáticamente el slice apropiado en función del tamaño de la imagen.

---

## Cómo se complementan ambas soluciones

El autoescalado y el slicing resuelven problemas distintos, en momentos distintos del
pipeline, pero son complementarios. El autoescalado opera antes de rasterizar y
garantiza que cada símbolo tenga el tamaño correcto en píxeles. El slicing opera
después del rasterizado y garantiza que cada porción del plano se procese a su
resolución natural sin que YOLO la achique. Sin autoescalado, los tiles tendrían
símbolos de tamaños inconsistentes y el modelo no los reconocería de manera
confiable. Sin slicing, los símbolos correctamente escalados se perderían igualmente
al ser redimensionados por YOLO.

Ambas técnicas son independientes del modelo de detección que se use después.
Funcionan exactamente igual si en lugar de YOLO se utiliza SIFT, template matching
clásico, transformers de visión, o cualquier otro detector. Esto es importante porque
mantiene el sistema modular y reutilizable: los componentes de autoescalado y
rasterizado pueden trasladarse a otros proyectos que necesiten convertir DXFs a
imágenes con escala consistente, sin arrastrar consigo las dependencias específicas
del flujo YOLO.

---

## Trazabilidad píxel a CAD

Una consecuencia importante del autoescalado es que las detecciones se producen en
coordenadas píxel de la imagen rasterizada, pero el usuario final trabaja en su
software de diseño con coordenadas CAD reales. Para cerrar el loop, durante el
rasterizado guardamos en un archivo de metadatos toda la información necesaria para
hacer la conversión inversa: el factor de escala usado, el origen del sistema de
coordenadas CAD respecto a la imagen, y las dimensiones de la imagen rasterizada.

Con esa metadata, cualquier coordenada píxel obtenida por el detector puede
convertirse de vuelta a coordenadas CAD. Esto permite que las detecciones sean
útiles dentro del flujo de trabajo real del usuario: pueden marcarse sobre el DXF
original, exportarse a otros sistemas, medirse en unidades reales, o usarse para
generar reportes accionables.

---

## Mejoras posibles

La solución actual cubre los casos típicos pero deja margen para perfeccionarse en
varias direcciones.

### En el autoescalado

- Combinar múltiples entidades de referencia (INSERT, círculo, texto) en lugar de
  quedarse con la primera disponible, ponderando cada una según su confiabilidad.
- Validar el factor calculado pre-rasterizando una región de muestra para confirmar
  que los símbolos efectivamente caen en el rango de tamaño esperado, antes de
  procesar el plano completo.
- Hacer configurable por proyecto la lista de capas a excluir (cajetines, marcos,
  viewports), ya que las convenciones de nombres varían entre estudios.
- Detectar planos con escalas mixtas, como detalles ampliados en una esquina, donde
  un único factor global no es óptimo. Idealmente, identificar las regiones a escala
  distinta y procesarlas por separado.
- Adaptar el `target_px` automáticamente a la distribución de tamaños del dataset de
  entrenamiento del modelo, en lugar de fijarlo manualmente.

### En el slicing

- Saltear tiles vacíos para acelerar el procesamiento de planos extensos con grandes
  zonas en blanco, donde la inferencia es desperdicio puro.
- Implementar slicing adaptativo donde el tamaño del tile varíe según la densidad
  local de contenido (tiles más chicos en zonas densas, más grandes en zonas con
  pocos elementos).
- Inferencia multi-escala que combine resultados de varios niveles de zoom para
  capturar tanto símbolos pequeños como variantes de tamaño atípico (multipolares,
  símbolos especiales).
- Paralelizar la inferencia entre tiles, que actualmente se procesa de manera
  secuencial, para reducir tiempos en planos grandes.
- Implementar render por franjas en el rasterizado de planos muy grandes, evitando
  cargar la imagen completa en memoria y permitiendo procesar planos arbitrariamente
  grandes sin un techo fijo de dimensiones.

---

## Resumen

El autoescalado responde a la realidad de que los DXFs profesionales son
heterogéneos en unidades, escalas y convenciones, y que un sistema robusto no puede
depender de configuración manual por archivo. La solución fue identificar
automáticamente una entidad de referencia dentro del propio DXF y usar su tamaño
para calcular la escala apropiada.

El slicing responde a una limitación técnica de los modelos de detección modernos:
no pueden procesar imágenes arbitrariamente grandes manteniendo la resolución de los
objetos pequeños. La solución fue dividir el plano en pedazos digestibles, procesar
cada uno por separado, y unificar los resultados con técnicas de deduplicación.

Las dos soluciones combinadas convierten un problema que parecía intratable
(detectar símbolos eléctricos en DXFs heterogéneos y de gran tamaño) en un pipeline
que funciona de manera automática, sin configuración por archivo, y de manera
independiente del modelo de detección elegido.
