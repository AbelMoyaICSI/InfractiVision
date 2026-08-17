# Extracted Thesis Content

UNIVERSIDAD PRIVADA ANTENOR ORREGO

FACULTAD DE INGENIERÍA

PROGRAMA DE ESTUDIO DE INGENIERÍA DE COMPUTACIÓN Y SISTEMAS

TESIS PARA OPTAR EL TÍTULO PROFESIONAL DE

INGENIERO DE COMPUTACIÓN Y SISTEMAS

Visión computacional de cruces en rojo para mejorar el proceso de registro de infracciones de tránsito en Trujillo 2025

Línea de Investigación:

Sistemas de Información

Jurado evaluador:Presidente:Secretario:Vocal:Jurado evaluador:Presidente:Secretario:Vocal:Autores:

Jurado evaluador:

Presidente:

Secretario:

Vocal:

Jurado evaluador:

Presidente:

Secretario:

Vocal:

Moya Acosta, Abel Jesús

Guerrero Belevan, Christopeer Eliott

Asesor:

Dr. Cieza Mostacero, Segundo Edwin

Código ORCID: https://orcid.org/0000-0002-3520-4383 

TRUJILLO - PERÚ

2025

Fecha de sustentación: 

Resumen

El registro manual de infracciones de tránsito en Trujillo presenta deficiencias operativas, demoras y errores derivados de la dependencia exclusiva del criterio humano. Este estudio propone el uso de visión computacional para automatizar el proceso de detección y registro de infracciones por cruce en luz roja (M.17), mejorando la precisión, rapidez y fiabilidad de los registros. El sistema desarrollado integra cámaras semafóricas, el modelo YOLOv8 para la detección de vehículos y estados del semáforo, y el uso de OCR para el reconocimiento de matrículas. La investigación adopta un enfoque cuantitativo-experimental con grupos control y experimental en tres intersecciones críticas de la ciudad, evaluando tres indicadores clave: tasa de infracciones correctamente detectadas, tiempo promedio de registro y número total de infracciones registradas. Los resultados demuestran una mejora significativa en la eficiencia del proceso, evidenciando que la automatización reduce los errores humanos y los tiempos de respuesta, fortaleciendo la transparencia y confiabilidad del sistema de control de tránsito. Se concluye que la implementación de visión computacional constituye una herramienta tecnológica viable para optimizar la gestión vial urbana y contribuir a la seguridad ciudadana.

Palabras clave: visión computacional; detección de infracciones; semáforo en rojo; reconocimiento de matrículas; tránsito urbano. 

Abstract

The manual registration of traffic violations in Trujillo shows operational deficiencies, delays, and errors caused by human dependency. This study proposes the use of computer vision to automate the detection and registration of red-light traffic violations (M.17), improving accuracy, speed, and reliability. The developed system integrates traffic cameras, the YOLOv8 model for vehicle and signal detection, and OCR for license plate recognition. The research follows a quantitative-experimental design with control and experimental groups in three critical intersections, evaluating key indicators: correctly detected violations rate, average registration time, and total number of recorded violations. Results show a significant improvement in process efficiency, reducing human errors and response time while enhancing transparency and reliability. The implementation of computer vision proves to be a viable technological tool to optimize urban traffic management and strengthen road safety.

Keywords: computer vision; traffic violation detection; red-light crossing; license plate recognition; urban mobility.

Presentación

Señores miembros del jurado:

De acuerdo con el cumplimiento de las disposiciones del reglamento de grados y títulos de la Universidad Privada Antenor Orrego, exponemos a vuestra consideración la tesis titulada: Visión computacional de cruces en rojo para mejorar el proceso de registro de infracciones de tránsito en Trujillo.

Desarrollado con el fin de obtener el título de Ingeniero de computación y sistemas. El objetivo principal es mejorar el proceso de registro de infracciones de tránsito en Trujillo, a través del uso de visión computacional durante el primer semestre del 2025.

A ustedes miembros del jurado, mostramos nuestro especial y mayor reconocimiento por el dictamen que se haga merecedor y correspondiente del presente trabajo.

Abel Jesús Moya Acosta

Guerrero Belevan Christopeer

DNI: 73146770

ORCID: 0009-0009-1484-3301

DNI: 74130075

ORCID: 0009-0004-2493-7698

Índice de contenidos

Resumenii

Abstractiii

Presentacióniv

Índice de contenidosv

Índice de tablasvii

Índice de figurasvii

1.Introducción8

1.1.Contexto y antecedentes8

1.2.Descripción y alcance del estudio10

2.Planteamiento del problema de investigación11

2.1.Descripción y delimitación del problema11

2.1.1.Formulación del problema11

2.1.2.Problema central del estudio11

2.2.Objetivos de la investigación13

2.2.1.Objetivo general13

2.2.2.Objetivos específicos13

2.3.Importancia del estudio13

2.4.Justificación de la investigación14

2.5.Limitaciones del estudio15

3.Marco teórico16

3.1.Marco histórico16

3.2.Investigaciones antecedentes relacionadas con el tema17

3.3.Base teórica – científica19

3.4.Definición de términos básicos28

4.Hipótesis y variables29

4.1.Supuestos básicos29

4.2.Hipótesis29

4.3.Variables30

4.4.Matriz de consistencia31

5.Marco metodológico33

5.1.Tipo de investigación33

5.2.Nivel de madurez tecnológica33

5.3.Método de investigación33

5.4.Diseño del estudio33

5.5.Población y muestra34

5.6.Técnicas e instrumentos de recolección de datos35

5.7.Procedimientos de ejecución del estudio35

5.8.Técnicas de procesamiento y análisis de datos36

6.Presentación de resultados37

6.1.Resultados de la investigación37

9.Recomendaciones60

10.Referencias bibliográficas61

Índice de tablas

Tabla 1. Operacionalización de variables.30

Tabla 2. Matriz de consistencia de la investigación.31

Tabla 3. Resultados de la posprueba para el indicador Número de Infracciones Detectadas (NID).37

Tabla 4. Estadísticos descriptivos para el indicador Número de Infracciones Detectadas (NID).38

Tabla 5. Resultados de la posprueba para el indicador Tiempo de Registro (TR).39

Tabla 6. Estadísticos descriptivos para el indicador Tiempo de Registro (TR).40

Tabla 7.Resultados de la posprueba para el indicador Tasa de Infracciones detectadas (TI).41

Tabla 8. Estadísticos descriptivos para el indicador Tasa de Infracciones detectadas (TI).42

Tabla 9. Test de normalidad de Shapiro-Wilk para NIDGC.43

Tabla 10. Test de normalidad de Shapiro-Wilk para NIDGE.45

Tabla 11. Estadístico de U de Mann-Whitney para NID.46

Tabla 12.  Test de normalidad de Shapiro-Wilk para TRGC.47

Tabla 13. Test de normalidad de Shapiro-Wilk para TRGE.48

Tabla 14. Estadístico de U de Mann-Whitney para TR.50

Tabla 15. Test de normalidad de Shapiro-Wilk para TIGC.51

Tabla 16. Test de normalidad de Shapiro-Wilk paraTIGE.52

Tabla 17. Estadístico de U de Mann-Whitney para NV.54

Índice de figuras

Figura 1. Diseño con posprueba únicamente y grupo de control.35

Figura 2. Histograma de la normalidad de los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo de Control (NIDGC).45

Figura 3. Histograma de la normalidad de los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo Experimental (NIDGE).46

Figura 4. Histograma de la normalidad de los datos del indicador Tiempo de Registro de la posprueba del Grupo de Control (TRGC).49

Figura 5. Histograma de la normalidad de los datos del indicador Número de Ventas de la posprueba del Grupo Experimental (NVGE).50

Figura 6. Histograma de la normalidad de los datos del indicador Tasa de Infracciones de la posprueba del Grupo de Control (TIGC).53

Figura 7. Histograma de la normalidad de los datos del indicador Tasa de Infracciones de la posprueba del Grupo Experimental (TIGE).54

Introducción

Contexto y antecedentes

En muchos países, el registro manual de infracciones de tránsito por agentes de policiales, basado en observación presencial y emisión de papeleta, presenta tiempos de respuesta irregulares, vulnerabilidad a errores frecuentes por distintos factores y cobertura limitada; sin protocolos estandarizados ni herramientas de apoyo, cada oficial aplica criterios propios, dificultando la obtención de datos precisos y oportunos.

En el panorama global, en Canadá, Yasanthi et al. (2024) analizaron el comportamiento de los agentes de la ley en la aplicación de la vigilancia del tráfico (TSE) para la seguridad vial; más tarde, se revelaron algunos factores como los sobornos a ciudadanos que afectaron el proceso de detención y emisión de multas. Además, la presión para cumplir con un mínimo de multas diarias o semanales en lugar de sancionar infracciones de mayor riesgo. La OMS (2023, citado por Yasanthi et al., 2024), también afirmó que  para muchos países, la TSE hizo poco para mejorar la seguridad vial.

En Brasil, Ang et al. (2020) evidenciaron que existe una muy pequeña cobertura de observación para los agentes policiales de tránsito al examinar las sanciones por infracciones de tránsito por abuso de velocidad en São Paulo y se encontró que la supervisión humana representa menos del 8% del rendimiento de dos a tres millones de multas, según la orden de restricción emitida de acuerdo con los números de matrícula, mientras que de cinco a seis millones de multas emitidas por exceso de velocidad están automatizadas.

En el ámbito nacional, en la ciudad de Chiclayo, Tejada (2024) identificó una deficiente gestión de multas de tránsito al analizar integralmente el proceso operativo y administrativo entre los años 2015 y 2019. El estudio evaluó tareas de los responsables y diversos indicadores como el tiempo de registro de papeletas en los sistemas respectivos, el número de multas impuestas y el nivel de cobranza, evidenciando importantes ineficiencias en el sistema local de sanciones.

En Trujillo (La Libertad, Perú), Flores y García (2024) analizaron cuales fueron las consecuencias al no cumplir las normas de tránsito por cometer exceso de velocidad; por ello, entrevistaron a distintos tipos de conductores, donde se destacó la necesidad de que los policías cumplan su trabajo operativo de detectar este tipo de infracciones y poder sancionarlas. Sin embargo, se pudo observar que algunos de ellos no desempeñaban esta función debido a la corrupción u omisión de multas, para sus propios beneficios.

El problema central de este estudio es la ineficiencia del proceso manual de registro de infracciones por pasarse el semáforo en rojo (M17), realizado únicamente por los policías de tránsito, el cual presenta tiempos de gestión muy variables sin un estándar que garantice su uniformidad. Esta variabilidad, limita la capacidad de capturar todas las infracciones y deja vacíos de cobertura en las intersecciones más críticas.

La causa principal se debe al uso exclusivo de un procedimiento manual, sin protocolos estandarizados ni herramientas de apoyo o ética, hace que cada agente registre las infracciones a su ritmo y criterio personales, lo que genera demoras, omisiones y errores en el llenado de las papeletas.

Esta falta de uniformidad provoca ineficiencias operativas (duplicación de tareas, pérdida de tiempo), fomenta la impunidad (multas no emitidas ni cobradas) y aumenta el riesgo vial (conductores reincidentes sin registro).

Para poder cuantificar estas deficiencias se analizará la tasa de infracciones de cruces en rojo, el tiempo de registro de cada infracción de cruces en rojo y la reincidencia de la infracción diaria, entendida como la proporción de infracciones observadas que finalmente se documentan en el registro de papeletas. Esta medición conjunta permitirá identificar cuellos de botella, evaluar la precisión operativa y determinar hasta qué punto el proceso manual logra capturar todos los incidentes de tránsito.

Por las razones antes expuestas, se plantea implementar la Visión Computacional de cruces en rojo (VI) para mejorar el proceso de registro de infracciones de tránsito (VD), garantizando registros más rápidos, completos y fiables.

Descripción y alcance del estudio

Esta investigación pretende mejorar el proceso de registro de infracciones de tránsito en la ciudad de Trujillo durante el año 2025, mediante la implementación de un sistema basado en visión computacional de cruces en rojo, que automatice la detección y registro de vehículos que incumplen la señal luminosa del semáforo. Para alcanzar este objetivo general, se plantea: (1) aumentar la tasa de infracciones correctamente detectadas, (2) disminuir el tiempo promedio de registro por evento, y (3) aumentar el número total de infracciones detectadas. El estudio es de tipo aplicado, dado que busca resolver un problema real mediante la integración de tecnologías de procesamiento de imágenes. Asimismo, su diseño es experimental puro, porque se manipula la variable independiente en un grupo experimental y se compara con un grupo control que mantiene el método manual tradicional, evaluando así la influencia de la automatización sobre la precisión y rapidez del proceso de registro de infracciones.

Planteamiento del problema de investigación

Descripción y delimitación del problema

Formulación del problema

¿De qué manera el uso de visión computacional de cruces en rojo influye en el proceso de registro de infracciones de tránsito de Trujillo en 2025?

Problema central del estudio

En muchos países, el registro manual de infracciones de tránsito por agentes de policiales, basado en observación presencial y emisión de papeleta, presenta tiempos de respuesta irregulares, vulnerabilidad a errores frecuentes por distintos factores y cobertura limitada; sin protocolos estandarizados ni herramientas de apoyo, cada oficial aplica criterios propios, dificultando la obtención de datos precisos y oportunos.

En el panorama global, en Canadá, Yasanthi et al. (2024) analizaron el comportamiento de los agentes de la ley en la aplicación de la vigilancia del tráfico (TSE) para la seguridad vial; más tarde, se revelaron algunos factores como los sobornos a ciudadanos que afectaron el proceso de detención y emisión de multas. Además, la presión para cumplir con un mínimo de multas diarias o semanales en lugar de sancionar infracciones de mayor riesgo. La OMS (2023, citado por Yasanthi et al., 2024), también afirmó que  para muchos países, la TSE hizo poco para mejorar la seguridad vial.

En Brasil, Ang et al. (2020) evidenciaron que existe una muy pequeña cobertura de observación para los agentes policiales de tránsito al examinar las sanciones por infracciones de tránsito por exceso de velocidad en São Paulo y se encontró que la supervisión humana representa menos del 8% del rendimiento de dos a tres millones de multas, según la orden de restricción emitida de acuerdo con los números de matrícula, mientras que de cinco a seis millones de multas emitidas por exceso de velocidad están automatizadas.

En el ámbito nacional, en la ciudad de Chiclayo, Tejada (2024) identificó una deficiente gestión de multas de tránsito al analizar integralmente el proceso operativo y administrativo entre los años 2015 y 2019. El estudio evaluó tareas de los responsables y diversos indicadores como el tiempo de registro de papeletas en los sistemas respectivos, el número de multas impuestas y el nivel de cobranza, evidenciando importantes ineficiencias en el sistema local de sanciones.

En Trujillo (La Libertad, Perú), Flores y García (2024) analizaron cuales fueron las consecuencias al no cumplir las normas de tránsito por cometer exceso de velocidad; por ello, entrevistaron a distintos tipos de conductores, donde se destacó la necesidad de que los policías cumplan su trabajo operativo de detectar este tipo de infracciones y poder sancionarlas. Sin embargo, se pudo observar que algunos de ellos no desempeñaban esta función debido a la corrupción u omisión de multas, para sus propios beneficios.

El problema central de este estudio es el proceso manual de registro de infracciones por pasarse el semáforo en rojo (M17), realizado únicamente por los policías de tránsito, el cual presenta tiempos de gestión muy variables sin un estándar que garantice su uniformidad. Esta variabilidad, limita la capacidad de capturar todas las infracciones y deja vacíos de cobertura en las intersecciones más críticas.

La causa principal se debe al uso exclusivo de un procedimiento manual, sin protocolos estandarizados ni herramientas de apoyo o ética, hace que cada agente registre las infracciones a su ritmo y criterio personales, lo que genera demoras, omisiones y errores en el llenado de las papeletas.

Esta falta de uniformidad provoca ineficiencias operativas (duplicación de tareas, pérdida de tiempo), fomenta la impunidad (multas no emitidas ni cobradas) y aumenta el riesgo vial (conductores reincidentes sin registro).

Para poder cuantificar estas deficiencias se analizará la tasa de infracciones de cruces en rojo, el tiempo de registro de cada infracción de cruces en rojo y la reincidencia de la infracción diaria, entendida como la proporción de infracciones observadas que finalmente se documentan en el registro de papeletas. Esta medición conjunta permitirá identificar cuellos de botella, evaluar la precisión operativa y determinar hasta qué punto el proceso manual logra capturar todos los incidentes de tránsito.

Por las razones antes expuestas, se plantea implementar la Visión Computacional de cruces en rojo (VI) para mejorar el proceso de registro de infracciones de tránsito (VD), garantizando registros más rápidos, completos y fiables.

Objetivos de la investigación

Objetivo general

Mejorar el proceso de registro de infracciones de tránsito en Trujillo, a través del uso de visión computacional de cruces en rojo en 2025.

Objetivos específicos

Aumentar la tasa de infracciones correctamente detectadas de cruces en rojo.

Disminuir el tiempo de registro de cada infracción de cruce en rojo. 

Aumentar el número de infracciones detectadas de cruces en rojo. 

Importancia del estudio

La investigación adquiere importancia al abordar los desafíos del registro de infracciones, derivado por demoras, omisiones y falta de cobertura en intersecciones críticas, que obligan a los policías de tránsito a depender de procesos posteriores de verificación. Frente a este panorama, el desarrollo de visión computacional de cruces en rojo representa una contribución clave para fortalecer la tasa de infracciones detectadas, reducir el tiempo de registro de infracciones y aumentar el número de infracciones detectadas.

Justificación de la investigación

Justificación teórica: La investigación se justifica de forma teórica porque utiliza como base para su ejecución los conceptos de visión computacional mencionado por Elgendy (2020, p. 4); referentes al reconocimiento de objetos mediante redes neuronales que extraen representaciones de alto nivel, y que, al concluir el estudio, permitirán confirmar la eficacia de estos en la detección de infracciones.

Justificación práctica: La investigación se justifica de forma práctica porque se busca solucionar el problema referido al proceso de registro de infracciones de tránsito en la Unidad de Tránsito y Seguridad Vial de la Policía Nacional del Perú en Trujillo, beneficiando directamente a los agentes de tránsito encargados del registro al reducir tiempos de atención y errores manuales. Asimismo, el estudio aporta al Objetivo de Desarrollo Sostenible (ODS) 7 “Energía asequible y no contaminante” al minimizar la demanda eléctrica y aprovechar la infraestructura existente.

Justificación metodológica: La investigación se justifica de forma metodológica porque para la recolección de datos se emplearán fichas de registro respondidas por los policías de tránsito de Trujillo respecto a nuestros indicadores. Los indicadores clave se miden con instrumentos adaptados de la literatura: la Tasa de infracciones detectadas (TI) se cuantifica siguiendo el protocolo de Pradhan et al. (2025) el Tiempo de Registro (TR) se registra según la metodología de Agarwal et al. (2018) y número de infracciones detectadas (NID) se calcula conforme a (Lu et al., 2025). Finalmente, los datos se analizaron estadísticamente para comparar el desempeño del método automatizado frente al método manual, garantizando la validez de los resultados.

Justificación social: La investigación se justifica de forma social porque al mejorar los indicadores clave tales como: Tasa de infracciones detectadas (TI), Tiempo de Registro (TR) y el número de infracciones detectadas (NID) permiten sancionar oportunamente y facilitan la identificación de puntos críticos de reincidencia. En conjunto, estos avances fortalecen la confianza ciudadana en las autoridades de tránsito, fomentan hábitos de conducción responsables y contribuyen a un entorno urbano más equitativo y seguro.

Limitaciones del estudio

Una de las tantas limitaciones radica en el alcance geográfico y técnico: el análisis se circunscribe a tres intersecciones críticas de Trujillo, por lo que sus resultados podrían no representar la realidad de todo el sistema vial municipal. La calidad de las grabaciones es puede ser afectada por variaciones de iluminación puede reducir la precisión del reconocimiento de matrículas, mientras que la disponibilidad de solo tres copias de respaldo de video, debido al gran tamaño de los archivos, limita la diversidad y robustez del conjunto de datos de entrenamiento.

Otra limitación es que existe escasa bibliografía sobre la variable dependiente a nivel local lo que dificulta el respaldo comparativo y la fundamentación académica de resultados. Asimismo, la disponibilidad de policías de tránsito para supervisar y validar manualmente las infracciones automatizadas es reducida, su rutina y capacitación están orientadas al patrullaje tradicional.

Marco teórico

Marco histórico

La visión computacional comenzó a delinearse en las décadas de 1960 y 1970 cuando diversos enfoques para el análisis digital de imágenes y fotografías se entendían como tareas de reconocimiento de patrones, análisis de escenas o comprensión visual, procesos que compartían el mismo núcleo: la detección de objetos representados en una imagen; con el paso del tiempo dos factores resultaron decisivos para transformar el rumbo de esta disciplina, por un lado la creciente dificultad de interpretar fotografías digitales que exigía algoritmos más especializados capaces de extraer información significativa y por otro el desarrollo de modelos computacionales inspirados en la visión biológica; hacia finales de los años ochenta esta área de investigación alcanzó una madurez conceptual suficiente para que diferentes investigadores coincidieran en adoptar de manera generalizada el término “computer vision” para referirse al campo especializado que integraba estos avances  (Dobson, 2023, p. 39-40).

Desde la década de 1990 hasta la actualidad, la visión computacional ha experimentado una evolución acelerada gracias al desarrollo de la detección de objetos, cuyo avance ha tenido un impacto profundo en todo el campo y ha sido objeto de una revisión técnica que abarca más de un cuarto de siglo; este recorrido histórico se reconoce comúnmente en dos grandes períodos: una primera etapa tradicional, anterior a 2014, basada en descriptores manuales y algoritmos clásicos, y una segunda fase, posterior a ese año, marcada por la incorporación del aprendizaje profundo, la cual permitió alcanzar niveles de precisión y velocidad sin precedentes, consolidando así a la visión computacional como un área clave para aplicaciones prácticas en contextos modernos como el transporte y la seguridad vial (Zou et al., 2023).

En la etapa más reciente, la visión computacional se ha consolidado como una disciplina aplicada en numerosos escenarios del mundo real, donde su alcance va desde el reconocimiento óptico de caracteres y de matrículas hasta la inspección automatizada de calidad en la industria, el comercio minorista con sistemas de autoservicio, la logística con robots y vehículos autónomos, el análisis de imágenes médicas y la generación de modelos tridimensionales a partir de fotografías aéreas o de drones; estas aplicaciones demuestran cómo la evolución histórica del campo ha desembocado en un conjunto de técnicas maduras que hoy son parte esencial del desarrollo tecnológico en áreas críticas como el transporte, la salud y la seguridad (Szeliski, 2022, p. 5).

Investigaciones antecedentes relacionadas con el tema

En primer plano, Pradhan et al. (2025) desarrollaron un sistema de estacionamiento inteligente con el objetivo de mejorar la eficiencia y fiabilidad, incluso en situaciones de poca iluminación. El estudio experimental resultó que la detección de las matrículas tuvo una tasa de precisión del 95 % en iluminación diurna y del 90 % en ambientes de baja luminosidad. Concluyeron que los resultados consolidaron la eficacia del uso de esta tecnología y amplió las posibilidades de aplicaciones, como en las autopistas, para las tarifas en la demanda y las condiciones de tráfico.

Por otro lado, Correa y Vílchez (2022) realizaron un estudio con diseño preexperimental mediante un enfoque cuantitativo, cuyo objetivo fue desarrollar un módulo de un cinemómetro en el sistema web SITRAN para optimizar la gestión  de infracciones de la SUTRAN; por ello, buscaron reducir específicamente el tiempo de registro de las infracciones y aumentar la importación de evidencia fotográfica del dispositivo. Los resultados mostraron una reducción del 50,32% con una media de 0.50 minutos (30s), durante el tiempo de registro y un aumento del 309,29% en las importaciones que presentan evidencias fotográficas, mostrando una mejora significativa en la efectividad en la gestión de infracciones de tránsito.

Además, Lu et al. (2025) desarrollaron AdvFuzz, una herramienta de simulación enfocada en probar vehículos autónomos, el objetivo fue incrementar el número de violaciones detectadas durante las pruebas de conducción simulada, los resultados mostraron que en 12 horas de simulación el sistema logró identificar 540 violaciones, frente a un promedio de 181 violaciones detectadas por otros cuatro métodos comparativos. Se concluye que este modelo evidencia una mejora significativa en la capacidad de detección y cuantificación de infracciones.

Consecuentemente, Owais et al. (2025) propusieron un marco basado en antecedentes sobre el uso de establecimiento de un sistema de monitoreo inteligente para la ciudad de Assiut, Egipto, mediante cámaras con inteligencia artificial (IA); cuyo objetivo fue identificar las tasas de infracciones diarias y aumentar la cobertura 24/7. Los resultados esperados incluyeron mejorar la precisión a un mínimo del 95 %, aumentar la tasa de detección en un 50 % por cada 1000 vehículos respecto al monitoreo manual y reducir el tiempo de respuesta a 2 segundos, al haber analizado los estudios concluyeron que lo mejor sería un establecimiento estándar para el sector transporte en el diseño de ciudades inteligentes sostenibles y dejar el sistema como una solución abierta al desarrollo del tráfico urbano.

Por otra parte, Thao et al. (2022) desarrollaron un sistema de detección de infracciones de semáforo en rojo aplicando redes neuronales convolucionales mediante el modelo YOLOv5; el estudio tuvo como objetivo diferenciar de manera automática tanto a los vehículos como al estado de la luz de tráfico, utilizando un conjunto de datos de imágenes de intersecciones. Los resultados alcanzaron una precisión cercana al 82 % en la identificación de vehículos y al 90 % en la detección del color del semáforo, lo que permitió registrar infracciones con un 86 % de exactitud. Los autores concluyeron que la integración de algoritmos de aprendizaje profundo en entornos urbanos contribuye a mejorar el control automatizado del tránsito y a generar evidencias confiables para la sanción de infractores, aspecto directamente relacionado con la tasa de infracciones detectadas en esta investigación. 

Ren (2024) presentó un sistema inteligente de detección de infracciones vehiculares bajo un enfoque de interacción humano–computador y visión por computadora, cuyo propósito fue resolver los problemas de lentitud e inestabilidad del registro manual de infracciones, la investigación fue de tipo aplicada y utilizó como muestra el conjunto de datos BIT Vehicle, complementado con técnicas de filtrado de Kalman y una interfaz de usuario optimizada para el seguimiento y gestión de vehículos. El sistema logró detectar ocho tipos de infracciones con una precisión superior al 96,8 % y, lo más importante, registrar cada evento en tiempo real, reduciendo significativamente la demora en el proceso, los autores concluyeron que este modelo incrementa la eficiencia de la gestión del tránsito y demuestra que la automatización permite registrar infracciones de manera más rápida en comparación con el método manual. 

De manera complementaria, González y Prada (2016) analizaron el efecto de la instalación de 19 cámaras en intersecciones de Cali durante el período 2010–2013. El estudio comparó los puntos intervenidos con un grupo de control mediante técnicas de emparejamiento y diferencias en diferencias. Los hallazgos revelaron que, tras la instalación, los accidentes no disminuyeron de forma significativa; incluso, en las primeras intersecciones tratadas se observó un aumento en choques leves. Los autores concluyen que el programa no se asignó con base en criterios de accidentalidad, lo cual limitó su efectividad en términos de seguridad vial.

Base teórica – científica

Visión Computacional 

La Visión computacional es aquella ciencia que busca comprender el entorno que rodea al ser humano, bajo la percepción visual por medio de imágenes y videos, los cuales están basados en un modelo físico para generar mediante sistemas de IA las decisiones más adecuadas; por ello, se señaló que para los humanos la visión es una percepción de los sentidos, mientras que los sistemas IA buscan acercarse a entender esa percepción, dependiendo de un recurso externo que se use (Elgendy, 2020, p. 4).

Por otro lado, Khan, Laghari y Awan (2021) mencionaron que la visión artificial desde el punto de vista del uso de machine learning (ML), se retrata como la extracción de información vital por medio de imágenes digitales a través de modelos computacionales.

Además, Thoma (2017) mencionó que esta rama de busca a través de información muy simple poder comprimir una información de alto nivel, transformando esa información compleja obtenida en conocimiento significativo visual, sin la necesidad de tener que utilizar imágenes mediante un procesamiento previo. 

La visión computacional ha sido estudiada desde múltiples puntos de vista sobre las técnicas que han llevado a usarse en múltiples campos y áreas, en los cuales se extienden inicialmente en numerosos datos sin procesar hasta diversas técnicas e ideas como la identificación de patrones, machine learning, gráficos computacionales y sobre todo el procesamiento de imágenes (Wiley y Lucas, 2018).

Una de las técnicas más destacadas en visión computacional es el procesamiento de imágenes. El cual es un aspecto fundamental del campo del ML, mayormente usado en la visión computacional y deep learning (DL) para poder obtener y realizar cambios de información crucial, útil y accesible, en el que se puede tener una mejora en el rendimiento de los modelos (Upadhyay y Gupta, 2024).

Para Archana y Jeevaraj (2024), el procesamiento de imágenes es un área multidisciplinar que engloba, dentro de ésta, distintas técnicas como la supresión de ruido, la mejora, la segmentación y la clasificación de las imágenes, con el fin de obtener datos relevantes de imágenes digitalizadas.

Dentro del ámbito de visión computacional y procesamiento de imágenes, se encuentra la técnica de la segmentación de imágenes, y tiene como función esencial, dividir en segmentos las imágenes, simplificando la comprensión. Esta segmentación es necesaria dentro de muchas áreas debido a que el objetivo es tener como resultado la división en base a las características y el color (Albukhnefis, Fatlawi y Al-Alsaeedi, 2024).

Por otro lado Minaee et al. (2021) mencionaron también que la segmentación es un aspecto esencial en el procesamiento de imágenes y visión computacional, debido los diversos usos que tiene como la compresión y análisis de imágenes médicas, en la realidad, en la robótica y entre otra más, la cual ha ido evolucionando y mejorando el uso del rendimiento a través los modelos más recientes de DL.

A su vez, se resaltan a las características como partes fundamentales del procesamiento de imágenes, las cuales actúan como las propiedades de las imágenes y la obtención de éstas permite las principales actividades como el reconocimiento de objetos, clasificación, entre otras. Para poder obtener dichas características es necesaria una técnica de extracción, con el propósito de conseguir la información más importante de las propiedades que poseen estas imágenes como el borde, color, las esquinas, vértice formas y otros elementos más (Upadhyay y Gupta, 2024).

Según Rashed y Popescu (2022), la técnica de extracción de características es el proceso que consiste en identificar y recuperar la información de los datos más relevantes que aún no han sido procesados. Esta fase es fundamental en el procesamiento de imágenes, este permite mejorar la apariencia del contenido al reducir su dimensionalidad, seleccionar los elementos distintivos y transformar los datos de entrada en un conjunto de atributos empleados para tareas de clasificación.

Luego se encuentra la técnica de clasificación de imágenes, también denominada reconocimiento de imágenes, es la tarea mediante la cual los sistemas computacionales identifican y etiquetan automáticamente los elementos y la temática de una imagen, reconociéndola de una forma similar al proceso como lo hace un humano y capturando detalles más allá de la percepción de un observador para asignarla a categorías predefinidas (Yadav y Sawale, 2023).

Según Singh y Singh (2020), la clasificación de imágenes ha llamado mucho la atención debido a su importancia en la visión computacional. Esta técnica busca clasificar una imagen la cual está conformada por una entrada, en base al contenido visual, señalando al comienzo con las personas que categorizan las imágenes manualmente mediante clasificadores; sin embargo, esto se complica cuando se tratan con cantidades numerosas, es por ello que se ha optado por el uso de DL para abordar esta gran cobertura.

Chen et al. (2021) señalaron que, el gran atractivo de esta técnica a nivel mundial ha impulsado su desarrollo a formas más modernas para poder clasificar las imágenes; por ello, con el surgimiento del DL, la implementación de modelos de redes neurales convolucionales, se han consolidado desde su creación en el 2012 como uno de los algoritmos más notables para clasificación de imágenes. Dicha arquitectura también es utilizada en otras actividades como el reconocimiento visual.

El modelo de CNN es una arquitectura de red neuronal de avance-propagación en la que cada neurona solo conecta con un pequeño vecindario de la capa anterior, en lugar de hacerlo con todas las neuronas. Mediante la alternancia de capas convolucionales seguidas de una o más capas, se obtiene un modelo profundo capaz de aprender jerarquías de representación directamente de las imágenes (Qiao, 2023).

Con el rápido ascenso que tuvo el DL y la notoria herramienta que lo hacía destacar, refiriéndose  a las CNN, las cuales se encuentran arraigadas a arquitecturas generales que mejoran la precisión para detectar objetos, donde previamente existían formas más convencionales para la detección de objetos en los que inicialmente se basaban en características autoajustables y arquitecturas simples, pero también tenían el objetivo de identificar y distinguir cada uno de los elementos existentes dentro de una imagen, estableciéndolos en secciones cuadriculadas con forma rectangular para evidenciar la existencia que poseen (Zhao et al., 2019).

Esta detección es parte de la visión computacional, el cual se centra en detectar, analizar y clasificar objetos específicamente en imágenes y vídeos, este campo tiene como finalidad desarrollar algoritmos eficientes y un rendimiento fiable, lo que los vuelve ideales y preparados para afrontar desafíos (Tsirtsakis et al., 2025).

Por otro lado, Kang et al. (2022) mencionó que la detección de objetos ofrece información muy importante en un entorno real; sin embargo, aún existen limitaciones, como la cantidad de datos, fallos en la resolución de imágenes y fondos que presentan una complejidad considerable. Debido a esto, algunos estudios han presentado un rendimiento de detección bajo de manera ineficaz por causa de estas dificultades.

Además, otra de las técnicas que ocupan una labor importante en la visión computacional es la estimación de flujo óptico, las cuales brindan información de datos en movimiento de bajo nivel; su objetivo consiste en entender el movimiento a través de los píxeles dentro de un conjunto secuencial de fotogramas. Esta técnica calcula cómo se ve el movimiento 3D en dos dimensiones proyectados por la cámara, analizando como se realizan los cambios de los píxeles entre un fotograma y el siguiente (Alfarano et al., 2024).

Mientras que, desde un punto de vista más formal “el método de flujo óptico es un enfoque diferencial basado en cálculos de la derivada temporal y el gradiente espacial de los campos de intensidad de la imagen, que requiere que los desplazamientos sean menores que la escala de longitud característica de las características tratables en el plano de la imagen” (Liu y Salazar, 2021).

La visión computacional está expuesta a una evolución constante al igual que su crecimiento en diversas aplicaciones, como la salud, conducción autónoma, la vigilancia y el entretenimiento. Esta ha potenciado un mayor interés al presentar un potencial de análisis en numerosos datos alrededor de diversas aplicaciones, es por ello que ha sido un factor clave para ser considerada en actividades importantes como la identificación de objetos, la reducción de escenas y la predicción de tareas en tiempo real (Gendy y Patel, 2024).

Por otro lado,  Cernadas (2024) mencionó que las primeras formas de aplicación que se dieron dentro del campo de la visión computacional fueron en el sector médico y la teledetección para diversas tareas, como la obtención de imágenes para diferentes ramas de la medicina y la teledetección en aviones y satélites, principalmente para uso militar, estudio de recursos disponibles, agricultura, entre otros.

Esta tecnología ha tomado suma relevancia dentro de lo sistemas de transporte inteligentes (ITS), los cuales se han llegado a implantar para poder contribuir en el crecimiento de su inteligencia en la seguridad vial, las técnicas utilizadas en la visión computacional dentro de las ITS se usan en diversas aplicaciones en detección como los reconocimientos automáticos de matrículas, peatones, obstáculos, anomalías por medio de cámaras de vigilancia, seguimiento de vehículos, infracciones de tránsito y personas (Dilek y Dener, 2023).

La visión computacional presenta ciertas limitaciones debido al entrenamiento de datos que se realizan en la máquina, muchos de estos problemas introdujeron los términos de “sobreajuste” y “subajuste”; esto conlleva a que un sistema no debe ser entrenado con una cantidad volumétrica o escasa de datos (Khan, Laghari y Awan, 2021).

Según Majhi y Waoo (2024), afirmaron que la visión artificial puede sufrir ataques adversariales de vulnerabilidad alterando de manera incorrecta a los datos de entrada que realizan sus modelos. Al mismo tiempo, el sesgo en datos pone ocasionalmente en duda sobre la calidad y rendimiento para su entrenamiento, generando complicaciones en la toma de decisiones; asimismo, existen problemas en la demanda de recursos computacionales, lo que ocasiona una limitación en su factibilidad en entornos más limitados.

Proceso del registro de infracciones de tránsito.

Proceso del registro de infracciones de tránsito

Para Senkus et al. (2021) el proceso se define como un conglomerado de diversas tareas comunes que operan entre sí para poder lograr un objetivo en concreto, tomando como base ciertos datos o recursos iniciales y poder tener como resultado una transformación en algo valioso para la persona o cliente que lo necesita.

Por otro lado, para Cardoso y Dias (2020) establecen que “es un conjunto de actividades/operaciones, junto con personas, equipos, procedimientos y flujo de información, que transforma una entrada en un producto o servicio, según las necesidades del cliente”.  

Según el Artículo 322 del Decreto Supremo N.º 016-2009-MTC (2014), es el proceso en donde las municipalidades o la Policía del Perú son responsables de registrar diariamente en el Registro Nacional de Sanciones todas las infracciones terrestres. Cada registro debe incluir la descripción de la falta y la sanción impuesta, el número de papeleta que la documenta, el nombre del conductor o peatón, el número de licencia de conducir o documento de identidad y, si aplica, la placa del vehículo. Además, debe consignarse el lugar donde ocurrió la infracción, si existió un accidente y si hubo daño personal, así como las reincidencias del infractor; finalmente, se deja abierto el campo para incluir cualquier otro dato pertinente.

Para el Decreto Supremo N.º 016-2009-MTC (2014), se constituye una infracción de tránsito el hecho de atravesar una intersección o girar el vehículo cuando la señal luminosa del semáforo está en rojo (M.17). Esta norma enfatiza que la prohibición aplica siempre y cuando no exista otra señalización que autorice expresamente la maniobra. 

Según en manual de infracciones de tránsito dictaminado por el Decreto Supremo N.º 016-2009-MTC (2014), se considera a la infracción M.17 cuando “al cruzar una intersección o girar, estando el semáforo con luz roja y no existiendo la indicación en contario, el cual está clasificado como una falta muy grave”. 

Dicha infracción se ha considerado como un desafío muy alarmante para la seguridad vial en intersecciones con señales de tránsito, debido a que puede ocasionar numerosos accidentes graves y múltiples lesiones dentro de intervalos de segundos del semáforo en rojo, lo que produce impactos en ángulo recto del tráfico donde pase (Hossain, Kang y Wu, 2025).

Visión computacional en el proceso de registro de infracciones de tránsito

En el avance de la tecnología, la visión computacional se ha evidenciado con el aumento relacionado a las aplicaciones contextualizadas al tráfico en los ITS, los cuales mejoran la eficiencia dentro de los diversos sistemas de transporte de manera inteligente, beneficiando a la seguridad vial (Dilek y Dener, 2023).

Por otro lado, la visión computacional presenta la característica de poder obtener los datos visuales con información relevante, lo cual facilita la manera convencional de la monitorización manual. Debido a que estas formas tradicionales suelen acumular mucho tiempo y consumir recursos que inducen al error humano, acelerar este proceso de monitorización, permitiría solucionar este problema y favorecer al cumplimiento de las leyes tránsito (Gehani, 2024).

Según Aliane et al. (2014), establece que “cuando se comete una infracción de tránsito, se registra el escenario correspondiente. Este escenario consiste en un conjunto de datos compuesto principalmente por el tipo de señal de tráfico detectada, su ubicación GPS, una imagen del entorno, la velocidad del vehículo, etc”.

El artículo de Yousef et al. (2020), mencionó un software completamente automatizado para la detección y lectura de placas permite acelerar el seguimiento de vehículos, minimizar errores y disminuir los costos asociados al registro de infracciones de tránsito. Es por esa razón que se menciona a los ANPR, como una forma de aplicación muy representativa en los ITS.

Por su parte, los estudios como el de Pradhan, Ranjan y Singh (2025) han evidenciado su utilidad para detectar y registrar vehículos, resaltando su valía en entornos de aparcamiento automatizado, mediante el análisis de las placas capturadas por cámaras con estos sistemas solucionar los problemas comunes en los estacionamientos e impulsa a presentar una estructura urbana mucho más avanzada e inteligente.

Las cámaras ANPR actuales han evolucionado hasta ofrecer, además de la lectura de matrículas, el registro de información complementaria como el recuento de vehículos, su dirección de desplazamiento, la clasificación por tipo y la estimación de velocidad lo que ha permitido integrar esta tecnología en múltiples aplicaciones de movilidad inteligente sin intervención humana (Lubna, Mufti y Shah, 2021).

Estos sistemas presentan diversos retos operativos relacionados con las condiciones de iluminación, la presencia de lluvia o polvo, las altas velocidades de los vehículos, los ángulos variables de las placas y la baja calidad de las imágenes capturadas; incluso el estilo tipográfico de las matrículas puede mermar la precisión del reconocimiento (Vargoorani y Suen 2024).

Además, para Yang y Wang (2019), los sistemas de reconocimiento de matrículas deben lidiar con variaciones extremas en el tamaño, estilo tipográfico y color de las placas, así como con imágenes de baja calidad debido a ángulos de captura inclinados, iluminación irregular, oclusiones y desenfoque, lo que complica la extracción fiable de caracteres y exige un procesamiento rápido para aplicaciones de vigilancia en tiempo real.

Definición de términos básicos

Detección de objetos: La detección de objetos puede facilitar información importante para la interpretación de imágenes y videos, y está asociada con varias aplicaciones, incluida la clasificación de imágenes (Zhao et al., 2019). 

Detección y reconocimiento de matrículas (LPDR): El sistema LPDR es una tecnología de procesamiento de imágenes que se utiliza para identificar vehículos mediante su matrícula y así facilitar la gestión del tráfico (Slimani et al., 2020). 

Red neuronal: Es un grafo que se encuentra orientado a los nodos, los cuales se les denomina neuronas y flechas que son llamadas aristas, estas últimas comúnmente se les llama sinapsis; de esta forma, estos nodos se etiquetan mediante símbolos (Manca 2024). 

Reconocimiento automático de matrículas (ALPR): El ALPR puede extraer información de las matrículas de las imágenes capturadas por cámaras lo que permite una identificación y un seguimiento eficientes de los vehículos (Pradhan, Ranjan y Singh, 2025). 

Sistema de transporte inteligente (ITS): Es un sistema de transporte integrado el cual abarca sistemas de transporte sostenibles, seguros e interconectados, incluyendo tranvías, autobuses, metro, automóviles, transporte marítimo y aéreo, bicicletas y peatones, con la finalidad de brindar una perspectiva diferente y más inteligente del sistema de transporte, que abarca la gestión del tráfico, la seguridad vial y muchos otros aspectos (Avcı y Koca , 2024).

Superresolución de imágenes: Tiene por objetivo restaurar una imagen de elevada resolución sobre la base de una de poca resolución (Dong et al., 2016). 

Hipótesis y variables

Supuestos básicos

No aplica.

Hipótesis

Hipótesis general 

Hg: Si se usa visión computacional de cruces en rojo, entonces mejoró significativamente el proceso de registro de infracciones de tránsito en Trujillo en 2025.

Hipótesis específicas

H1: El uso de visión computacional de cruces en rojo incrementa significativamente la tasa de infracciones correctamente detectadas (TI) de la posprueba del Grupo Experimental (TIGE) con respecto a la posprueba del Grupo Control (TIGC). 

H2: El uso de visión computacional de cruces en rojo disminuye significativamente el tiempo de registro de infracciones (TR) de la posprueba del Grupo Experimental (TRGE) con respecto a la posprueba del Grupo Control (TRGC).

                         

H3: El uso de visión computacional de cruces en rojo incrementa significativamente el número de infracciones detectadas (NID) de la posprueba del Grupo Experimental (NIDGE) con respecto a la posprueba del Grupo Control (NIDGC). 

Variables

Tabla 1. Operacionalización de variables.

Variable

Definición conceptual

Definición operacional

Indicador

Escala de Medición

Visión computacional

“La Visión computacional es la ciencia que busca comprender el entorno que abarca bajo la percepción visual por medio de imágenes y videos, los cuales están basados en un modelo físico para generar mediante sistemas de IA las decisiones más adecuadas. Para los humanos la visión es una percepción de los sentidos, mientras que los sistemas IA buscan acercarse a entender esa percepción, dependiendo de un recurso externo que se use (Elgendy, 2020, p.4). 

La variable “Visión computacional” se mide mediante el indicador Presencia_Ausencia, que toma el valor “No” si el sistema de análisis automático de vídeo no está implementado en la intersección, y “Sí” si está operativo.

Presencia_Ausencia

Nominal

Proceso de registro de infracciones de Tránsito

Según el Artículo 322 del Decreto Supremo N.º 016-2009-MTC (2014), es el proceso en donde las municipalidades o la Policía del Perú son responsables de registrar diariamente en el Registro Nacional de Sanciones todas las infracciones terrestres. Cada registro debe incluir la descripción de la falta y la sanción impuesta, el número de papeleta que la documenta, el nombre del conductor o peatón, el número de licencia de conducir o documento de identidad y, si aplica, la placa del vehículo. Además, debe consignarse el lugar donde ocurrió la infracción, si existió un accidente y si hubo daño personal, así como las reincidencias del infractor; finalmente, se deja abierto el campo para incluir cualquier otro dato pertinente.

Para medir la variable proceso de registro de infracciones de tránsito se realiza con los indicadores: tasa de infracciones detectadas, tiempo de registro de número de infracciones detectadas. Además, estos indicadores se miden mediante una ficha de observación (por cada uno).

Tasa de infracciones correctamente detectadas (TI)

Razón

Tiempo de Registro (TR)

Número de Infracciones Detectadas (NID)

Fuente: Elaborado por los autores.

Matriz de consistencia

Tabla 2. Matriz de consistencia de la investigación.

Título de la investigación

Problema general

Objetivo General

Hipótesis general

Variables

Indicadores

Metodología

El proceso de registro de infracciones por cruce en rojo en Trujillo se realiza de forma manual apoyado en anotaciones en papel y validaciones presenciales que generan amplias demoras en la incorporación de cada evento al sistema. Esta ineficiencia administrativa no sólo retrasa la aplicación de sanciones, sino que también debilita la transparencia y la capacidad de las autoridades para diseñar políticas preventivas basadas en datos actualizados.

Mejorar el proceso de registro de infracciones de tránsito en Trujillo, a través del uso de visión computacional de cruces en rojo en 2025.

Si se usa visión computacional de cruces en rojo, entonces mejoró significativamente el proceso de registro de infracciones de tránsito en Trujillo en 2025.

Independiente

Visión computacional

Presencia_Ausencia

Enfoque:

Cuantitativo

Tipo de investigación: Aplicada

Diseño de investigación:

Experimental de grado experimental puro

Población:

Todos los registros de infracciones de tránsito por cruce en rojo en intersecciones urbanas del Perú.

Muestra:

Dos grupos de 30 registros: manuales (control) y automatizados (experimental) en intersecciones de Trujillo.

Problemas específicos

Objetivos específicos

Hipótesis específicas

Dependiente

Proceso de registro de infracciones de Tránsito

Tasa de infracciones correctamente detectadas (TI)

Tiempo de Registro (TR)

Número de Infracciones Detectadas (NID)

La cobertura manual y la falta de un sistema automatizado de captura permanente provocan que gran parte de los cruces en rojo no sean registrados

El procesamiento secuencial de anotaciones en papel, transcripción al sistema digital y validación presencial extiende el tiempo de registro por evento, dificultando una respuesta ágil.

La ausencia de un archivo digital unificado impide rastrear de forma sistemática el número de infractores detectados.

Aumentar la tasa de infracciones correctamente detectadas en infracciones de cruces en rojo.

Disminuir el tiempo de registro de cada infracción de cruce en rojo.

Aumentar el número de infracciones detectadas de cruces en rojo.  

H1: El uso de visión computacional de cruces en rojo incrementa significativamente la tasa de infracciones correctamente detectadas (TI) de la posprueba del Grupo Experimental (TIGE) con respecto a la posprueba del Grupo Control (TIGC).  

H2: El uso de visión computacional de cruces en rojo reduce significativamente el tiempo de registro de infracciones (TR) de la posprueba del Grupo Experimental (TRGE) con respecto a la posprueba del Grupo Control (TRGC).

H3: El uso de visión computacional de cruces en rojo incrementa significativamente el número de infracciones detectadas (NID) de la posprueba del Grupo Experimental (IRGE) con respecto a la posprueba del Grupo Control (IRGC).

Fuente: Elaborado por los autores.

Marco metodológico

Tipo de investigación

La investigación, de acuerdo a su finalidad, es de tipo aplicada, debido a que busca mejorar la eficiencia en la detección y del registro de infracciones relacionadas con los cruces en rojo, mediante el uso de un sistema inteligente para la automatización del proceso. Además, de acuerdo con la técnica de contrastación, es experimental, porque se manipula la variable de visión computacional en la evaluación de su impacto sobre las variables dependientes (tasa de infracciones correctamente detectadas, tiempo de registro de cada infracción y número de infracciones detectadas).

Nivel de madurez tecnológica

El nivel de madurez tecnológica del sistema desarrollado se ubica en el TRL 4 (Validación de componentes en entorno de laboratorio controlado), dado que la investigación cuenta con un prototipo funcional de visión computacional que ha integrado exitosamente los módulos de captura de video, detección de vehículos y semáforos mediante YOLOv8, segmentación de matrículas con procesamiento de imágenes y reconocimiento de caracteres a través de OCR.

La elección del TRL 4 se justifica porque, aunque los componentes principales del sistema ya han sido diseñados, implementados y validados en pruebas experimentales controladas dentro de un entorno delimitado simulando las condiciones reales de las intersecciones, aún no se ha desplegado de manera completa en un entorno operativo real.

Método de investigación

La investigación adopta el método de investigación cuantitativo-experimental, porque a través de la manipulación intencional de la variable independiente: visión computacional, se pretende determinar su influencia en la variable dependiente: proceso de registro de infracciones de tránsito.

Diseño del estudio

La investigación tiene un diseño experimental de grado experimental puro, porque se tiene a la visión computacional (VI) la cual tendrá un único indicador “Presencia_Ausencia”, además se conformó dos grupos referidos al proceso de registro de infracciones de tránsito (VD). El primero llamado Grupo de Control (GC), el cual no utilizara la VI y otro Grupo Experimental (GE) que si la utilizara.  Cabe señalar que existirá asignación aleatoria de los tratamientos (VI) a los grupos (GC y GE) de la VD, lo que permite tener un mayor grado de control sobre variables externas y, así obtener los datos de cada indicador: Tasa de infracciones correctamente detectadas (TI), Tiempo de Registro (TR), Número de Infracciones Detectadas (NID).

Figura 1. Diseño con posprueba únicamente y grupo de control.

Fuente: Hernández et al. (2014).

Donde:

R: Selección al azar de las intersecciones críticas y asignación aleatoria de estas al grupo experimental (Ge) o al grupo control (Gc).

Ge: Grupo experimental que recibió la condición de visión computacional de cruces en rojo (X).

O1: Observaciones de la posprueba en el grupo experimental. Se midieron los indicadores.

X: Condición experimental: implementación del sistema de visión computacional para detección y registro automatizado de infracciones de cruces en rojo.

Gc: Grupo control que siguió con el proceso manual habitual de registro de infracciones.

O2: Observaciones de la posprueba en el grupo control, midiendo los mismos indicadores (TI, TR, NID).

--: Falta de la condición experimental.

Población y muestra

La población de la investigación está conformada por todos los registros de infracciones de tránsito por cruce en rojo realizados en intersecciones urbanas del Perú por la policía de tránsito, como no se puede cuantificar, se considera como “infinita o indeterminada” (N = indeterminado). 

La muestra está conformada por los registros de infracciones de tránsito por cruce en rojo obtenidos en las intersecciones críticas de la ciudad de Trujillo, distribuidos en dos grupos de 30 registros cada uno: el grupo de control (registros manuales) y el grupo experimental (registros automatizados). Un número mayor a 30 está avalado por lo expresado por Cohen (1988), en su libro “Statistical Power Analysis for the Behavioral Sciences”, quien indicó que la muestra de alrededor de 30 puede ser apropiada dependiendo del efecto que se pretende medir.

Finalmente, es importante señalar que la unidad muestral corresponde a cada registro individual de una infracción por cruce en rojo, y el tipo de muestreo es aleatorio simple.

Técnicas e instrumentos de recolección de datos

Para la recolección de datos en la investigación se utilizará como técnica a la observación indirecta y como instrumento a la ficha de observación, esta última se elaborará una por cada indicador. Tasa de infracciones correctamente detectadas (TI), Tiempo de Registro (TR) y Número de Infracciones Detectadas (NID) de la variable dependiente: Visión Computacional.

Cabe señalar que las fichas de observación no pasarán por un proceso de validación ni confiabilidad, pues solo se calculará datos numéricos de cada indicador.

Procedimientos de ejecución del estudio

Para la recolección de requisitos (01/04/2025 – 15/04/2025), se llevaron a cabo inicialmente en reuniones con el personal del Centro de Monitoreo de Trujillo y se revisaron los manuales técnicos de las cámaras semafóricas y los protocolos vigentes, con el fin de precisar los requisitos funcionales y no funcionales del sistema, luego se dirigió a la subgerencia de transporte para el conocimiento del funcionamiento de los semáforo y posteriormente se fue la Unidad de Tránsito y Seguridad Vial para la recolección de datos de las papeletas para luego realizar una encuesta en el mismo lugar a 30 policías de tránsito para que pudieran responder ciertas preguntas que permitieron recolectar necesaria para nuestro indicadores.

En el diseño de la solución (16/04/2025 – 21/04/2025), se elaboraron la arquitectura de software y los diagramas UML/BPMN, definiendo los módulos de captura y preprocesamiento de vídeo, detección de vehículos y semáforo, segmentación de placa y OCR, así como el esquema de la base de datos para el almacenamiento de infracciones.

Para el desarrollo e implementación (22/04/2025 – 03/06/2025), se seleccionaron Python, OpenCV y TensorFlow; se entrenaron modelos YOLOv8 para detección de vehículos y matrículas; se integraron algoritmos de mejora de imagen; se configuró Paddle para OCR; se trabajó además con la metodología XP y se desplegó la infraestructura de backend en Google Cloud.

En las pruebas (04/06/2025 – 14/06/2025), se realizaron pruebas unitarias de cada componente, pruebas de integración del flujo completo de vídeo y mediciones de rendimiento para TI, TR y precisión de OCR, aplicando validación cruzada y análisis estadístico.

Finalmente, para el despliegue (15/06/2025 – 23/06/2025), el sistema se desplegó en tres intersecciones críticas de Trujillo, donde se midieron los indicadores, se recogió retroalimentación de los operadores y se ajustaron los umbrales de detección y la interfaz de usuario.

Técnicas de procesamiento y análisis de datos

En el marco de la investigación para el desarrollo de visión computacional aplicada al registro de infracciones de tránsito en Trujillo, se implementarán técnicas de estadística descriptiva e inferencial sobre los tres indicadores principales: Tasa de infracciones correctamente detectadas (TI), Tiempo de Registro (TR) y Número de Infracciones Detectadas (NID). Mediante el análisis descriptivo se calcularán medidas de tendencia central (media y mediana) y de dispersión (desviación estándar y rango intercuartílico) para cada indicador, complementadas con histogramas y diagramas de caja que faciliten la detección de patrones y valores atípicos entre el método manual y el automatizado. 

En el análisis inferencial se aplicará la prueba de normalidad (Shapiro–Wilk) para determinar si los datos de TI, TR y NID siguen esta distribución. En función de este resultado, se empleará la prueba U de Mann-Whitney para muestras independientes, con el fin de evaluar si la implementación de visión computacional produce cambios estadísticamente significativos en cada indicador. Estos resultados permitirán confirmar la efectividad del sistema automatizado en comparación con el método manual y cuantificar su impacto real en el registro de infracciones.

Presentación de resultados

Resultados de la investigación

Análisis descriptivo

A continuación, se muestran los valores de la posprueba del Grupo de Control (GC) y del Grupo Experimental (GE) de los indicadores del proceso de registro de infracciones: Tasa de Infracciones correctamente detectadas (TI), Tiempo de Registro (TR) y Número de Infracciones Detectadas (NID).

Resultado de la posprueba para el indicador 1: Número de Infracciones Detectadas (NID), tanto para el Grupo de Control (GC) como para el Grupo Experimental (GE).

Tabla 3. Resultados de la posprueba para el indicador Número de Infracciones Detectadas (NID).

N°

Grupo de Control(NIDGC)

Grupo Experimental(NIDGE)

Grupo Experimental(NIDGE)

Grupo Experimental(NIDGE)

1

10

181

181

181

2

10

410

410

410

3

8

565

565

565

4

10

154

154

154

5

5

98

98

98

6

10

87

87

87

7

15

706

706

706

8

5

320

320

320

9

15

216

216

216

10

6

29

29

29

11

3

120

120

120

12

10

814

814

814

13

11

151

151

151

14

4

115

115

115

15

10

103

103

103

16

3

381

381

381

17

1

101

101

101

18

10

480

480

480

19

2

195

195

195

20

8

72

72

72

21

3

93

93

93

22

5

364

364

364

23

4

184

184

184

24

4

1297

1297

1297

25

3

828

828

828

26

4

270

270

270

27

5

1263

1263

1263

28

6

552

552

552

29

6

188

188

188

30

8

132

132

132

Promedio

7

349

Meta planteada

100

N Mayor al promedio

11

25

30

% mayor al promedio

36.66

83.33

100

Fuente: Elaborado por los autores en base a la ficha de observación del número de infracciones detectadas.

En la tabla 3 se pudo precisar que el 36.66% del Número de Infracciones Detectadas (NID) en la posprueba del Grupo de Experimental (GE) fueron mayores que su promedio (349). Adicionalmente, el 83.33% del NID en la posprueba del Grupo Experimental (GE) fueron mayores a la meta planteada (100). Finalmente, se determinó que el 100% del NID en la posprueba del Grupo Experimental (GE) fueron mayores al promedio (7) de la posprueba del Grupo de Control (GC).

Posteriormente, en la tabla 4 se presentan los estadísticos descriptivos del indicador NID, tales como la media, mediana, desviación estándar, valor mínimo, valor máximo, asimetría y error estándar, tanto para los datos de la posprueba del Grupo de Control (GC) como del Grupo Experimental (GE).

Tabla 4. Estadísticos descriptivos para el indicador Número de Infracciones Detectadas (NID).

 

N

Media

Mediana

DE

Mínimo

Máximo

Asimetría

EE

NIDGC

30

6.80

6.00

3.66

1

15

0.582

0.427

NIDGE

30

348.97

191.50

336.15

29

1297

1.642

0.427

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Resultado de la posprueba para el indicador 2: Tiempo de Registro (TR), tanto para el Grupo de Control (GC) como para el Grupo Experimental (GE).

Tabla 5. Resultados de la posprueba para el indicador Tiempo de Registro (TR).

N°

Grupo de Control(TRGC)

Grupo Experimental(TRGE)

Grupo Experimental(TRGE)

Grupo Experimental(TRGE)

1

13

2,65

2,65

2,65

2

7

1,17

1,17

1,17

3

10

0,85

0,85

0,85

4

5

3,11

3,11

3,11

5

5

4,89

4,89

4,89

6

10

5,52

5,52

5,52

7

5

0,68

0,68

0,68

8

4

1,5

1,5

1,5

9

8

2,22

2,22

2,22

10

10

16,84

16,84

16,84

11

8

4,01

4,01

4,01

12

30

0,59

0,59

0,59

13

10

3,17

3,17

3,17

14

7

4,17

4,17

4,17

15

15

4,68

4,68

4,68

16

5

1,26

1,26

1,26

17

8

4,77

4,77

4,77

18

15

1

1

1

19

10

2,46

2,46

2,46

20

8

6,7

6,7

6,7

21

5

5,18

5,18

5,18

22

12

1,32

1,32

1,32

23

12

2,61

2,61

2,61

24

3

0,37

0,37

0,37

25

12

0,58

0,58

0,58

26

15

1,78

1,78

1,78

27

10

0,38

0,38

0,38

28

12

0,87

0,87

0,87

29

12

2,55

2,55

2,55

30

10

3,65

3,65

3,65

Promedio

9,87

3,05

Meta planteada

5

N Menor al promedio

18

26

29

% menor al promedio

60

86,66

96,66

Fuente: Elaborado por los autores en base a la ficha de observación del tiempo de registro por infracción.

Con respecto a la tabla 5 se pudo definir que el 60% del Tiempo de Registro (TR) en la posprueba del Grupo de Experimental (GE) fueron menores que su promedio (3). Además de esto, el 86.66% del TR en la posprueba del Grupo Experimental (GE) fueron menores a la meta planteada (5). Finalmente, se determinó que el 96.66% del TR en la posprueba del Grupo Experimental (GE) fueron menores al promedio (9,87) de la posprueba del Grupo de Control (GC).

Consecuentemente, en la tabla 6 se presentan los estadísticos descriptivos del indicador TR, tales como la media, mediana, desviación estándar, valor mínimo, valor máximo, asimetría y error estándar, tanto para los datos de la posprueba del Grupo de Control (GC) como del Grupo Experimental (GE).

Tabla 6. Estadísticos descriptivos para el indicador Tiempo de Registro (TR).

 

N

Media

Mediana

DE

Mínimo

Máximo

Asimetría

EE

TRGC

30

9.87

10.00

5.08

3

30

2.11

0.427

TRGE

30

3.05

2.50

3.15

0.370

16.8

3.06

0.427

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Resultado de la posprueba para el indicador 3: Tasa de Infracciones detectadas (TI), tanto para el Grupo de Control (GC) como para el Grupo Experimental (GE).

Tabla 7.Resultados de la posprueba para el indicador Tasa de Infracciones detectadas (TI).

N°

Grupo de Control(TIGC)

Grupo Experimental(TIGE)

Grupo Experimental(TIGE)

Grupo Experimental(TIGE)

1

66,6

97.1

97.1

97.1

2

71,4

95.75

95.75

95.75

3

72,7

97.5

97.5

97.5

4

76,9

96.86

96.86

96.86

5

55,5

97.56

97.56

97.56

6

83,3

98,00

98,00

98.0

7

75

97.66

97.66

97.66

8

62,5

94.75

94.75

94.75

9

71,4

96.3

96.3

96.3

10

60

98.5

98.5

98.5

11

60

90.4

90.4

90.4

12

55,5

95.6

95.6

95.6

13

78,5

98.6

98.6

98.6

14

25

95.2

95.2

95.2

15

55,5

96.1

96.1

96.1

16

60

97.95

97.95

97.95

17

10.1

94.7

94.7

94.7

18

50

96.2

96.2

96.2

19

20

98.6

98.6

98.6

20

80

96.8

96.8

96.8

21

60

98.05

98.05

98.05

22

55,5

83.5

83.5

83.5

23

57,1

97.4

97.4

97.4

24

66,6

94.95

94.95

94.95

25

37,5

95,00

95,00

95.0

26

57,1

93,00

93,00

93.0

27

55,5

95.2

95.2

95.2

28

60

95.3

95.3

95.3

29

60

97.5

97.5

97.5

30

80

96.65

96.65

96.65

Promedio

59

95

Meta planteada

90

N Mayor al promedio

23

29

30

% mayor al promedio

76.66

96.66

100

Fuente: Elaborado por los autores en base a la ficha de observación de la tasa de infracciones detectadas.

Para la tabla 7 se pudo precisar que el 76.66% de la Tasa de Infracciones correctamente detectadas (TI) en la posprueba del Grupo de Experimental (GE) fueron mayores que su promedio (95). Asimismo, el 96.66% de la TI en la posprueba del Grupo Experimental (GE) fueron mayores a la meta planteada (90). Finalmente, se determinó que el 100% de la TI en la posprueba del Grupo Experimental (GE) fueron mayores al promedio (59) de la posprueba del Grupo de Control (GC).

Acto seguido, en la tabla 8 se presentan los estadísticos descriptivos del indicador TI, tales como la media, mediana, desviación estándar, valor mínimo, valor máximo, asimetría y error estándar, tanto para los datos de la posprueba del Grupo de Control (GC) como del Grupo Experimental (GE).

Tabla 8. Estadísticos descriptivos para el indicador Tasa de Infracciones detectadas (TI).

 

N

Media

Mediana

DE

Mínimo

Máximo

Asimetría

EE

TIGC

30

59.3

60.0

17.26

11.1

83.3

-1.23

0.427

TIGE

30

95.9

96.5

2.94

83.5

98.6

-2.89

0.427

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Análisis inferencial

Para el análisis inferencial se llevo acabo la prueba de normalidad y la contrastación de la hipótesis, y para ello, se propusieron los siguientes criterios de decisión de la posprueba del Grupo de Control (GC) y del Grupo Experimental (GE) de los indicadores: Tasa de Infracciones correctamente detectadas (TI), Tiempo de Registro (TR) y Número de Infracciones Detectadas (NID):

-Si 𝑝 < 0.05, entonces se rechaza la hipótesis nula () y se acepta la hipótesis alterna ().

-Si 𝑝 ≥ 0.05, entonces se acepta la hipótesis nula () y se rechaza la hipótesis alterna ().

Indicador 1: Número de Infracciones Detectadas (NID)

Prueba de normalidad: A partir de esto, se plantean las hipótesis para el indicador Número de Infracciones Detectadas (NID) tanto de la posprueba del Grupo de Control (GC) como la del Grupo Experimental (GE):

Número de Infracciones Detectadas de la posprueba del Grupo de Control (NIDGC)

-H0: Los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo de Control (NIDGC) se distribuyen normalmente.  

-Ha: Los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo de Control (NIDGC) no se distribuyen normalmente.

Tabla 9. Test de normalidad de Shapiro-Wilk para NIDGC.

Indicador

Estadístico

p

NIDGC

0.927

0.041

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Figura 2. Histograma de la normalidad de los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo de Control (NIDGC).

Fuente. Jamovi 2.3.28.

Como el número de datos del indicador Número de Infracciones Detectadas (NID) de la posprueba del Grupo de Control (GC), son menores a 50, se tomó en cuenta la prueba Shapiro-Wilk (tabla 9), el cual dio como valor 𝑝 = 0.041, que por ser inferior a 0.05 (∝), se deduce que los datos no se distribuyen normalmente, además, este resultado se puede evidenciar gráficamente en la Figura 2 de este documento.

Número de Infracciones Detectadas de la posprueba del Grupo de Control (NIDGE)

-H0: Los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo Experimental (NIDGE) se distribuyen normalmente.  

-Ha: Los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo Experimental (NIDGE) no se distribuyen normalmente.

Tabla 10. Test de normalidad de Shapiro-Wilk para NIDGE.

Indicador

Estadístico

p

NIDGE

0.789

0.001

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Como el conjunto de datos del indicador Número de Infracciones Detectadas (NID) de la posprueba del Grupo Experimental (GE), son menores a 50, se tomó en cuenta la prueba Shapiro-Wilk (tabla 10), el cual dio como valor 𝑝 = 0.001, que por ser inferior a 0.05 (∝), se concluye que los datos no se distribuyen normalmente, además, este resultado se puede evidenciar gráficamente en la Figura 3 de este documento.

Figura 3. Histograma de la normalidad de los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo Experimental (NIDGE).

Fuente. Jamovi 2.3.28.

Por ello, al concluir que los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo de Control (NIDGC), no se distribuyen normalmente y los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo Experimental (NIDGE), no se distribuyen normalmente, se aplicó la prueba estadística no paramétrica U de Mann-Whitney para probar la diferencias entre grupos independientes.

Contrastación de la hipótesis: Para la prueba de hipótesis del indicador Número de Infracciones (NID) se plantearon las siguientes:

-Ha: Si se usa un sistema de visión computacional de cruces en rojo, entonces aumenta el número de infracciones detectadas de la posprueba del Grupo Experimental (NIDGE) con respecto a la muestra de la posprueba del Grupo Control (NIDGC).

Ha: μ1 < μ2

-Ho: Si se usa un sistema de visión computacional de cruces en rojo, entonces disminuye el número de infracciones detectadas de la posprueba del Grupo Experimental (NIDGE) con respecto a la muestra de la posprueba del Grupo Control (NIDGC).

Ho: μ1 >= μ2

Donde:

μ1 = Media poblacional del número de infracciones detectadas en la posprueba del grupo de Control (NIDGC).

μ2 = Media poblacional del número de infracciones detectadas en la posprueba del grupo de Control (NIDGE).

Tabla 11. Estadístico de U de Mann-Whitney para NID.

Indicador

Estadístico

p

Número de Infracciones Detectadas (NID)

0.00

< 0.001

Nota. Hₐ: μ 1 < μ 2

Fuente. Jamovi 2.3.28.

Por lo tanto, según los datos de la Tabla 11, el valor de p es <0.001 y este es inferior a 0.05, en consecuencia, estos resultados aportan suficiente evidencia estadística para rechazar la hipótesis nula (Ho) y aceptar la hipótesis alterna (Ha).

Indicador 2: Tiempo de Registro (TR)

Prueba de normalidad: A partir de esto, se plantean las hipótesis para el indicador Tiempo de registro (TR) tanto de la posprueba del Grupo de Control (GC) como la del Grupo Experimental (GE):

Tiempo de Registro de la posprueba del Grupo de Control (TRGC)

-H0: Los datos del indicador Tiempo de Registro de la posprueba del Grupo de Control (TRGC) se distribuyen normalmente. 

-Ha: Los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo de Control (TRGC) no se distribuyen normalmente.

Tabla 12.  Test de normalidad de Shapiro-Wilk para TRGC.

Indicador

Estadístico

p

TRGC

0.816

0.001

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Como el número de datos del indicador Tiempo de Registro (TR) de la posprueba del Grupo de Control (GC), son menores a 50, se tomó en cuenta la prueba Shapiro-Wilk (tabla 6), el cual dio como valor 𝑝 = 0.001, que por ser inferior a 0.05 (∝), se infiere que los datos no se distribuyen normalmente, además, este razonamiento se puede evidenciar gráficamente en la Figura 3 de este documento.

Figura 4. Histograma de la normalidad de los datos del indicador Tiempo de Registro de la posprueba del Grupo de Control (TRGC).

Fuente. Jamovi 2.3.28.

Tiempo de Registro de la posprueba del Grupo Experimental (TRGE)

-H0: Los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo Experimental (TRGE) se distribuyen normalmente.  

-Ha: Los datos del indicador Número de Infracciones Detectadas de la posprueba del Grupo Experimental (TRGE) no se distribuyen normalmente.

Tabla 13. Test de normalidad de Shapiro-Wilk para TRGE.

Indicador

Estadístico

p

TRGE

0.696

0.001

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Como el conjunto de datos del indicador Tiempo de Registro (NV) de la posprueba del Grupo Experimental (GE), son menores a 50, se tomó en cuenta la prueba Shapiro-Wilk (tabla 7), el cual dio como valor 𝑝 = 0.001, que por ser menor a 0.05 (∝), se concluye que los datos no se distribuyen normalmente, además, este resultado se puede evidenciar gráficamente en la Figura 4 de este documento.

Figura 5. Histograma de la normalidad de los datos del indicador Número de Ventas de la posprueba del Grupo Experimental (NVGE).

Fuente. Jamovi 2.3.28.

Por esta razón, al concluir que los datos del indicador Tiempo de Registro de la posprueba del Grupo de Control (TRGC), no se distribuyen normalmente y los datos del indicador Tiempo de Registro de la posprueba del Grupo Experimental (TRGE), no se distribuyen normalmente, se aplicó la prueba estadística no paramétrica U de Mann-Whitney para probar la diferencias entre grupos independientes.

Contrastación de la hipótesis: Para la prueba de hipótesis del indicador Tiempo de Registro (TR) se plantearon las siguientes:

-Ha: Si se usa un sistema de visión computacional de cruces en rojo, entonces disminuye el tiempo de registro de la posprueba del Grupo Experimental (TRGE) en relación con la muestra de la posprueba del Grupo Control (TRGC).

Ha: μ1 > μ2

-Ho: Si se usa un sistema de visión computacional de cruces en rojo, entonces aumenta el tiempo de registro de la posprueba del Grupo Experimental (TRGE) en relación con la muestra de la posprueba del Grupo Control (TRGC).

Ho: μ1 <= μ2

Donde:

μ1 = Media poblacional del tiempo de registro en la posprueba del grupo de Control (TRGC).

μ2 = Media poblacional del tiempo de registro en la posprueba del grupo de Control (TRGE).

Tabla 14. Estadístico de U de Mann-Whitney para TR.

Indicador

Estadístico

p

Tiempo de Registro (TR)

63.0

< 0.001

Nota. Hₐ: μ 1 > μ 2

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Por eso, según los datos de la Tabla 5, el valor de p es <0.001 y este es inferior a 0.05, así pues, estos resultados ofrecen suficiente evidencia estadística para rechazar la hipótesis nula (Ho) y aceptar la hipótesis alterna (Ha).

Indicador 3: Tasa de infracciones (TI)

Prueba de normalidad: A continuación, se plantean las hipótesis para el indicador Tasa de Infracciones (TI) tanto de la posprueba del Grupo de Control (GC) como la del Grupo Experimental (GE):

Tasa de Infracciones de la posprueba del Grupo de Control (TIGC)

-H0: Los datos del indicador Tasa de Infracciones de la posprueba del Grupo de Control (TIGC) se distribuyen normalmente.  

-Ha: Los datos del indicador Tasa de Infracciones de la posprueba del Grupo de Control (TIGC) no se distribuyen normalmente.

Tabla 15. Test de normalidad de Shapiro-Wilk para TIGC.

Indicador

Estadístico

p

TIGC

0.877

0.002

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Como el número de datos del indicador Tiempo de Registro (TR) de la posprueba del Grupo de Control (GC), son menores a 50, se tomó en cuenta la prueba Shapiro-Wilk (tabla 15), el cual dio como valor 𝑝 = 0.002, que por ser inferior a 0.05 (∝), se deduce que los datos no se distribuyen normalmente, además, este resultado se puede evidenciar gráficamente en la Figura 5 de este documento.

Figura 6. Histograma de la normalidad de los datos del indicador Tasa de Infracciones de la posprueba del Grupo de Control (TIGC).

Fuente. Jamovi 2.3.28.

Tasa de Infracciones de la posprueba del Grupo de Control (TIGE)

-H0: Los datos del indicador Tasa de Infracciones de la posprueba del Grupo Experimental (TIGE) se distribuyen normalmente.  

-Ha: Los datos del indicador Tasa de Infracciones de la posprueba del Grupo Experimental (TIGE) no se distribuyen normalmente.

Tabla 16. Test de normalidad de Shapiro-Wilk paraTIGE.

Indicador

Estadístico

p

NIDGE

0.708

0.001

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Como el conjunto de datos del indicador Tasa de Infracciones (TI) de la posprueba del Grupo Experimental (GE), son menores a 50, se tomó en cuenta la prueba Shapiro-Wilk (tabla 16), el cual dio como valor 𝑝 = 0.001, que por ser inferior a 0.05 (∝), se concluye que los datos no se distribuyen normalmente, además, este resultado se puede evidenciar gráficamente en la Figura 6 de este documento.

Figura 7. Histograma de la normalidad de los datos del indicador Tasa de Infracciones de la posprueba del Grupo Experimental (TIGE).

Fuente. Jamovi 2.3.28.

Por ello, al concluir que los datos del indicador Tasa de Infracciones de la posprueba del Grupo de Control (TIGC), no se distribuyen normalmente y los datos del indicador Tasa de Infracciones de la posprueba del Grupo Experimental (TIGE), no se distribuyen normalmente, se aplicó la prueba estadística no paramétrica U de Mann-Whitney para probar la diferencia entre grupos independientes.

Contrastación de la hipótesis: Para la prueba de hipótesis del indicador Tasa de Infracciones (TI) se plantearon las siguientes:

-Ha: Si se usa un sistema de visión computacional de cruces en rojo, entonces aumenta la tasa de infracciones de la posprueba del Grupo Experimental (TIGE) con respecto a la muestra de la posprueba del Grupo Control (TIGC).

Ha: μ1 < μ2

-Ho: Si se usa un sistema de visión computacional de cruces en rojo, entonces disminuye el número de ventas de la posprueba del Grupo Experimental (TIGE) con respecto a la muestra de la posprueba del Grupo Control (TIGC).

Ho: μ1 >= μ2

Donde:

μ1 = Media poblacional de la Tasa de Infracciones en la posprueba del grupo de Control (TIGC).

μ2 = Media poblacional de la Tasa de Infracciones en la posprueba del grupo de Control (TIGE).

Tabla 17. Estadístico de U de Mann-Whitney para NV.

Indicador

Estadístico

p

Tasa de infracciones (TI)

0.00

< 0.001

Nota. Hₐ: μ 1 < μ 2

Fuente. Elaborado por los autores en base a los datos procesados en el software Jamovi.

Por consiguiente, según los datos de la Tabla 17, el valor de p es <0.001 y este es inferior a 0.05, por ello, estos resultados brindan suficiente evidencia estadística para rechazar la hipótesis nula (Ho) y aceptar la hipótesis alterna (Ha).

Discusión de resultados

Se planteó como primer objetivo específico aumentar la tasa de infracciones correctamente detectadas (TI) mediante la implementación del sistema visión computacional en los cruces en rojo de Trujillo, se realizó la prueba de normalidad de Shapiro-Wilk, obteniéndose un valor p calculado = 0.001 (< p tabular = 0.05), lo que nos hace entender que hay una relación significativa entre las variables. Para lograr este objetivo se tiene que tomar en cuenta nuestros esfuerzos de conseguir los videos, escenarios y calidades ideales para las pruebas, debido a que este resultado puede variar según el vídeo en el cual se encuentre e influye de manera significativa ante dichos factores. Con respecto a lo mencionado, se rechaza la hipótesis nula y se acepta la hipótesis de la investigación, donde se afirma que la aplicación del sistema de visión computacional incrementa significativamente la tasa de infracciones correctamente detectadas en comparación con el método manual tradicional. Los resultados obtenidos guardan concordancia con lo reportado por Pradhan et al. (2025), quienes desarrollaron un sistema inteligente que alcanzó una precisión del 95% en la detección de matrículas bajo condiciones diurnas y del 90% en baja iluminación. A su vez, los resultados obtenidos refieren a la tasa de infracciones correctamente detectadas y registradas, los cuales se encuentran en un 95% sobre el 59% que representa los métodos convencionales. En línea teórica que menciona Szeliski (2022, p. 5), la visión computacional se consolida como una disciplina aplicada capaz de procesar imágenes y video en tiempo real, cuyas técnicas maduras, en particular, el OCR que se usa en tareas como la lectura de matrículas, se consideran pieza clave en ámbitos críticos como el transporte y la seguridad. En concordancia con lo mencionado, estos sistemas permiten interpretar el entorno visual de forma autónoma, lo que explica el aumento de precisión observado en este estudio. Se concluye que la integración de visión computacional mediante el monitoreo de los tiempos en semáforos contribuye al fortalecimiento del control del tránsito y a la mejora de la transparencia en el registro de infracciones.

Se planteó como segundo objetivo específico disminuir el tiempo de registro de infracciones (TR) mediante la implementación del sistema de visión computacional en el proceso de control de tránsito, se obtuvo un valor p calculado = 0.001 (< p tabular = 0.05) a través de la prueba de normalidad de Shapiro-Wilk, lo que nos muestra que existe una relación significativa entre las variables. Durante esta etapa se graban diferentes videos en las calles de Trujillo en distintos momentos del día con el fin de probar el algoritmo de optimización del tiempo de procesamiento y reducir los lapsos innecesarios concentrando la atención en los intervalos de luz roja bajo condiciones reales, considerando variables como la iluminación, la distancia de la cámara, el flujo vehicular y los ángulos de grabación, lo que permite ajustar los parámetros del sistema para evitar retrasos en la detección y el registro de infracciones, implicando múltiples pruebas en campo, repeticiones de grabación y sincronización manual con los tiempos del semáforo para calibrar el modelo y lograr que procese los eventos de manera continua y con una respuesta más rápida frente a las variaciones del entorno. Por lo expuesto previamente, se rechaza la hipótesis nula y se acepta la hipótesis de investigación, al confirmarse que la implementación del sistema de visión computacional reduce el tiempo de registro de infracciones. Estos resultados son coherentes con los obtenidos por Correa y Vílchez (2022), quienes reportaron una reducción del 50.32%  con una media de 0.50 min (30s) en los tiempos de registro mediante automatización del flujo de datos, y se evidenció que hubo una disminución en el tiempo de respuesta respecto a los métodos convencionales. En esta investigación, el tiempo promedio de registro se reduce de 9.87 minutos para el GC a 3.05 minutos en el GE, lo que representa una disminución aproximada del 69%, evidenciando que la automatización del proceso mediante visión computacional acelera la gestión y optimiza la eficiencia en el control de tránsito. En correspondencia con lo anterior, Yousef et al. (2020) muestran que los sistemas de reconocimiento automático de matrículas automatizan la detección y la lectura, lo que agiliza el registro y reduce errores operativos; este fundamento explica la disminución del tiempo de registro observada en nuestro estudio.   Con esto se evidencia que la eliminación de tareas repetitivas y la capacidad del sistema para procesar múltiples eventos de manera paralela permiten un funcionamiento más ágil y coordinado. Esto demuestra que la incorporación de tecnologías de automatización y procesamiento, como la visión computacional acelera las operaciones y también mejora la gestión del tiempo en el control vehicular.

Se planteó como tercer objetivo específico incrementar el número de infracciones detectadas (NID) mediante la implementación de un sistema de visión computacional para los cruces semafóricos en Trujillo, se aplicó la prueba de normalidad de Shapiro-Wilk, obteniéndose un valor p calculado = 0.001 (< p tabular = 0.05), lo que nos da a entender que existe una relación significativa entre las variables. Al grabar los videos con la cámara en las intersecciones, no es posible permanecer durante un periodo prolongado, como una jornada completa de ocho horas, debido a limitaciones de tiempo y disponibilidad en el entorno, además de no poder emplear un escenario completamente real de tránsito en funcionamiento; esta condición implica trabajar con grabaciones representativas que permiten analizar el desempeño del sistema en la detección y cuantificación de infracciones bajo condiciones controladas, pero cercanas a la realidad urbana. En relación con lo ya dicho, se rechaza la hipótesis nula y se acepta la hipótesis de investigación, al comprobarse que la implementación del sistema de visión computacional incrementa significativamente el número de infracciones detectadas Los resultados obtenidos se relacionan directamente con los hallazgos de Lu et al. (2025), quienes desarrollaron la herramienta AdvFuzz, logrando incrementar el número de infracciones detectadas durante pruebas de hasta 540 casos en 12 horas, en comparación con los 181 detectados por métodos tradicionales. En esta investigación, el número promedio de infracciones detectadas alcanza los 349 casos en el GE, frente a las 7 registradas en el GC que representa el método manual tradicional realizado por los agentes de tránsito. Este resultado se obtiene tras proyectar el funcionamiento del sistema durante una jornada de 8 horas, equivalente a la labor diaria de un efectivo policial, lo que permite dimensionar con mayor realismo la capacidad del modelo para mantener una detección constante a lo largo de todo el periodo de monitoreo. Desde la base teórica, Dilek y Dener (2023) mencionan que la visión computacional aplicada a los sistemas de transporte inteligentes permite detectar y seguir vehículos de forma automatizada y monitorizada mediante cámaras de vigilancia, mejorando la cobertura y precisión en la identificación de infracciones. En tal sentido, al analizar los resultados obtenidos, se evidencia que el aumento en el número de infracciones detectadas no solo responde a la precisión técnica del modelo, sino también a la eliminación de limitaciones humanas como la fatiga, la distracción o el tiempo de reacción. Esto demuestra que la automatización de este proceso permite una vigilancia más objetiva y constante, fortaleciendo la transparencia y la eficiencia en la gestión del tránsito urbano.

Para el objetivo general se planteó mejorar el proceso de registro de infracciones de tránsito mediante la implementación de un sistema de visión computacional en los cruces semafóricos de la ciudad de Trujillo, los resultados obtenidos evidencian una relación significativa entre la implementación del sistema y la mejora del proceso de gestión de infracciones. Durante el desarrollo del sistema, el trabajo de recolección y análisis de videos en entornos urbanos muestra las limitaciones propias del contexto real, como la variación de iluminación, el ruido del tráfico y la imposibilidad de registrar extensas jornadas continuas. Estas condiciones permitieron comprobar en la práctica el comportamiento del sistema ante escenarios cambiantes y validar su capacidad de adaptación y detección continua en distintas circunstancias del tránsito local. Tomando en cuenta lo mencionado, se rechaza la hipótesis nula y se acepta la hipótesis general de investigación, confirmando que la implementación del sistema de visión computacional mejora de manera significativa la gestión del registro de infracciones. Estos resultados son respaldados por Owais et al. (2025), los cuales desarrollaron un marco de monitoreo inteligente con cámaras basadas en inteligencia artificial para la ciudad de Assiut, Egipto, con el propósito de incrementar la cobertura de detección y la eficiencia operativa del control vehicular. Sus resultados muestran una precisión mínima del 95 %, un aumento del 50 % en la tasa de detección por cada 1000 vehículos respecto al monitoreo manual y una reducción del tiempo de respuesta a 2 segundos, evidenciando la efectividad del uso de visión computacional para mejorar los procesos de registro y gestión del tránsito urbano. Asimismo, en la investigación se evidencia que el sistema logra una mejora conjunta en los tres indicadores evaluados: la tasa de infracciones correctamente detectadas (TI) alcanza niveles del 95 % frente al 59 % del método tradicional; el tiempo de registro (TR) se reduce de 9.87 a 3.05 minutos; y el número de infracciones detectadas (NID) pasa de 7 a 349 en una jornada de observación. Estos resultados reflejan una optimización integral del proceso, confirmando la eficacia del modelo de visión computacional frente al registro manual convencional. En esa línea, Gehani (2024) explica que los sistemas de visión computacional procesan el video de cámaras en intersecciones para identificar de forma automática conductas infractoras, sustituyendo la vigilancia manual, disminuyendo errores operativos y acelerando la aplicación de la norma; este fundamento técnico sustenta la mejora simultánea observada en la tasa de detección, la reducción del tiempo de registro y el incremento de casos procesados en nuestro contexto. En tal sentido, se confirma que la adopción de tecnologías de visión computacional representa un paso clave hacia la modernización de los sistemas de tránsito en el contexto local, reduce el margen de error humano y sienta las bases para una mejora en el proceso de registro de infracciones M17, cumpliendo plenamente con el propósito del objetivo general de la investigación.

En el transcurso del desarrollo de la investigación, se presentan ciertas limitaciones que condicionan parcialmente el alcance de los resultados. En primer lugar, el sistema de visión computacional se implementa únicamente en un entorno controlado con videos pregrabados de Trujillo, lo que impide evaluar su rendimiento en condiciones de tráfico real con variaciones de iluminación, clima y flujo vehicular. Además, no se considera la integración directa del sistema con la base de datos oficial de la Policía Nacional ni con la plataforma municipal de infracciones, limitando la validación jurídica de los registros generados. Asimismo, no se evalúa el comportamiento del modelo frente a diferentes tipos de cámaras o resoluciones, lo cual podría afectar la precisión del reconocimiento de matrículas en contextos operativos diversos. Finalmente, el estudio no abarca el análisis de costos ni la proyección económica para una implementación a escala urbana, aspecto necesario para una futura adopción institucional del sistema.

A partir de los resultados obtenidos, este estudio abre una línea prometedora para futuras investigaciones orientadas al fortalecimiento de los sistemas inteligentes de control vehicular en el país. Se considera necesario continuar explorando la aplicación de modelos de visión computacional más avanzados, integrando redes neuronales de detección multiclase y algoritmos de aprendizaje continuo que permitan al sistema adaptarse a nuevos escenarios urbanos sin reentrenamiento completo. Asimismo, resulta relevante implementar proyectos piloto en distintas ciudades para validar el desempeño del sistema en condiciones reales de tránsito, evaluando su precisión ante factores como la lluvia, la congestión o la baja iluminación. De manera especial, se busca motivar a otros investigadores, estudiantes y entidades públicas a seguir profundizando en la aplicación ética y responsable de la visión computacional en la gestión del transporte urbano. Por lo que continuar investigando en esta línea no solo es deseable, sino necesario para construir ciudades más inteligentes, seguras y sostenibles.

Conclusiones

Se mejoró el proceso de registro de infracciones de tránsito en Trujillo mediante la implementación del sistema de visión computacional, logrando una optimización integral en la fiscalización vehicular. La validación estadística y operativa demuestra que la automatización supera las limitaciones del método manual, garantizando una gestión más eficiente, rápida y con mayor cobertura en las intersecciones semaforizadas.

Se aumentó significativamente la tasa de infracciones correctamente detectadas (TI) en los cruces en rojo, alcanzando una precisión del 95 % en el grupo experimental frente al 59 % registrado por el método manual. Este incremento de 36 puntos porcentuales confirma que el sistema de visión computacional posee una capacidad superior para identificar vehículos infractores con alta fiabilidad, reduciendo drásticamente los falsos negativos.

Se disminuyó el tiempo de registro (TR) de infracciones de manera contundente, reduciendo el promedio de 9.87 minutos (método manual) a solo 3.05 minutos con el sistema automatizado. Esta reducción del 69 % evidencia que la tecnología propuesta agiliza la respuesta operativa, eliminando cuellos de botella administrativos y permitiendo un procesamiento de eventos casi en tiempo real.

Se incrementó masivamente el número de infracciones detectadas (NID), pasando de 7 infracciones registradas manualmente a 349 infracciones detectadas por el sistema en una jornada proyectada comparable. Este aumento demuestra que la capacidad de vigilancia continua y sin fatiga de la visión computacional amplía la cobertura de fiscalización a niveles inalcanzables para el recurso humano tradicional.

Recomendaciones

Se recomienda institucionalizar el uso de sistemas de visión computacional en la gestión del tránsito, incorporándolos en la infraestructura tecnológica municipal de Trujillo e implementar un plan de mantenimiento preventivo del sistema, capacitación continua y la integración con bases de datos policiales para fortalecer la trazabilidad de los reportes.

Se recomienda validar el sistema en distintas ciudades y entornos urbanos del país, bajo condiciones variables de iluminación y flujo vehicular, e implementar un módulo de verificación automatizada de falsos positivos para incrementar la confiabilidad del sistema y garantizar la imparcialidad en la emisión de sanciones.

Se recomienda escalar la automatización a un sistema centralizado de monitoreo, conectado con la base de datos de infracciones y el sistema de pago de multas, permitiendo una gestión integral e incorporar alertas inteligentes en tiempo real para detectar fallas en los equipos o infracciones recurrentes, garantizando una respuesta inmediata y continua.

Se recomienda ampliar la instalación del sistema de visión computacional a más intersecciones de alta incidencia, priorizando las zonas con mayor índice de infracciones, evaluar la viabilidad económica y sostenibilidad operativa del sistema para garantizar su implementación a largo plazo en todo el territorio urbano.

Referencias bibliográficas

AGARWAL, P., CHOPRA, K., KASHIF, M. y KUMARI, V., 2018. Implementing ALPR for detection of traffic violations: a step towards sustainability. Procedia Computer Science, vol. 132, pp. 738-743. ISSN 18770509. DOI 10.1016/j.procs.2018.05.085. 

ALBUKHNEFIS, A., FATLAWI, T. y AL-ALSAEEDI, A., 2024. Image Segmentation Techniques: An In-Depth Review and Analysis. Journal of Al-Qadisiyah for Computer Science and Mathematics [en línea], vol. 16, no. 2, [consulta: 30 mayo 2025]. ISSN 2521-3504, 2074-0204. DOI 10.29304/jqcsm.2024.16.21613. Disponible en: https://jqcsm.qu.edu.iq/index.php/journalcm/article/view/1613. 

ALFARANO, A., MAIANO, L., PAPA, L. y AMERINI, I., 2024. Estimating optical flow: A comprehensive review of the state of the art. Computer Vision and Image Understanding, vol. 249, pp. 104160. ISSN 10773142. DOI 10.1016/j.cviu.2024.104160. 

ALIANE, N., FERNANDEZ, J., MATA, M. y BEMPOSTA, S., 2014. A System for Traffic Violation Detection. Sensors, vol. 14, no. 11, pp. 22113-22127. ISSN 1424-8220. DOI 10.3390/s141122113. 

ANG, A., CHRISTENSEN, P. y VIEIRA, R., 2020. Should congested cities reduce their speed limits? Evidence from São Paulo, Brazil. Journal of Public Economics, vol. 184, pp. 104155. ISSN 00472727. DOI 10.1016/j.jpubeco.2020.104155. 

ARCHANA, R. y JEEVARAJ, P.S.E., 2024. Deep learning models for digital image processing: a review. Artificial Intelligence Review, vol. 57, no. 1, pp. 11. ISSN 0269-2821, 1573-7462. DOI 10.1007/s10462-023-10631-z. 

AVCI, İ. y KOCA, M., 2024. Intelligent Transportation System Technologies, Challenges and Security. Applied Sciences, vol. 14, no. 11, pp. 4646. ISSN 2076-3417. DOI 10.3390/app14114646. 

CARDOSO, G.O. y DIAS, I.C.P., 2020. MAPPING PROCESS IMPROVEMENT AND SEQUENCING ANALYSIS FOR PRODUCTIVE DEFINITIONS: ITEGAM-JETIA, vol. 6, no. 21, pp. 66-71. ISSN 2447-0228. 

CERNADAS, E., 2024. Applications of Computer Vision, 2nd Edition. Electronics, vol. 13, no. 18, pp. 3779. ISSN 2079-9292. DOI 10.3390/electronics13183779. 

CHEN, L., LI, S., BAI, Q., YANG, J., JIANG, S. y MIAO, Y., 2021. Review of Image Classification Algorithms Based on Convolutional Neural Networks. Remote Sensing, vol. 13, no. 22, pp. 4712. ISSN 2072-4292. DOI 10.3390/rs13224712. 

COHEN, J., 2009. Statistical power analysis for the behavioral sciences. 2. ed., reprint. New York, NY: Psychology Press. ISBN 978-0-8058-0283-2. 

CORREA, A. y VÍLCHEZ, M., 2022. Implementación del módulo de cinemómetro en el sistema web SITRAN para la gestión de infracciones de tránsito de SUTRAN, Jesús María – 2022 [en línea]. Tesis (Ingeniero de Sistemas). Lima: Universidad César Vallejo. [consulta: 24 mayo 2025]. Disponible en: https://repositorio.ucv.edu.pe/handle/20.500.12692/103958. 

DILEK, E. y DENER, M., 2023. Computer Vision Applications in Intelligent Transportation Systems: A Survey. Sensors, vol. 23, no. 6, pp. 2938. ISSN 1424-8220. DOI 10.3390/s23062938. 

DOBSON, J.E., 2023. The birth of computer vision [en línea]. Minneapolis: University of Minnesota Press. ISBN 978-1-5179-1421-9. Disponible en: https://www.upress.umn.edu/9781517914219/the-birth-of-computer-vision/. TA1634 .D63 2023

DONG, C., LOY, C.C., HE, K. y TANG, X., 2016. Image Super-Resolution Using Deep Convolutional Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 2, pp. 295-307. ISSN 1939-3539. DOI 10.1109/TPAMI.2015.2439281. 

ELGENDY, M., 2020. Deep learning for vision systems. Shelter Island, NY: Manning Publications Co. ISBN 978-1-61729-619-2. TA1634 .E44 2020

FLORES, K.A. y GARCÍA, M.M., 2024. Consequences for non-compliance with traffic regulations for exceeding speed limits in Peru. SCIÉNDO, vol. 27, no. 2, pp. 161-165. ISSN 26173735. DOI 10.17268/sciendo.2024.023. 

GEHANI, H., 2024. Traffic Signal Violation Detection System Using Computer Vision. Journal of Electrical Systems, vol. 20, no. 2, pp. 2661-2670. ISSN 1112-5209. DOI 10.52783/jes.2037. 

GENDY, W. y PATEL, D., 2024. Advancements in Computer Vision: A Comprehensive Survey of Image Processing and Interdisciplinary Applications. Academic Journal of Science and Technology, vol. 13, no. 2, pp. 28-34. ISSN 2771-3032. DOI 10.54097/5e1cqw59. 

GONZÁLEZ, J.F. y PRADA, S.I., 2016. Cámaras de fotodetección y accidentalidad vial. Evidencia para la ciudad de Cali. Desarrollo y Sociedad, no. 77, pp. 131-182. ISSN 1900-7760, 0120-3584. DOI 10.13043/dys.77.4. 

HOSSAIN, M.R., KANG, M.-W. y WU, S., 2025. Engineering Countermeasures for Red-Light Running: A State-of-the-Art Review. Journal of Transportation Technologies, vol. 15, no. 02, pp. 275-311. ISSN 2160-0473, 2160-0481. DOI 10.4236/jtts.2025.152014. 

KANG, J., TARIQ, S., OH, H. y WOO, S.S., 2022. A Survey of Deep Learning-Based Object Detection Methods and Datasets for Overhead Imagery. IEEE Access, vol. 10, pp. 20118-20134. ISSN 2169-3536. DOI 10.1109/ACCESS.2022.3149052. 

KHAN, A., LAGHARI, A. y AWAN, S., 2021. Machine Learning in Computer Vision: A Review. ICST Transactions on Scalable Information Systems, pp. 169418. ISSN 2032-9407. DOI 10.4108/eai.21-4-2021.169418. 

LIU, T. y SALAZAR, D.M., 2021. OpenOpticalFlow_PIV: An Open Source Program Integrating Optical Flow Method with Cross- Correlation Method for Particle Image Velocimetry. Journal of Open Research Software, vol. 9, no. 1, pp. 3. ISSN 2049-9647. DOI 10.5334/jors.326. 

LU, Y., TIAN, Y., WANG, D., CHEN, B. y PENG, X., 2025. DynNPC: Finding More Violations Induced by ADS in Simulation Testing through Dynamic NPC Behavior Generation [en línea]. 24 junio 2025. S.l.: arXiv. [consulta: 18 septiembre 2025]. arXiv:2411.19567. Disponible en: http://arxiv.org/abs/2411.19567. 

LUBNA, MUFTI, N. y SHAH, S.A.A., 2021. Automatic Number Plate Recognition:A Detailed Survey of Relevant Algorithms. Sensors, vol. 21, no. 9, pp. 3028. ISSN 1424-8220. DOI 10.3390/s21093028. 

MAJHI, R.K. y WAOO, A.A., 2024. ADVANCES IN COMPUTER VISION: NEW HORIZONS AND ONGOING CHALLENGES. ShodhKosh: Journal of Visual and Performing Arts [en línea], vol. 5, no. 5, [consulta: 1 junio 2025]. ISSN 2582-7472. DOI 10.29121/shodhkosh.v5.i5.2024.1893. Disponible en: https://www.granthaalayahpublication.org/Arts-Journal/ShodhKosh/article/view/1893. 

MINAEE, S., BOYKOV, Y.Y., PORIKLI, F., PLAZA, A.J., KEHTARNAVAZ, N. y TERZOPOULOS, D., 2021. Image Segmentation Using Deep Learning: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1-1. ISSN 0162-8828, 2160-9292, 1939-3539. DOI 10.1109/TPAMI.2021.3059968. 

MINISTERIO DE TRANSPORTES Y COMUNICACIONES (MTC)., 2014. Decreto Supremo N.o 016-2009-MTC, Aprueba el Texto Único Ordenado del Reglamento Nacional de Tránsito – Código de Tránsito [en línea]. 22 abril 2014. S.l.: s.n. [consulta: 15 mayo 2024]. Disponible en: http://www.sutran.gob.pe/wp-content/uploads/2020/06/Texto-%C3%9Anico-Ordenado-del-Reglamento-Nacional-de-Tr%C3%A1nsito-DS-N%C2%BA-016-2009-MTC.pdf. 

OWAIS, M., SHEHATA, A., SHABAND, A. y MOUSSA, G., 2025. A Framework for Establishing an Automated Traffic Violation Detection System in New Assiut City Using Ordinary CCTV Units. MEJ Mansoura Engineering Journal, vol. 50, DOI 10.58491/2735-4202.3293. 

PRADHAN, G., PRUSTY, M.R., NEGI, V.S. y CHINARA, S., 2025. Advanced IoT-integrated parking systems with automated license plate recognition and payment management. Scientific Reports, vol. 15, no. 1, pp. 2388. ISSN 2045-2322. DOI 10.1038/s41598-025-86441-w. 

QIAO, X., 2023. Research on Traffic sign recognition based on CNN Deep Learning Network. Procedia Computer Science, vol. 228, pp. 826-837. ISSN 1877-0509. DOI 10.1016/j.procs.2023.11.102. 

RASHED, B.M. y POPESCU, N., 2022. Critical Analysis of the Current Medical Image-Based Processing Techniques for Automatic Disease Evaluation: Systematic Literature Review. Sensors, vol. 22, no. 18, pp. 7065. ISSN 1424-8220. DOI 10.3390/s22187065. 

REN, Y., 2024. Intelligent Vehicle Violation Detection System Under Human–Computer Interaction and Computer Vision. International Journal of Computational Intelligence Systems, vol. 17, no. 1, pp. 40. ISSN 1875-6883. DOI 10.1007/s44196-024-00427-6. 

SENKUS, P., GLABISZEWSKI, W., WYSOKINSKA-SENKUS, A. y PANKA, A., 2021. Process Definitions - Critical Literature Review. EUROPEAN RESEARCH STUDIES JOURNAL, vol. XXIV, no. Issue 3, pp. 241-255. ISSN 1108-2976. DOI 10.35808/ersj/2352. 

SINGH, A. y SINGH, P., 2020. Image Classification: A Survey. Journal of Informatics Electrical and Electronics Engineering (JIEEE), vol. 1, no. 2, pp. 1-9. ISSN 25827006. DOI 10.54060/JIEEE/001.02.002. 

SLIMANI, I., ZAARANE, A., AL OKAISHI, W., ATOUF, I. y HAMDOUN, A., 2020. An automated license plate detection and recognition system based on wavelet decomposition and CNN. Array [en línea], vol. 8, pp. 100040. [consulta: 5 mayo 2025]. ISSN 2590-0056. DOI 10.1016/j.array.2020.100040. Disponible en: https://www.sciencedirect.com/science/article/pii/S2590005620300254. 

SZELISKI, R., 2022. Computer Vision: Algorithms and Applications [en línea]. Cham: Springer International Publishing. [consulta: 18 septiembre 2025]. Texts in Computer Science, ISBN 978-3-030-34371-2. Disponible en: https://link.springer.com/10.1007/978-3-030-34372-9. 

TEJADA OLIVERA, R.W., 2024. Valor compartido para una eficaz y eficiente gestión de multas de tránsito en la ciudad de Chiclayo [en línea]. Maestría. Chiclayo: Pontificia Universidad Católica del Perú. [consulta: 13 mayo 2025]. Disponible en: http://hdl.handle.net/20.500.12404/27709. 

THAO, L.Q., CUONG, D.D., ANH, N.T., ANH, P.M., DUC, H.M. y MINH, N., 2022. Automatic Traffic Red-Light Violation Detection Using AI. Ingénierie des systèmes d information, vol. 27, no. 1, pp. 75-80. ISSN 16331311, 21167125. DOI 10.18280/isi.270109. 

THOMA, M., 2017. Analysis and Optimization of Convolutional Neural Network Architectures. En: arXiv:1707.09725 [cs] [en línea], [consulta: 12 mayo 2025]. DOI 10.48550/arXiv.1707.09725. Disponible en: http://arxiv.org/abs/1707.09725. 

TSIRTSAKIS, P., ZACHARIS, G., MARASLIDIS, G.S. y FRAGULIS, G.F., 2025. Deep learning for object recognition: A comprehensive review of models and algorithms. International Journal of Cognitive Computing in Engineering, vol. 6, pp. 298-312. DOI 10.1016/j.ijcce.2025.01.004. Scopus

UPADHYAY, U. y GUPTA, S., 2024. A Survey on Image Feature Extraction Techniques. International Journal of Scientific Research, vol. 10, no. 3, 

VARGOORANI, Z.E. y SUEN, C.Y., 2024. License Plate Detection and Character Recognition Using Deep Learning and Font Evaluation. En: arXiv:2412.12572 [cs], vol. 15154, pp. 231-242. DOI 10.1007/978-3-031-71602-7_20. 

WILEY, V. y LUCAS, T., 2018. Computer Vision and Image Processing: A Paper Review. International Journal of Artificial Intelligence Research, vol. 2, no. 1, pp. 22. ISSN 2579-7298. DOI 10.29099/ijair.v2i1.42. 

YADAV, S. y SAWALE, M.D., 2023. A review on image classification using deep learning. World Journal of Advanced Research and Reviews, vol. 17, no. 1, pp. 480-482. ISSN 25819615. DOI 10.30574/wjarr.2023.17.1.0064. 

YANG, X. y WANG, X., 2019. Recognizing License Plates in Real-Time. [en línea], [consulta: 24 junio 2025]. Disponible en: https://arxiv.org/abs/1906.04376. 

YASANTHI, R.G.N., WICKENS, C.M., JONAH, B., MEHRAN, B. y SUGGETT, B., 2024. Determinants of traffic safety enforcement behaviour among police officers: A narrative review. Case Studies on Transport Policy, vol. 16, pp. 101206. ISSN 2213624X. DOI 10.1016/j.cstp.2024.101206. 

YOUSEF, K.M.A., MOHD, B.J., AL-KHALAILEH, Y.A.-A.-H., AL-HMEADAT, A.H. y EL-ZIQ, B.I., 2020. Automatic license plate detection and recognition for jordanian vehicles. Advances in Science, Technology and Engineering Systems, vol. 5, no. 6, pp. 699-709. DOI 10.25046/aj050684. Scopus

ZHAO, Z.-Q., ZHENG, P., XU, Shou-Tao y WU, X., 2019. Object Detection With Deep Learning: A Review. IEEE Transactions on Neural Networks and Learning Systems [en línea], vol. 30, no. 11, pp. 3212-3232. [consulta: 3 mayo 2025]. ISSN 2162-2388. DOI 10.1109/TNNLS.2018.2876865. Disponible en: https://ieeexplore.ieee.org/document/8627998. 

ZOU, Z., CHEN, K., SHI, Z., GUO, Y. y YE, J., 2023. Object Detection in 20 Years: A Survey [en línea]. 18 enero 2023. S.l.: arXiv. [consulta: 18 septiembre 2025]. arXiv:1905.05055. Disponible en: http://arxiv.org/abs/1905.05055. 

Anexos

Anexo 1. Instrumentos de recolección de datos.

Anexo 1. Ficha de Observación del GC para el indicador Número de Infracciones Detectadas (NID)

Anexo 2. Ficha de Observación del GC para el indicador Tasa de Infracciones (TI)

Anexo 3. Ficha de Observación del GC para el indicador Tiempo de Registro (TR)

Anexo 4. Cuadro de registros y resultados de cálculos para todos los indicadores.

Anexo 5. Encuesta realizada para la recolección de datos a la policía de tránsito.

Anexo 2. Evidencias de la ejecución de la investigación.

Anexo 6. Ejecución del software para el GE en la comisaria de la Noria, octubre 2025.

Anexo 7. Ejecución del software para el GE en la comisaria de la Noria, octubre 2025.

Anexo 8. Ejecución del software para el GE en la comisaria de la Noria, octubre 2025.

Anexo 9. Ejecución del software para el GE en la comisaria de la Noria, octubre 2025.

Anexo 10. Ejecución del software para el GE en la comisaria de la Noria, octubre 2025.

Anexo 11. Ejecución del software para el GE en la comisaria de la Noria, octubre 2025.

Anexo 3. Metodología de desarrollo de software.

Planificación (Anexo 11 al 16)

En esta fase se definieron los requerimientos del sistema, identificando las funcionalidades principales y los módulos que compondrían el software. Se elaboró la metodología de desarrollo y se documentó la aplicación de la metodología XP (Anexo 11), donde se establecieron las historias de usuario, los criterios de aceptación y las iteraciones semanales. Asimismo, se incluyeron las solicitudes realizadas a la comandante Milagros Quispe, al Centro de Monitoreo de la ciudad de Trujillo y a la División de Tránsito, así como la relación de cámaras obtenidas para el análisis de datos.  Esta etapa permitió alinear los objetivos funcionales del sistema con los indicadores de la investigación.

Anexo 12. Metodología XP usada para el desarrollo del software de Tesis.

Anexo 13. Arquitectura de Componentes del Software.

Tabla 18. Tabla de módulos desarrollados con sus criterios de aceptación.

Módulo

Objetivo técnico

Criterios de aceptación

Módulo de detección de infracciones 

Desarrollar el núcleo de detección de vehículos y cruces en rojo utilizando YOLOv8.

El sistema detecta con una precisión mínima del 90 % los vehículos que cruzan en rojo en videos de prueba y genera alertas automáticas de infracción.

Módulo de gestión de infracciones

Implementar el registro, almacenamiento y visualización de las infracciones detectadas.

Cada infracción se guarda correctamente con su placa, hora, fecha y evidencia visual; permite consultar y filtrar los registros sin errores de carga.

Módulo de bienvenida e interfaz de usuario 

Diseñar la pantalla inicial y la navegación entre los módulos principales del sistema.

El sistema inicia sin errores, muestra el nombre del proyecto y permite acceder a los módulos de detección y gestión mediante botones activos y rutas funcionales.

Fuente. Elaborado por los autores en base a la metodología XP.

Anexo 14. Solicitud realizada a la comandante policial Milagros Quispe Alvarado de la unidad de tránsito.

Anexo 15. Solicitud realizada al centro de monitoreo de la ciudad de Trujillo.

Anexo 16. Relación de cameras obtenidas del centro de monitoreo para la investigación.

Anexo 17. Solicitud a la división de tránsito PNP Trujillo.

Diseño (Anexos 18 al 22)

Durante la fase de diseño, se realizaron los diagramas de secuencia (Anexo 16) y de arquitectura de componentes (Anexo 17), además se desarrollaron los prototipos de las interfaces gráficas para cada módulo del sistema: el módulo de detección de infracciones (Anexo 18), el módulo de gestión (Anexo 19) y la pantalla de bienvenida (Anexo 20). Estos prototipos sirvieron como base visual para validar la usabilidad y el flujo del sistema antes de la codificación. Asimismo, se integró el repositorio GitHub (Anexo 21) como herramienta colaborativa para el control de versiones.

Anexo 18. Diagrama de secuencia del software.

Anexo 19. Arquitectura de componentes del software.

Anexo 20. Prototipo del módulo de detección de infracciones.

Anexo 20. Prototipo del módulo de gestión de infracciones.

Anexo 21. Prototipo del módulo de bienvenida del sistema.

Anexo 22. Github utilizado para la iteración del sistema.

Codificación e Iteración (Anexo 23 al 32)

En esta fase se implementaron los diferentes módulos del sistema mediante programación en Python. Los archivos del Anexo 16 en adelante documentan los componentes principales del software, como el procesamiento de video (preprocessing_dialog.py), la detección de vehículos (vehicle_detector.py), el reconocimiento de placas (anpr.py, recognizer.py, plate_recognizer.py), la mejora de resolución (superresolution.py), la reproducción de video con análisis inteligente (videoplayer_opencv.py) y la gestión central del sistema (app_manager.py, main.py). Esta etapa reflejó la aplicación de buenas prácticas XP, como la codificación en pares y la integración continua.

Anexo 23. Archivo preprocessing_dialog.py encargado de cargar los modelos de visión computacional y hacer el procesamiento.

Anexo 24. Archivo anpr.py encargado de detectar placas vehiculares en videos y extraer su texto mediante OCR con validación SIIV.

Anexo 25. Archivo plate_recognizer.py su función es reconocer el texto de una placa vehicular usando LPRNet con PyTorch.

Anexo 26. Archivo vehicle_detector.py encargado de detectar vehículos en los videos.

Anexo 27. Archivo superresolution.py encargado de mejorar la calidad de imágenes de placas vehiculares.

Anexo 28. Archivo videoplayer_opencv.py reproductor de video inteligente con detección de infracciones M17.

Anexo 29. Archivo app_manager.py gestor central de navegación entre las 3 pantallas principales de la aplicación.

Anexo 30. Archivo video_selector_window.py maneja la interfaz gráfica de selección de videos.

Anexo 31. Archivo recognizer.py ayuda a reconocer placas vehiculares con OCR, usando PaddleOCR.

Anexo 32. Archivo main.py inicializa la ventana principal, precarga PaddleOCR y lanza la aplicación.

Pruebas (Anexos 33 al 39)

Se realizaron pruebas funcionales y de integración sobre los módulos finalizados, verificando la detección de infracciones, la precisión del OCR y la gestión de datos en el sistema. Los Anexos 32 al 39 evidencian los resultados de las iteraciones de prueba: módulo de bienvenida, detección de infracciones, configuración de videos, análisis automatizado y visualización de infracciones con placa. Cada módulo fue ajustado tras las pruebas de usuario y revisión de código, cumpliendo los criterios de calidad establecidos.

Anexo 33. Módulo finalizado de bienvenida del sistema.

Anexo 34. Módulo de detección de infracciones.

Anexo 35. Módulo de configuración de videos.

Anexo 36. Módulo de análisis de video para detectar infracciones.

Anexo 37. Módulo de visualización de la infracción con placa e indicadores.

Anexo 38. Módulo de gestión de infracciones culminado.

Anexo 39. Modelo PaddleOCR usado para las pruebas del software cargado exitosamente.

Anexo 40. Pruebas unitarias del software funcional.

Implementación y mantenimiento (Anexo 41 y 42)

Finalmente, se desarrolló la fase de migración a la nube (Anexo 40 y 1) mediante la plataforma Firebase, garantizando la persistencia, accesibilidad y escalabilidad del sistema. Esta fase consolidó la última iteración de XP, enfocada en la entrega continua, la documentación del código y la preparación para futuras mejoras o despliegues en entornos reales.

Anexo 41. Módulo de migraciones a la nube en Firebase.

Anexo 42. Firestore con los resultados guardados del GE.