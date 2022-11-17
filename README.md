# Customer Affinity for Loan

### Business Case:
La empresa analizada corresponde a una Institucion Bancaria Portuguesa; por lo tanto los datos de este Dataset estan relacionados con campañas de marketing directo (llamadas telefónicas) de la misma. La institución está interesada en predecir si un cliente será propenso a realizar un depósito a plazo fijo en el banco a partir de los datos obtenidos.
<br>
<br>
### Objetivos del análisis:
A partir de la Campana de marketing intentamos entender los pratrones de comportamiento de cada usuario para poder predecir si un futuro cliente va a acceder a un deposito a plaza fijo. Con esta informacion, la empresa estará en condiciones de segmentar correctamente sus campañas publicitarias.
<br>
<br>
### Descripcion de los datos:
El Dataset está formado por más de 40.000 registros, donde cada uno corresponde a una llamada teléfonica realizada en campaña de marketing anteriores. Se cuenta con una variable Target (y), la cual indica si un cliente se suscribió o no a un depósito a plazo fijo. 
<br>
<br>
### Attribute Information:
#### Bank client data:

Age (numeric)

Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')

Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)

Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')

Default: has credit in default? (categorical: 'no', 'yes', 'unknown')

Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')

Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')

#### Related with the last contact of the current campaign:

Contact: contact communication type (categorical:
'cellular','telephone') Month: last contact month of year (categorical: 'jan', 'feb', 'mar', …, 'nov', 'dec')

Dayofweek: last contact day of the week (categorical:
'mon','tue','wed','thu','fri')

Duration: last contact duration, in seconds (numeric). Important
note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#### Other attributes:

Campaign: number of contacts performed during this campaign and for
this client (numeric, includes last contact)

Pdays: number of days that passed by after the client was last
contacted from a previous campaign (numeric; 999 means client was not previously contacted) Previous: number of contacts performed before this campaign and for this client (numeric)

Poutcome: outcome of the previous marketing campaign (categorical:
'failure','nonexistent','success')

#### Social and economic context attributes:

Emp.var.rate: employment variation rate - quarterly indicator
(numeric)

Cons.price.idx: consumer price index - monthly indicator (numeric)
Cons.conf.idx: consumer confidence index - monthly indicator
(numeric)

Euribor3m: euribor 3 month rate - daily indicator (numeric)

Nr.employed: number of employees - quarterly indicator (numeric)

#### Output variable (desired target):

y - has the client subscribed a term deposit? (binary: 'yes', 'no')
<br>
<br>
### Modelos:
Se definió entrenar 7 modelos distintos para la predicción de la variable binaria target (y_yes: indica si una persona realizará un depósito a plazo o no). 
<br>
<br>
### Entrenamiento del Modelo:
Para el entrenamiento del modelo usamos el método Stratified K Fold y la librería Optuna para encontrar los mejores hiperparámetros.
<br>
<br>
### Metricas:
Utilizamos la métrica Recall ya que permite lidiar correctamente con datasets desbalanceadas. Ayuda a responder la pregunta ¿Qué porcentaje de los clientes que sí están interesados, somos capaces de identificar? (no es relevante identificar los que NO, porque son la gran mayoría).

