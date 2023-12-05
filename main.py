from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

# Einlesen der Daten aus der CSV-Datei
data = pd.read_csv("DS_PROB_1/p001_1.csv", delimiter=";")

# Definition des Bayes-Netzwerks
model = BayesianNetwork([('Qualifikation', 'Studierfähigkeitstest'),
                       ('Qualifikation', 'Mathe'),
                       ('Qualifikation', 'Physik'),
                       ('Qualifikation', 'Deutsch'),
                       ('Qualifikation', 'Schultyp'),
                       ('Studierfähigkeitstest', 'Studiengang'),
                       ('Mathe', 'Studiengang'),
                       ('Physik', 'Studiengang'),
                       ('Deutsch', 'Studiengang'),
                       ('Schultyp', 'Studiengang'),
                       ('Alter', 'Studiengang'),
                       ('Geschlecht', 'Studiengang'),
                       ('Jahreseinkommen der Eltern', 'Studiengang'),
                       ('Staatsbürgerschaft', 'Studiengang'),
                       ('Studiengang', 'Abschluss')])

# Trainieren des Modells mit Maximum-Likelihood-Schätzern
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Bayes-Netzwerk-Inferenz
inference = VariableElimination(model)

# Beispielabfrage für Erfolgsaussichten
query = inference.map_query(variables=['Studiengang'], evidence={'Qualifikation': 'Abitur', 'Mathe': 2.0, 'Physik': 2.0, 'Deutsch': 2.0})
print(query)
