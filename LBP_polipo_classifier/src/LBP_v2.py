import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from utils.utils import get_LBP_describe, plot_validation, plot_image, plot_cm

imgs = glob(os.path.join('data', 'train', '*.jpg'))
imgs_test = glob(os.path.join('data', 'test', '*.jpg'))


# Extracao de caracteristicas - LBP
labels = []
data = []
for imgnm in imgs:
    gray, lbp, desc = get_LBP_describe(imgnm)

    # plot_image(gray, lbp, desc)

    labels.append(imgnm.split('\\')[2].split('.')[0].split()[0])
    data.append(desc)


# Treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.3
    )


# Modelo
model = SVC()
parameters = {
    'C': np.arange(0.01, 100, 10),
    'gamma': np.arange(100, 0.001, -10),
    'kernel': ['linear', 'rbf']
    }

grid_search = GridSearchCV(estimator=model,
                           param_grid=parameters,
                           scoring='accuracy',
                           verbose=2,
                           n_jobs=-1)

# Treinamento
grid_search.fit(x_train, y_train)

print('GridSearch CV best score: {:.4f}\n'.format(grid_search.best_score_))
print(f'Melhores Paremetros: {grid_search.best_params_}')


# Predicoes
predictions = grid_search.predict(x_test)

# Validação
plot_cm(y_test, predictions)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
plot_validation(model, x_train, y_train)

# Teste Real
for imgnm_teste in imgs_test:
    _, _, hist = get_LBP_describe(imgnm_teste)
    prediction = grid_search.predict(hist.reshape(1, -1))
    print(prediction[0])
