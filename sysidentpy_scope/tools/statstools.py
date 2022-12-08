from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
from statsmodels.tsa import stattools


def correlation(y, u=None, confidence=0.95, interval_correction=False):
    """Referencias
    [1] Dangers and uses of cross-correlation in analyzing time series
    in perception, performance, movement, and neuroscience: The
    importance of constructing transfer function autoregressive models.
    Doi: 10.3758/s13428-015-0611-2
    """

    # Prewhitening
    # Dados normalizados possuem menor probabilidade de apresentar
    # correlação espuria [1].
    y_aux = (y - np.mean(y))/np.std(y)
    if u is None:
        u_aux = deepcopy(y_aux)
    else:
        u_aux = (u - np.mean(u))/np.std(u)

    # Seta os tempos discretos "k" indo de -(len(y)-1) a len(y)
    # k<0 indica a correlação de u com y (r_uy)
    # k>0 indica a correlação de y com u (r_yu)
    # k=0 representa a correlação de Pearson
    k = np.arange(-1*(len(y)-1), len(y))
    if u is None: #autocorrelação (r_uy = r_yu)
        k_0 = int(len(k)/2)
    else: #correlação cruzada
        k_0 = 0

    # Calculo da correlação (covariância)
    correlation_value = np.correlate(y_aux, u_aux, mode='full')
    correlation_value /= len(y)


    # Determinação do intervalo de confiança.
    # Como base, toma-se como intervalo +/- Z_score(confianca) / sqrt(N)
    confidence_interval = np.asarray(spst.norm.interval(confidence))
    confidence_interval /= np.sqrt(len(y))

    # Como a função de transferência para o sistema é desconhecido,
    # então não é possível corrigir a correlação, portanto, o intervalo
    # de confiança é corrigido por meio dos coeficientes de
    # autocorrelação dos sinais, auemntando a largura do intervalo de
    # confiança e assim reduzindo-se a probabilidade de considerar
    # correlações espurias. [1]
    if interval_correction:
        def _get_autocorr_coef(s):
            return stattools.acf(s, fft=False, nlags=1)[1]

        ab = _get_autocorr_coef(y_aux) * _get_autocorr_coef(u_aux)
        correction = np.sqrt((1 + ab)/(1 - ab))
        confidence_interval *= correction

    return correlation_value[k_0:], k[k_0:], confidence_interval


def plot_correlation(correlation_values, k, limits, title=""):
    """Gera um plot com a correlação cruzada entre sinais
    ou autocorrelação de um sinal único

    Args:
        correlation_values (_type_): Valores de correlação
        k (_type_): Tempo (parametrização do sinal)
        limits (_type_): Limites superior e inferior para o intervalo de confiança
        title (str, optional): Título do gráfico. Defaults to "".
    """
    plt.figure()
    plt.plot(k, correlation_values, 'b-')
    for limit in limits:
        plt.axhline(y=limit, color='black', linestyle='--')
    plt.grid()
    plt.title(title)
    plt.show()
