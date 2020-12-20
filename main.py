import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.descriptivestats as smsd
import statsmodels.stats.gof as smsg
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from scipy import stats
from statsmodels.graphics.regressionplots import plot_leverage_resid2


# Получение данных
def get_data(par=3):
    nsample = 100
    x = np.linspace(0, 10, 100)
    X = np.column_stack((x, x ** 2))
    for i in range(3, par):
        X = np.column_stack((X, x ** i))
    con = []
    for i in range(par):
        con.append(1 / 10 ** i)
    beta = np.asarray(con)
    e = np.random.normal(size=nsample)
    X = sm.add_constant(X)
    y = np.dot(X, beta) + e
    return y, X


def check_mean_residuals_zero(res):
    return res.resid.mean()


def homoscedasticity_test(res):
    name = ['Lagrange multiplier statistic', 'p-value',
            'f-value', 'f p-value']
    bp_test = sms.het_breuschpagan(res.resid, res.model.exog)
    print('\n Breusch-Pagan test ----')
    print(lzip(name, bp_test))
    name = ['F statistic', 'p-value']
    gq_test = sms.het_goldfeldquandt(res.resid, res.model.exog)
    print('\n Goldfeld-Quandt test ----')
    print(lzip(name, gq_test))


def normality_of_residuals_test(model):
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    jb_test = sms.jarque_bera(model.resid)
    print(lzip(name, jb_test))
    print('\n White test ----')
    name = ['Lagrange multiplier statistic', 'p-value',
            'f-value', 'f p-value']
    w_test = sms.het_white(model.resid, model.model.exog)
    print(lzip(name, w_test))


def goodness_test(model):
    print(smsg.gof_chisquare_discrete(model))


def residual_autocorrelation_test(model):
    name = ['Lagrange multiplier statistic', 'p-value',
            'f-value', 'f p-value']
    bg_test = sms.acorr_breusch_godfrey(model)
    print('\n Breusch-Godfrey test ----')
    print(lzip(name, bg_test))


def draw_resid(res):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig = plot_leverage_resid2(res, ax=ax)
    plt.show()
    plt.scatter(res.model.exog[:, 1], res.resid)
    plt.show()


if __name__ == '__main__':
    y, X = get_data(4)
    model = sm.OLS(y, X)
    res = model.fit()

    # Оценки регрессионной модели
    print(res.summary())

    # График остатков
    draw_resid(res)

    # Проверка гипотезы о нормальности остатков регрессионной модели
    normality_of_residuals_test(res)

    # проверка адекватности модели
    print(res.fvalue)

    # проверка значимости модели
    print(res.pvalues)

    # Проверка предположения об отсутствии систематической погрешности
    print(check_mean_residuals_zero)

    # гипотеза о гомоскедастичности остатков
    homoscedasticity_test(res)

    # проверка гипотезы об отсутствии автокорреляции ошибок
    residual_autocorrelation_test(res)

    # Учет гетероскедастичности посредсвом весов
    new_model = sm.WLS(y, X)
    new_res = model.fit()
    print(new_res.summary())
    print(res.params)
    print(new_res.params)

    # Учет автокоррелированности (Ньюи-Вест)
    new = res.get_robustcov_results(cov_type='HAC', maxlags=1)
    print(new.summary())

    # Учет автокоррелированности и гетероскедастичности
    calc_model = sm.WLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    print(calc_model.summary())
