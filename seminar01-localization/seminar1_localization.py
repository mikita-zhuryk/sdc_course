#!/usr/bin/env python
# coding: utf-8

# $\newcommand{\Normal}{\mathcal{N}}
# \newcommand{\lp}{\left(}
# \newcommand{\rp}{\right)}
# \newcommand{\lf}{\left\{}
# \newcommand{\rf}{\right\}}
# \newcommand{\ls}{\left[}
# \newcommand{\rs}{\right]}
# \newcommand{\lv}{\left|}
# \newcommand{\rv}{\right|}
# \newcommand{\state}{x}
# \newcommand{\State}{\boldx}
# \newcommand{\StateR}{\boldX}
# \newcommand{\Covariance}{\Sigma}
# \newcommand{\CovX}{\Covariance_{\boldX}}
# \newcommand{\CovY}{\Covariance_{\boldY}}
# \newcommand{\CovZ}{\Covariance_{\boldZ}}
# \newcommand{\CovXY}{\Covariance_{\boldX\boldY}}
# \newcommand{\hatState}{\hat{\State}}
# \newcommand{\StateNum}{N}
# \newcommand{\StateDim}{K}
# \newcommand{\NumStates}{N}
# \newcommand{\StateToState}{A}
# \newcommand{\stateToState}{A}
# \newcommand{\StateCov}{\Sigma}
# \newcommand{\StateJac}{A}
# \newcommand{\hatStateCov}{\hat{\StateCov}}
# \newcommand{\StateMean}{\boldmu}
# \newcommand{\hatStateMean}{\hat{\StateMean}}
# \newcommand{\StateToStateHistory}{\boldA}
# \newcommand{\StateNoise}{\boldr}
# \newcommand{\StateNoiseCov}{R}
# \newcommand{\StateHistory}{\boldX}
# \newcommand{\StatesHistory}{\StateHistory}
# \newcommand{\StateToObserv}{C}
# \newcommand{\StateToobserv}{\boldc}
# \newcommand{\StateToObservHistory}{\boldC}
# \newcommand{\DState}{\bolddelta}
# \newcommand{\hatDState}{\hat{\DState}}
# \newcommand{\DStateMean}{\boldlambda}
# \newcommand{\hatDStateMean}{\hat{\DStateMean}}
# \newcommand{\DStateCov}{\Lambda}
# \newcommand{\hatDStateCov}{\hat{\DStateCov}}
# \newcommand{\DObserv}{\boldgamma}
# \newcommand{\hatDObserv}{\hat{\DObserv}}
# \newcommand{\observ}{z}
# \newcommand{\Observ}{\boldsymbol{\observ}}
# \newcommand{\ObservCov}{\Lambda}
# \newcommand{\observMean}{\lambda}
# \newcommand{\ObservMean}{\boldlambda}
# \newcommand{\hatobserv}{\hat{\observ}}
# \newcommand{\hatObserv}{\hat{\Observ}}
# \newcommand{\hatObservCov}{\hat{\ObservCov}}
# \newcommand{\hatobservMean}{\hat{\observMean}}
# \newcommand{\hatObservMean}{\hat{\ObservMean}}
# \newcommand{\ObservSet}{\ZZ}
# \newcommand{\ObservNum}{N}
# \newcommand{\ObservDim}{D}
# \newcommand{\ObservSourceNum}{M}
# \newcommand{\ObservHistory}{\boldZ}
# \newcommand{\ObservsHistory}{\ObservHistory}
# \newcommand{\Timestamps}{\boldT}
# \newcommand{\ObservJac}{H}
# \newcommand{\observNoise}{q}
# \newcommand{\ObservNoise}{\boldq}
# \newcommand{\ObservNoiseCov}{Q}
# \newcommand{\ObservNoiseCovHistory}{\boldQ}
# \newcommand{\Jacobian}{\boldJ}
# \newcommand{\Kalman}{K}
# \newcommand{\kalman}{\boldk}
# \newcommand{\CC}{\mathbb{C}}
# \newcommand{\NN}{\mathbb{N}}
# \newcommand{\RR}{\mathbb{R}}
# \newcommand{\XX}{\mathbb{X}}
# \newcommand{\ZZ}{\mathbb{Z}}
# \renewcommand{\AA}{\mathbb{A}}
# \newcommand{\boldzero}{\boldsymbol{0}}
# \newcommand{\boldone}{\boldsymbol{1}}
# \newcommand{\bolda}{\boldsymbol{a}}
# \newcommand{\boldb}{\boldsymbol{b}}
# \newcommand{\boldc}{\boldsymbol{c}}
# \newcommand{\boldd}{\boldsymbol{d}}
# \newcommand{\bolde}{\boldsymbol{e}}
# \newcommand{\boldf}{\boldsymbol{f}}
# \newcommand{\boldg}{\boldsymbol{g}}
# \newcommand{\boldh}{\boldsymbol{h}}
# \newcommand{\boldi}{\boldsymbol{i}}
# \newcommand{\boldj}{\boldsymbol{j}}
# \newcommand{\boldk}{\boldsymbol{k}}
# \newcommand{\boldl}{\boldsymbol{l}}
# \newcommand{\boldm}{\boldsymbol{m}}
# \newcommand{\boldn}{\boldsymbol{n}}
# \newcommand{\boldo}{\boldsymbol{o}}
# \newcommand{\boldp}{\boldsymbol{p}}
# \newcommand{\boldq}{\boldsymbol{q}}
# \newcommand{\boldr}{\boldsymbol{r}}
# \newcommand{\bolds}{\boldsymbol{s}}
# \newcommand{\boldt}{\boldsymbol{t}}
# \newcommand{\boldu}{\boldsymbol{u}}
# \newcommand{\boldv}{\boldsymbol{v}}
# \newcommand{\boldw}{\boldsymbol{w}}
# \newcommand{\boldx}{\boldsymbol{x}}
# \newcommand{\boldy}{\boldsymbol{y}}
# \newcommand{\boldz}{\boldsymbol{z}}
# \newcommand{\boldA}{\boldsymbol{A}}
# \newcommand{\boldB}{\boldsymbol{B}}
# \newcommand{\boldC}{\boldsymbol{C}}
# \newcommand{\boldD}{\boldsymbol{D}}
# \newcommand{\boldE}{\boldsymbol{E}}
# \newcommand{\boldF}{\boldsymbol{F}}
# \newcommand{\boldH}{\boldsymbol{H}}
# \newcommand{\boldJ}{\boldsymbol{J}}
# \newcommand{\boldK}{\boldsymbol{K}}
# \newcommand{\boldL}{\boldsymbol{L}}
# \newcommand{\boldM}{\boldsymbol{M}}
# \newcommand{\boldI}{\boldsymbol{I}}
# \newcommand{\boldP}{\boldsymbol{P}}
# \newcommand{\boldQ}{\boldsymbol{Q}}
# \newcommand{\boldR}{\boldsymbol{R}}
# \newcommand{\boldS}{\boldsymbol{S}}
# \newcommand{\boldT}{\boldsymbol{T}}
# \newcommand{\boldO}{\boldsymbol{O}}
# \newcommand{\boldU}{\boldsymbol{U}}
# \newcommand{\boldV}{\boldsymbol{V}}
# \newcommand{\boldW}{\boldsymbol{W}}
# \newcommand{\boldX}{\boldsymbol{X}}
# \newcommand{\boldY}{\boldsymbol{Y}}
# \newcommand{\boldZ}{\boldsymbol{Z}}
# \newcommand{\boldXY}{\boldsymbol{XY}}
# \newcommand{\boldmu}{\boldsymbol{\mu}}$

# <a id='toc'></a>
# # Баейесовская фильтрация и задача локализации
# * [Фильтр Калмана](#kalman_filter)
#     * [Понятие фильтрации](#kalman_intro)
#     * [Линейный фильтр Калмана](#kalman_filter_linear)
#     * [Вывод в фильтре Калмана](#kalman_filter_der)
#     * [Оптимальность фильтра Калмана](#kalman_filter_opt)
#     * [Формулы обновления прогноза в фильтре Калмана](#kalman_filter_results)
# * [Теоретическая модель калмановской локализации](#localization_model)
#     * [Задача локализации](#localization_task)
#     * [Фильтр Калмана в локализации](#localization_kalman)
#     * [1D модель движения материальной точки](#localization_kalman_1d_example)
#     * [2D модель движения материальной точки](#2d_point_movement)
#     * [Альтернативная 2D модель движения материальной точки (для семинара и ДЗ)](#2d_with_yaw_model)
# * [Семинар](#seminar)
#     * [Программная модель движения автомобиля](#car_model_architecture)
#     * [Программная модель калмановской локализации](#kalman_car_model_architecture)
#     * [Визуализация](#visualization)
#     * [Тестирование калмановской фильтрации](#test_kalman_localization)
# * [Реальные данные](#real_data)
# * [Домашнее задание](#assignment)
#     * [Реализация методов фильтра Калмана](#implement_kalman_methods)
#     * [Оценка качества локализации](#localization_quality_estimation)
#     * [Использование IMU](#use_imu)

# <a id='kalman_filter'></a>
# # Фильтр Калмана<sup>[toc](#toc)</sup>
# * [Понятие фильтрации](#kalman_intro)
# * [Линейный фильтр Калмана](#kalman_filter_linear)
# * [Вывод в фильтре Калмана](#kalman_filter_der)
# * [Оптимальность фильтра Калмана](#kalman_filter_opt)
# * [Формулы обновления прогноза в фильтре Калмана](#kalman_filter_results)

# <a id='kalman_intro'></a>
# ## Понятие фильтрации<sup>[toc](#toc)</sup>
# 
# Пусть у нас есть некоторая система, состояние которой в произвольный момент времени полностью описывается вектором $\State$. Однако это состояние от нас скрыто, и мы можем наблюдать лишь связанные с этим состоянием величины. Причем на практике наблюдаемые величины зашумлены и не позволяют сделать какой-то однозначный вывод о состоянии системы.
# 
# 
# Цель фильтрации состоит в том, чтобы восстановить скрытое состояние системы. Также можно сказать, что задача фильтрации состоит восстановлении незашумленного сиглана из шумного входа.
# 
# 
# Мы наблюдаем некоторые связанные с состоянием системы значения (измерения) $\Observ_{t_1}, \dots, \Observ_{t_n}, \dots$, однако сами состояния $\State_{t_1}, \dots, \State_{t_n}$ системы в моменты наблюдений мы не знаем &mdash; они является __скрытыми__,
# * Мы не уверены в измерениях, но можем говорить о вероятности ошибки измерения ($\Observ$)
# * Мы не измеряем интересные вещи напрямую, т.е. работаем с латентными переменными ($\State$)
# $$
# P(\State|\Observ) = \frac{P(\Observ|\State) P(\State)}{P(\Observ)}.
# $$
# 
# 
# Пусть мы знаем $\State_{k-1}$ &mdash; оценку латентных переменных на момент времени $k-1$ 
# 
# * 1) Мы пронаблюдали $\Observ_{k-1}$.
# * 2) Используем эту оценку для уточнения $\State_{k-1}$.
# * 3) Зная динамику системы, предсказываем $\State_k$, ждем нового измерения $\Observ_k$, идем в первый пункт, рекурсия.
# 
# Собственно из-за того, что (пере)оценка состояния системы происиходит на основе предшествующих оценок, байесовский фильтр называется **рекурсивным**.
# 
# Мы будем рассматривать так называемый фильтра Калмана
# 
# > Фи́льтр Ка́лмана &mdash; эффективный рекурсивный фильтр, оценивающий вектор состояния динамической системы, используя ряд неполных и зашумленных измерений. Назван в честь Рудольфа Калмана.
# 
# Фильтр Калмана называется **эффективным**, потому что при некоторых допущениях он дает оптимальную оценку вектора состояния.
# 
# В качестве бонуса фильтр достаточно легко считается: все что нам потребуется &mdash; обращать матрицы небольшой размерности.
# 
# **Источники:**
# * [Википедия](https://ru.wikipedia.org/wiki/Фильтр_Калмана)
# * [Probabilistic robotics](http://www.probabilistic-robotics.org/)
# 
# ![](img/recursive.png)

# <a id='kalman_filter_linear'></a>
# ## Линейный фильтр Калмана<sup>[toc](#toc)</sup>
# Рассмотрим вывод прогнозов на состояния в линейном дискретном фильтре Калмана, т.е. в следующих предположениях:
# * Время дискретно: $t \in \NN$. Будем писать $k$ вместо $t$.
# * $\State_{k}$ &mdash; вектор состояния системы в момент времени $k$.
# * $\Observ_{k}$ &mdash; наблюдение в момент времени $k$.
# * Модель эволюции:
# $$
# \State_{k} = \StateToState_{k} \State_{k - 1} + \StateNoise_k, \qquad \StateNoise_k \sim \Normal(\boldzero, \StateNoiseCov_k).
# $$
# * Модель наблюдений:
# $$
# \Observ_{k} = \StateToObserv_{k} \State_{k} + \ObservNoise_k, \qquad \StateNoise_k \sim \Normal(\boldzero, \ObservNoiseCov_k).
# $$
# 
# __Фильтр называется линейным__, потому что как модель эволюции, так и модель наблюдений являются линейными.
# 
# Несложно показать (что и будет проделано ниже), что при нормальном априорном распределении на состояние $\State_1$, все прогнозы на распределения вектора состояний будут нормальными, т.е.
# $\State_{k} \sim \Normal(\StateMean_{k}, \StateCov_{k})$.

# #### 1) Априорный прогноз<sup>[toc](#toc)</sup>
# Априорный прогноз на состояние $\State_{k}$ &mdash; это распределение состояния в момент времени $\State_{k}$ при условии, что известны наблюдения $\Observ_1, \dots, \Observ_{k-1}$.
# 
# Пусть нам известно распределение для $\State_{k-1}$ при условии всех наблюдений до момента $k - 1$ включительно. Пусть $p(\State_{k-1}|\Observ_{1}, \dots, \Observ_{k-1}) = \Normal(\StateMean_{k - 1}, \StateCov_{k - 1})$. Тогда
# $$
# p(\State_{k}|\Observ_1, \dots, \Observ_{k-1}) = \Normal(\State_{k};\hatStateMean_{k}, \hatStateCov_{k}),
# $$
# где
# $$
# \hatStateMean_{k} = \StateToState_{k}\StateMean_{k - 1}, \qquad \hatStateCov_{k} = \StateToState_{k}^T \StateCov_{k - 1} \StateToState_{k} + \StateNoiseCov_{k}.
# $$

# #### 2) Обработка измерений. Апостериорный прогноз<sup>[toc](#toc)</sup>
# Из модели наблюдений
# $$
# \Observ_{k} = \StateToObserv_{k} \State_{k} + \ObservNoise_k, \qquad \StateNoise_k \sim \Normal(\boldzero, \ObservNoiseCov_k)
# $$
# следует, что
# $$
# p(\Observ_{k} \vert \State_{k}) \sim N(\StateToObserv_{k} \State_{k}, \ObservNoiseCov_{k})
# $$
# 
# Из формулы Байеса следует, что
# $$
# p(\State_{k} | \Observ_{1}, \dots, \Observ_{k}) \propto p(\Observ_{k}|\State_{k}) p(\State_{k}|\Observ_{1},\dots,\Observ_{k-1}).
# $$
# Получаем произведение двух нормальных распределений, которое также является нормальным. Остается лишь найти его параметры. Вывод параметров приводится в разделе ниже. Здесь же запишем результаты:
# \begin{align*}
# &K_{k} = \hatStateCov_{k} \StateToObserv_{k}^T (\StateToObserv_{k} \hatStateCov_{k} \StateToObserv_{k}^T + \ObservNoiseCov_{k})^{-1},\\
# &\StateMean_{k} = \hatStateMean_{k} + K_{k} (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}),\\
# &\Sigma_{k} = (I - K_{k} \StateToObserv_{k}) \hatStateCov_{k}.
# \end{align*}
# 
# **Такое $K_k$ является оптимальным, если искать $\mu_k$ в виде линейной комбинации измерения и предсказания и пытаться минимизировать сумму квадратов отклонений $\State_{k}$ от $\mu_k$.**

# <a id='kalman_filter_der'></a>
# ## Вывод в фильтре Калмана<sup>[toc](#toc)</sup>
# В выражении для вероятности
# $$
# p(\State_{k} | \Observ_{1}, \dots, \Observ_{k}) \propto p(\Observ_{k}|\State_{k}) p(\State_{k}|\Observ_{1},\dots,\Observ_{k-1})
# $$
# имеем произведение двух экспонент. Показатели просуммируем и назовем $J_k$:
# $$
# p(\State_{k} \vert \Observ_{k}) = \eta \exp \lp J_k \rp,
# $$
# где
# $$
# J_k = \frac{1}{2} (\Observ_{k} - \StateToObserv_{k} \State_{k})^T \ObservNoiseCov_{k}^{-1}(\Observ_{k} - \StateToObserv_{k} \State_{k}) + \frac{1}{2} (\State_{k} - \hatStateMean_{k})^T \hatStateCov_{k}^{-1}(\State_{k} - \hatStateMean_{k}),
# $$
# т.е. получили форму по $\State_{k}$, а значит распределение нормальное. Минимум у нормального распределение расположен в мат. ожидании, а гессиан является обратным к матрице ковариации:
# $$
# \frac{\partial J_k}{\partial \State_{k}} = -\StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \State_{k}) + \hatStateCov_{k}^{-1}(\State_{k} - \hatStateMean_{k}).
# $$
# $$
# \frac{\partial^2 J_k}{\partial^2 \State_{k}} = \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} \StateToObserv_{k} + \hatStateCov_{k}^{-1}.
# $$
# С матрицей ковариаций $\StateCov_{k}$ разобрались $\StateCov_{k} = (\StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} \StateToObserv_{k} + \hatStateCov_{k}^{-1})^{-1}$. Среднее $\StateMean_{k}$ найдем, "обнулив" производную:
# $$
# \hatStateCov_{k}^{-1}(\StateMean_{k} - \hatStateMean_{k}) = \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \StateMean_{k}).
# $$
# Произведем некоторые преоразования в правой части:
# $$
# \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \StateMean_{k}) = \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \StateMean_{k} + \StateToObserv_{k} \hatStateMean_{k} - \StateToObserv_{k} \hat {\StateMean_{k}}) = \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}) - \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1}\StateToObserv_{k}(\StateMean_{k} - \hat {\StateMean_{k}})
# $$
# Подставим обратно и перенесем второе слагаемое налево:
# $$
# (\StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1}\StateToObserv_{k} + \hatStateCov_{k}^{-1})(\StateMean_{k} - \hatStateMean_{k}) = \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}).
# $$
# 
# $$
# \StateCov_{k}^{-1} (\StateMean_{k} - \hatStateMean_{k}) = \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}).
# $$
# 
# $$
# \StateMean_{k} = \StateCov_{k} \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1} (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}) + \hatStateMean_{k}.
# $$
# 
# Введем обозначение $$K_k = \StateCov_{k} \StateToObserv_{k}^T \ObservNoiseCov_{k}^{-1},$$
# при использовании которого для мат. ожидания апостериорного прогноза получим
# $$
# \StateMean_{k} = \hatStateMean_{k} + K_k (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}).
# $$
# 
# 
# Последнее выражение завершает вывод формул. Полученные формулы для $\StateCov_{k}$, $\StateMean_{k}$, $K_k$ можно эквивалентно переписать в следующем виде:
# 
# \begin{align*}
# &K_k = \hatStateCov_{k} \StateToObserv_{k}^T (\StateToObserv_{k} \hatStateCov_{k} \StateToObserv_{k}^T + \ObservNoiseCov_{k})^{-1},\\
# &\StateMean_{k} = \hatStateMean_{k} + K_k (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}),\\
# &\StateCov_{k} = (I - K_k \StateToObserv_{k}) \hatStateCov_{k}.
# \end{align*}
# 
# Такая форма удобнее с точки зрения времени вычисления, так как тут обращается матрица размерности наблюдений, что практически всегда меньше размерности состояния.

# <a id='kalman_filter_opt'></a>
# ### Оптимальность фильтра Калмана<sup>[toc](#toc)</sup>
# 
# Пусть до получения измерения $\Observ_{k}$ мы получили следующий априорный прогноз на вектор состояния $\State_{k}$:
# $$
# p(\State_{k}|\Observ_{1}, \dots, \Observ_{k-1}) = \Normal(\State_{k};\hatStateMean_{k}, \hatStateCov_{k}),
# $$
# т.е.
# $\State_{k} \sim N(\StateMean_{k}, \StateCov_{k})$.
# 
# Положим, что $\StateMean_{k} = \hatStateMean_{k} + K_k(\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k})$ - это соответствует нашей интуиции о том, что состояние должно быть где-то между измерением и предсказанием, т.е. линейно связано с разницей между ожиданием и действительностью.
# 
# Будем искать такую матрицу $K_k$, чтобы минимизировать дисперсию  $\State_{k}$, что эквивалентно минимизации следа матрицы $\StateCov_{k}$
# 
# \begin{gather*}
# \StateCov_{k} = cov(\State_{k} - \StateMean_{k}) = cov(\State_{k} - \hatStateMean_{k} - K_k(\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}))
# = cov(\State_{k} - \hatStateMean_{k} - K_k \Observ_{k} + K_k \StateToObserv_{k} \hatStateMean_{k})
# = cov(\State_{k} - \hatStateMean_{k} - K_k \StateToObserv_{k} \State_{k} - K_k \delta_k + K_k \StateToObserv_{k} \hatStateMean_{k})
# \end{gather*}
# 
# Шум измерения не зависит от $\State_{k}$ и обладает ковариацией $\ObservNoiseCov_{k}$
# \begin{gather*}
# \dots = cov(\State_{k} - \hatStateMean_{k} - K_k \StateToObserv_{k} \State_{k} + K_k \StateToObserv_{k} \hatStateMean_{k}) + K_k \ObservNoiseCov_{k} K_k^T
# = cov((\State_{k} - \hatStateMean_{k}) (I - K_k \StateToObserv_{k})) + K_k \ObservNoiseCov_{k} K_k^T
# = (I - K_k \StateToObserv_{k}) \hatStateCov_{k} (I - K_k \StateToObserv_{k})^T + K_k \ObservNoiseCov_{k} K_k^T = \\
# = \hatStateCov_{k} - K_k \StateToObserv_{k} \hatStateCov_{k} - \hatStateCov_{k} \StateToObserv_{k}^T K_k^T + K_k (\ObservNoiseCov_{k} + \StateToObserv_{k} \hatStateCov_{k} \StateToObserv_{k}^T) K_k^T = \dots
# \end{gather*}
# 
# $S_k = (\ObservNoiseCov_{k} + \StateToObserv_{k} \hatStateCov_{k} \StateToObserv_{k}^T)$ &mdash; ковариационная матрица шума измерения.
# 
# \begin{gather*}
# \dots = \hatStateCov_{k} - K_k \StateToObserv_{k} \hatStateCov_{k} - \hatStateCov_{k} \StateToObserv_{k}^T K_k^T + K_k S_k K_k^T = \hatStateCov_{k} + (K_k - \hatStateCov_{k} \StateToObserv_{k}^T S_k^{-1}) S_k (K_k - \hatStateCov_{k} \StateToObserv_{k}^T S_k^{-1})^T - \hatStateCov_{k} \StateToObserv_{k}^T S_k^{-1} \StateToObserv_{k} \hatStateCov_{k} = \dots
# \end{gather*}
# Тут используется тот факт, что раз $S_k$ ковариационная матрица, то $S_k = S_k^T$
# \begin{gather*}
# \dots = \ls \hatStateCov_{k} - \hatStateCov_{k} \StateToObserv_{k}^T S_k^{-1} \StateToObserv_{k} \hatStateCov_{k} \right] + \left[ (K_k - \hatStateCov_{k} \StateToObserv_{k}^T S_k^{-1}) S_k (K_k - \hatStateCov_{k} \StateToObserv_{k}^T S_k^{-1})^T \rs.
# \end{gather*}
# 
# Мы хотим минимизировать след этой матрицы по $K_k$, при этом только второе слагаемое зависит от $K_k$. Кроме того, видно, что это слагаемое является некоторой матрицей ковариации, а значит его след в лучшем случае может быть равен нулю, т.е. оптимальное $K_k$ &mdash; то, которое обнуляет это слагаемое:
# 
# $$
# \boxed{
# K_k = \hatStateCov_{k} \StateToObserv_{k}^T S_k^{-1} = \hatStateCov_{k} \StateToObserv_{k}^T (\StateToObserv_{k} \hatStateCov_{k} \StateToObserv_{k}^T + \ObservNoiseCov_{k})^{-1}
# }
# $$

# <a id='kalman_filter_results'></a>
# ## Формулы обновления прогноза в фильтре Калмана<sup>[toc](#toc)</sup>
# 
# Пусть $\State_{k - 1}$ &mdash; вектор состояния системы. Пусть сделан прогноз $\State_{k - 1} \sim \Normal(\mu_{k-1}, \Sigma_{k-1})$. В момент времени $k$ поступает измерение $\Observ_{k}$.
# 
# Динамика системы и модель измерения считаются линейными:
# \begin{gather*}
# \State_{k} = \StateToState_{k} \State_{k - 1} + \StateNoise_{k}, \qquad \StateNoise_{k} \sim \Normal(\boldzero, \StateNoiseCov_{k})\\
# \Observ_{k} = \StateToObserv_{k} \State_{k} + \ObservNoise_{k}, \qquad \ObservNoise_{k} \sim
# \Normal(\boldzero, \ObservNoiseCov_{k}).
# \end{gather*}
# 
# #### 1) Предсказание 
# $$
# \boxed{\hatStateMean_{k} = A_k \mu_{k-1}, \qquad \hatStateCov_{k} = A_k \Sigma_{k-1} A_k^T + R_k.}
# $$
# 
# #### 2) Обновление
# \begin{align*}
# &K_k = \hatStateCov_{k} \StateToObserv_{k}^T (\StateToObserv_{k} \hatStateCov_{k} \StateToObserv_{k}^T + \ObservNoiseCov_{k})^{-1},\\
# &\StateMean_{k} = \hatStateMean_{k} +K_k (\Observ_{k} - \StateToObserv_{k} \hatStateMean_{k}),\\
# &\StateCov_{k} = (I - K_k \StateToObserv_{k}) \hatStateCov_{k}.
# \end{align*}

# <a id='localization_model'></a>
# # Теоретическая модель калмановской локализации<sup>[toc](#toc)</sup>
# * [Задача локализации](#localization_task)
# * [Фильтр Калмана в локализации](#localization_kalman)
# * [1D модель движения материальной точки](#localization_kalman_1d_example)
# * [2D модель движения материальной точки](#2d_point_movement)
# * [Альтернативная модель движения материальной точки (для семинара и ДЗ)](#2d_with_yaw_model)
# 
# В данном разделе мы сначала поговорим про теоретическую модель движения автомобиля, который будем рассматривать как материальную точку (такое предположение является чрезвычайно грубым, однако всегда нужно начинать с простых моделей). Затем в данном ноутбуке будет представлена программная реализация модели для тестирования локализации на основе фильтра Калмана.

# <a id='localization_task'></a>
# ## Задача локализации<sup>[toc](#toc)</sup>
# В задаче локализации требуется определить **позу** автомобиля в пространстве. Под **положением** далее понимаются совокупность координат некоторой выбранной точки автомобиля, и **ориентации** автомобиля в пространстве.
# 
# * Положение твердого тела в трехмерном пространстве описывается 6-ю координатами: 3 координаты $(x, y, z)$ &mdash; положение некоторой точки твердого тела + 3 угла поворота $(\psi, \theta, \phi)$, задающие ориентацию твердого тела (о них далее). В случае оценки положения в трехмерном мире говорят о **3D-локализации**.
# * Положение твердого тела в двухмерном пространстве описывается тремя координатами: 2 координаты $(x, y)$ и угол $\psi$. В случае оценки положения в двухмерном мире говорят о **2D-локализации**.
# 
# Существует множество подходов к заданию ориентации твердого тел. Одним из наиболее известных является **система углов Эйлера $\psi$, $\theta$, $\phi$** (левый рисунок ниже). В этой системе ориентация твердого тела задается тремя вращениями в связанной с твердым телом системе координат.

# ![](img/eiler.png)

# Систему углов Эйлера можно образно описать тройкой $ZXZ$: сначала поворот вокруг оси $Z$, потом $X$, а затем снова $Z$. Если говорить более детально, то последовательность применения поворотов выглядит следующим образом:
# * Поворот вокруг оси $Z$, т.е. в плоскости $XY$, на угол $\psi$. В результате получаем систему координат $X'Y'Z$
# * Поворот вокруг оси $X'$, т.е. в плоскости $Y'Z$, на угол $\theta$. В результате получаем связанную систему координат $X'Y''Z'$.
# * Поворот вокруг оси $Z'$, т.е. в плоскости $X'Y''$, на угол $\phi$. В результае получаем связанную систему координат $X''Y'''Z'$.
# 
# У углов Эйлера есть специальные названия:
# * $\psi$ &mdash; **угол прецессии**;
# * $\theta$ &mdash; **угол нутации**;
# * $\phi$ &mdash; **угол собственного вращения**.
# 
# В случае 2D-локализации существует лишь угол прецессии $\psi$.
# 
# 
# Существуют и другие подходы к заданию углов. Один из наиболее известных &mdash; углы Крылова-Булгакова ($\alpha$, $\beta$, $\gamma$), которые задают вращения в связанной системе координат последовательно вокруг осей $XYZ$ (правый рисунок выше). У этих углов также есть специальные названия:
# * $\alpha$ &mdash; **крен (rool)**;
# * $\beta$ &mdash; **тангаж (pitch)**;
# * $\gamma$ &mdash; **рыскание (yaw)**.
# 
# В случае 2D-локализации существует лишь угол рыскания $\gamma$.

# <a id='localization_kalman'></a>
# ## Фильтр Калмана в локализации<sup>[toc](#toc)</sup>
# 
# В первую очередь мы ожидаем, что нам известны законы (хорошо когда *линейные*), по которым развивается динамическая система, т.е. рассматриваем **линейную динамическую систему (ЛДС)**. Текущие характеристики системы формируют вектор состояния.
# 
# ![](img/dynamic.png)
# 
# 
# Для составления фильтра Калмана в локализации требуется:
# * Определится с содержимым вектора состояния системы.
# * Исходя из динамики системы записать выражение для состояния $\State_{k}$ системы в момент времени $k$ через состояние $\State_{k-1}$:
# $$
# \State_{k} = \StateToState_{k} \cdot \State_{k - 1} + \StateNoise_{k}, \qquad \StateNoise_{k} \sim \Normal(\boldzero, \StateNoiseCov_{k}).
# $$
# где $A_{k}$ &mdash; известные матрицы, задающие динамику системы, $\StateNoise_{k}$ &mdash; случайный шум (например, моторы робота не в точности выполняют команды или вектор управления измеряется не точно).
# * Исходя из природы сенсоров, порождающих наблюдения, построить модели наблюдений:
# $$
# \Observ_{k} = \StateToObserv_{k} \State_{k} + \ObservNoise_{k}, \qquad \ObservNoise_{k} \sim \Normal(\boldzero, \ObservNoiseCov_{k}).
# $$
# 
# Рассмотрим все три этапа на примере 1D-локализации.

# Пусть $\State(t) \in \RR^{\StateDim}$ &mdash; состояние автомобиля в момент времени $t$. В большинстве случаев движение автомобиля по некоторой траектории описывается следующим дифференциальным уравнением:
# $$
# \State(t) = \boldf(\State(t), t),
# $$
# где $\boldf(\cdot)$ &mdash; это вектор функция, зависящая в общем случае от текущего момента времени $t$. Тогда мы можем перейти к дискретным моментам времени, начав рассматривать движение автомобиля с шагом $\Delta t$:
# $$
# \State(t + \Delta t) = \State(t) + \boldf(\State(t), t)\Delta t + O(\Delta t^2) \approx \State(t) + \boldf(\State(t), t)\Delta t.
# $$
# Если функция $\boldf(\cdot)$ линейна по $\State$ (пример ниже), то сразу можем построить матрицу перехода $\StateToState(t,t+\Delta t)$. Если же нет, то сначала ее можно разложить по Тейлору с целью линеаризации.
# 
# Таким образом, мы можем численно проинтегрировать уравнение движения и найти траектории. Но у нас еще есть шум $\StateNoise(t) \sim \Normal(\boldzero, ?)$. Знак вопроса поставлен не случайно: какова матрица ковариации шума при переходе вперед на момент времени $t + \Delta t$? Оказывается, что в этом случае говорят не о матрице ковариации, а о плотности матрица ковариации. При этом
# $$
# \State(t + \Delta t) = \State(t) + \boldf(\State(t), t)\Delta t + \StateNoise(t,t+\Delta t),
# $$
# где  $\StateNoise(t,t+\Delta t)$ &mdash; это шум, при прогнозе от момента времени $t$ на момент $t + \Delta t$:
# $$
# \StateNoise(t,t+\Delta t) \sim \Normal(\boldzero, \StateNoiseCov(t) \Delta t), \qquad \StateNoiseCov(t) \in \RR^{\StateDim \times \StateDim}
# $$
# т.е. чем на больше шаг $\Delta t$, на который мы хотим сделать прогноз, тем больше уровень шума. И это вполне ожидаемо: неопределенность нарастает линейно со временем: чем больше ждем, тем больше система "расплывается".

# <a id='localization_kalman_1d_example'></a>
# ## 1D модель движения материальной точки<sup>[toc](#toc)</sup>
# 
# Рассмотрим задачу оценки пололжения в одномерном мире. В таком случае положение полность описывается одной координатой $x$. Вектор состояния, однако, может быть определен множеством способов:
# * $\boldz = (x) \in \RR^1$
# * $\boldz = (x, \dot{x}) \equiv (x, v) \in \RR^2$
# * $\boldz = (x, \dot{x}, \ddot{x}) \equiv (x, v, a) \in \RR^3$
# 
# Вообще говоря, какое из этих представлений следует использовать, определяется наблюдениями, которые нам доступны.
# * Если нам доступны __только наблюдения для положения__ $x$, то хранить скорость в состоянии бессмысленно &mdash; у нас принципиально нет входных данных, чтобы ее обновлять/корректировать. Можем только дифференцировать полученные оценки на координату $x$.
# * Пусть теперь есть данные с одометрии, т.е. есть источник информации о скорости. Тогда имеет смысл добавить в состояние еще одну переменную. Но нет смысла добавлять ускорение, так как нет источников для его обновления/коррекции.
# * Пусть дополнительно доступен акселерометр, т.е. источник информации об ускорении, поэтому можем добавить ускорение в качестве еще одной переменной состояния.
# 
# Можно сформировать следующее правило: **если какая-то переменная состояния не участвует в формировании наблюдений, то хранить ее бессмысленно**.
# 
# $$
# \State_{t+\Delta t} = A_t \State_{t} + \StateNoise_{t}, \qquad \StateNoise_t \sim \Normal(\boldzero, \StateNoiseCov_t).
# $$
# 
# $$
# \begin{pmatrix}
# x(t + \Delta t)\\
# v(t + \Delta t)\\
# a(t + \Delta t)\\
# \end{pmatrix} \approx
# \begin{pmatrix}
# x(t) + v(t) \Delta t\\
# v(t) + a(t) \Delta t\\
# a(t)\\
# \end{pmatrix}
# $$
# Здесь использовано разложение в ряд Тейлора для $x(t + \Delta t)$ и $v(t + \Delta t)$ до первого порядка, так как хотим работать с линейной системой. Для матрицы перехода получаем
# $$
# \StateToState(\Delta t) =
# \begin{pmatrix}
# 1 &\Delta t &0\\
# 0 &1 &\Delta t\\
# 0 &0        &1\\
# \end{pmatrix}
# $$
# Пусть у нас есть три типа наблюдений: показания GPS ($x$), одометрии ($v$) и акселерометра ($a$):
# $$
# \Observ_t = \StateToObserv_t \State_t + \ObservNoise_t, \quad \ObservNoise_t \sim \Normal(\boldzero, \ObservNoiseCov_t).
# $$
# Матрицы наблюдений имеют вид:
# $$
# \StateToObserv_{x}  =\begin{pmatrix}
# 1 &0 &0
# \end{pmatrix}, \qquad
# \StateToObserv_{v}  =\begin{pmatrix}
# 0 &1 &0
# \end{pmatrix}, \qquad
# \StateToObserv_{a}  =\begin{pmatrix}
# 0  &0 &1
# \end{pmatrix}.
# $$
# 
# Осталось лишь задать значения шумов $\StateNoiseCov_t$, $\ObservNoiseCov_{x, t}$, 
# $\ObservNoiseCov_{v, t}$, $\ObservNoiseCov_{a, t}$ и можно применять калмановскую фильтрацию для оценки положения автомобиля.

# <a id='2d_point_movement'></a>
# ## 2D модель движения материальной точки<sup>[toc](#toc)</sup>
# Пусть считать, что у нас есть показания GPS (получаем оценки на положение в двухмерном мире) и показания одометрии (получаем значения скоростей). В качестве скрытого вектора состояния системы возьмем
# $$
# \State_t = \begin{pmatrix}
# x\\
# y\\
# v_x\\
# v_y\\
# \end{pmatrix},
# $$
# где 
# * $x$ &mdash; координата точки по оси $X$;
# * $y$ &mdash; координата точки по оси $Y$;
# * $v_x$ &mdash; проекция скорости на ось $X$;
# * $v_y$ &mdash; проекция скорости на ось $Y$.
# 
# $$
# \State(t + \Delta t) = \boldf(\State(t)) \Rightarrow 
# \begin{pmatrix}
# x(t + \Delta t)\\
# y(t + \Delta t)\\
# v_x(t + \Delta t)\\
# v_y(t + \Delta t)\\
# \end{pmatrix} = 
# \begin{pmatrix}
# x(t) + v_x(t)\Delta t\\
# y(t) + v_y(t)\Delta t\\
# v_x(t)\\
# v_y(t)\\
# \end{pmatrix}.
# $$
# 
# Тогда
# * Матрица перехода от момента $t$ к моменту $t + \Delta t$ имеет вид
# $$
# \StateToState(t, t + \Delta t) \equiv \StateToState(\Delta t) = \begin{pmatrix}
# 1 &0 &\Delta t &0\\
# 0 &1 &0        &\Delta t\\
# 0 &0 &1        &0\\
# 0 &0 &0        &1\\
# \end{pmatrix}
# $$
# * Матрица для наблюдений GPS:
# $$
# \StateToObserv_{(x,y)} = \begin{pmatrix}
# 1 &0 &0 &0\\
# 0 &1 &0 &0
# \end{pmatrix}.
# $$
# * Матрица для наблюдений одометрии (CAN):
# $$
# \StateToObserv_{(v_x,v_y)} = \begin{pmatrix}
# 0 &0 &1 &0\\
# 0 &0 &0 &1
# \end{pmatrix}.
# $$

# <a id='2d_with_yaw_model'></a>
# ## Альтернативная 2D модель движения материальной точки (для семинара и ДЗ)<sup>[toc](#toc)</sup>

# ### Состояние системы
# Теперь рассмотрим чуть более близкую к реальности модель движения материальной точки. В данной модели состояние движущейся описывается следующим вектором:
# $$
# \State = \begin{pmatrix}
# x\\
# y\\
# \gamma\\
# v\\
# \omega\\
# \end{pmatrix},
# $$
# где 
# * $x$ &mdash; координата точки по оси $X$;
# * $y$ &mdash; координата точки по оси $Y$;
# * $\gamma$ &mdash; угол рыскания; далее просто yaw;
# * $v$ &mdash; проекция скорости на ось $X$;
# * $\omega$ &mdash; угловая скорость материальной точки.

# ### Модель эволюции системы
# #### Переход из момента времени $t$ к моменту $t + \Delta t$<sup>[toc](#toc)</sup>
# $$
# \State(t + \Delta t) = \boldf(\State(t)) \Rightarrow 
# \begin{pmatrix}
# x(t + \Delta t)\\
# y(t + \Delta t)\\
# \gamma(t + \Delta t)\\
# v(t + \Delta t)\\
# \omega(t + \Delta t)
# \end{pmatrix}
# \approx
# \begin{pmatrix}
# x(t) + v(t)\cos(\gamma(t))\Delta t\\
# y(t) + v(t)\sin(\gamma(t))\Delta t\\
# \gamma(t) + \omega(t) \Delta t\\
# v(t)\\
# \omega(t)\\
# \end{pmatrix}.
# $$
# Заметим, что зависимость между состояниям нелинейная. Т.е. уже нельзя представить переход из момента $t$ в момент $t + \Delta t$ в виде:
# $$
# \State(t + \Delta t) \approx A(\Delta t) \State(t)
# $$
# Поэтому в данном случае мы будем использовать **EKF-фильтр**.
# 
# #### Якобиан функции перехода $f(x(t))$<sup>[toc](#toc)</sup>
# Итак, мы знаем, что $\State(t) \sim \Normal(\mu(t), \Sigma(t))$. Также мы знаем, что состояние в момент $\State(t + \Delta t)$ выражается через состояние в момент $\State(t)$ через некоторую нелинейную функцию $f(\cdot)$:
# $$
# \State(t + \Delta t) \approx f(\State(t), \Delta t) = \begin{pmatrix}
# x(t) + v(t)\cos(\gamma(t))\Delta t\\
# y(t) + v(t)\sin(\gamma(t))\Delta t\\
# \gamma(t) + \omega(t) \Delta t\\
# v(t)\\
# \omega(t)\\
# \end{pmatrix}
# $$
# Поэтому
# $$
# \State(t + \Delta t) \approx f(\mu(t)) + J_f(\mu(t)) (\State(t) - \mu(t)) + \text{H.O.D.},
# $$
# где 
# * $f(\mu(t), \Delta t)$ &mdash; состояние в момент времени $t + \Delta t$, при условии, что в момент $t$ мы детерминированно находились в состоянии $\State(t) = \mu(t)$.
# * $J_f(\mu(t))$ &mdash; значение Якобиана $f(\State)$ по $\State$ в точке $\mu(t)$. 
# * $\State(t) - \mu(t)$ &mdash; отклонение истинного значения состояния $\State(t)$ от ожидаемого $\mu(t)$; имеет распределение $\Normal(\boldzero, \Sigma(t))$, чем и обуславливает нормальное распределение для $\State(t + \Delta t)$.
# * H.O.D. &mdash; компоненты разложения высшего порядка, которыми пренебрегаем (Higher Order Degress).
# 
# Якобиан имеет вид:
# $$
# J_f(\State(t)) =
# \begin{pmatrix}
# 1 &0 &-v(t)\sin(\gamma(t))\Delta t &\cos(\gamma(t))\Delta t &0\\
# 0 &1 &v(t)\cos(\gamma(t))\Delta t &\sin(\gamma(t))\Delta t &0\\
# 0 &0 &1 &0 &\Delta t\\
# 0 &0 &0 &1 &0\\
# 0 &0 &0 &0 &1\\
# \end{pmatrix}
# $$
# 
# #### Модель эволюции системы<sup>[toc](#toc)</sup>
# В результате получаем
# \begin{align}
# &\mu(t + \Delta t) = f(\mu(t), \Delta t),\\
# &\Sigma(t + \Delta t) = J_f(\mu(t)) \Sigma(t)J_f(\mu(t))^T + R(t) \Delta t,
# \end{align}
# где 
# * $\mu(t)$ &mdash; текущее среднее состояния;
# * $\Sigma(t)$ &mdash; текущая дисперсия состояния;
# * $\Delta t$ &mdash; шаг по времени;
# * $R(t)$ &mdash; матрица плотности шума (показывает, как быстро дисперсия состояния наростает со временем).

# ### Модель наблюдений системы
# По сути достаточно лишь указать матрицы наблюдений. Так как в них нет нелинейностей, то тут можно применять обычную обработку показаний (без появления Якобианов и т.п.).
# #### GPS
# $$
# C_{gps} =
# \begin{pmatrix}
# 1 &0 &0 &0 &0\\
# 0 &1 &0 &0 &0\\
# \end{pmatrix}
# $$
# 
# #### CAN
# $$
# C_{can} =
# \begin{pmatrix}
# 0 &0 &0 &1 &0\\
# \end{pmatrix}
# $$
# 
# #### IMU
# $$
# C_{imu} =
# \begin{pmatrix}
# 0 &0 &0 &0 &1\\
# \end{pmatrix}
# $$

# <a id='seminar'></a>
# # Семинар<sup>[toc](#toc)</sup>
# * [Программная модель движения автомобиля](#car_model_architecture)
# * [Программная модель калмановской локализации](#kalman_car_model_architecture)
# * [Визуализация](#visualization)
# * [Тестирование калмановской фильтрации](#test_kalman_localization)

# <a id='car_model_architecture'></a>
# ## Программная модель движения автомобиля<sup>[toc](#toc)</sup>
# * [Время в модели (класс `Timestamp`)](#program_model_time)
# * [Создание автомобиля с некоторым начальным состоянием](#program_model_car)
# * [Задание истинной траектории движения автомобиля](#program_car_movement_model)
# * [Создание и установка необходимых датчиков](#car_program_model_sensors)
# * [Проверка движения модели автомобиля и показаний датчиков](#program_car_model_check_movement)
# * [Создание автомобиля, настрока движения и установка необходимых датчиков](#program_car_model_create)
# 
# В данной работе уже написана значительная часть кода, позволяющая создать модель автомобиля (класс `Car`), добавить к нему сенсоры (классы `GpsSensor`, `CanSensor`, `ImuSensor`) и указать автомобилю некоторую траекторию движения (наследники класса `MovementModelBase`).

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from IPython import display
import time
from matplotlib.lines import Line2D
    
get_ipython().run_line_magic('matplotlib', 'inline')

# Специальный класс для удобного представления времени в модели
from sdc.timestamp import Timestamp


# Класс для представления машины
from sdc.car import Car
# Модель прямолинейного движения машины
from sdc.linear_movement_model import LinearMovementModel
# Моедль движения машины по циклоиде
from sdc.cycloid_movement_model import CycloidMovementModel
# Сенсор одометрии
from sdc.can_sensor import CanSensor
# GPS-датчик
from sdc.gps_sensor import GpsSensor
# IMU-датчик
from sdc.imu_sensor import ImuSensor


# <a id='program_model_time'></a>
# ### Время в модели (класс `Timestamp`)<sup>[toc](#toc)</sup>
# В первую очередь нам понадобится способ представления времени в модели. Предполагается использовать для этих целей класс `Timestamp`, который хранит время с точностью до наносекунд. Для удобной работы этот класс должен поддерживает:
# * арифметрические операции
# * преобразования к различным масштабам времени: секундам, миллисекундам и т.д.
# * создание из статических методов из секунд, миллисекунд и т.д.
# 
# Ниже приведены примеры его использования.

# In[2]:


t1 = Timestamp(sec=1, nsec=12333)
t2 = Timestamp(sec=5, nsec=33149)
print('t1 = {}'.format(t1))
print('t2 = {}'.format(t2))

# Создание
print('t1 + t2 = {}'.format(t1 + t2))
print('t2 - t1 = {}'.format(t2 - t1))

# Преобразования
print('t1.to_seconds() = {}s'.format(t1.to_seconds()))
print('t1.to_milliseconds() = {}ms'.format(t1.to_milliseconds()))
print('t1.to_microseconds() = {}us'.format(t1.to_microseconds()))
print('t1.to_nanoseconds() = {}ns'.format(t1.to_nanoseconds()))

# Статические методы-фабрики
t = Timestamp.seconds(5)
t = Timestamp.milliseconds(5)
t = Timestamp.microseconds(5)
t = Timestamp.nanoseconds(5)


# <a id='program_model_car'></a>
# ### Создание автомобиля с некоторым начальным состоянием<sup>[toc](#toc)</sup>

# In[3]:


# По умолчанию должен создавать автомобиль в нулевой точке с нулевой скоростью в нулевой момент времени
initial_position = [1., 2.]
initial_velocity = 5.
initial_yaw = 0.
initial_omega = 0.
car = Car(initial_position=initial_position, initial_velocity=initial_velocity, initial_yaw=initial_yaw)

# При печати должно выводиться текущее истинное состояние автомобиля и время
print(car)

print('\nAccessing real car state via properties:')
print('car.x     = {}[m]'.format(car._position_x))
print('car.y     = {}[m]'.format(car._position_y))
print('car.yaw   = {}[rad]'.format(car._yaw))
print('car.v     = {}[m/s]'.format(car._velocity))
print('car.omega = {}[rad/s]'.format(car._omega))


# <a id='program_car_movement_model'></a>
# ### Задание истинной траектории движения автомобиля<sup>[toc](#toc)</sup>
# Автомобиль движется по некоторой истинной траектории. В рамках задачи локализации мы эту траекторию не знаем и пытаемся оценить ее по данным с различных сенсоров. Однако при проведении модельных экспериментов мы можем задать нашей машине `Car` некоторое вполне определенное движение: прямолинейное, циклоидное, синусоидальное и т.п.
# Для этих целей имеется класс `MovementModelBase`, предлагающий интерфейся для реализации истинной траектории движения машины. У этого класса есть метод `_move` (используем нижнее подчеркивание для обозначения семантической приватности), которым можно "сдвинуть" машину вдоль траектории на время `dt`:
# ```
# movement_model._move(car, dt)
# ```
# Напрямую с моделиями движения вам работать не придется. Достаточно лишь знать, что при вызове метода `move` у `Car`:
# ```
# car.move(dt)
# ```
# внутри происходит доступ к модели движения и вызов соответствующего метода `_move`.
# 
# Для каждой траектории движения у `MovementModelBase` должен быть свой специфичный наследник:
# * `LinearMovementModel` &mdash; движение с постоянной начальной скоростью
# * `CycloidMovementModel` &mdash; движение по циклоиде

# **Добавим к нашей машине модель прямолинейного движения:**

# In[4]:


movement_model = LinearMovementModel()
car.set_movement_model(movement_model)


# **Теперь можно наблюдать затем, как с помощью вызова метода `Car.move` машина продвигается вперед на запрошенное время:**

# In[5]:


# Теперь автомобиль можно будет двигать методом move, который переадресует вызов модели движения
print(car)
dt = Timestamp.milliseconds(100)
car.move(dt)
print(car)


# <a id='car_program_model_sensors'></a>
# ### Создание и установка необходимых датчиков<sup>[toc](#toc)</sup>
# Теперь добавим к нашей машине следующие датчики:
# * Датчик GPS &mdash; в нашей модели возвращает значения $(x, y)$;
# * Датчик одометрии &mdash; в нашей модели возвращает абсолютное значение скорости $v$.

# In[6]:


# Стандартное отклонение 5 метров по каждой из координат X, Y
gps_noise = [5, 5]
gps_sensor = GpsSensor(gps_noise, random_state=0)

# Стандартное отклонение 0.5 м/с для наблюдения скорости
can_noise = [0.25]
can_sensor = CanSensor(can_noise, random_state=1)

# Установка датичков на автомобиль
car.add_sensor(gps_sensor)
car.add_sensor(can_sensor)

# Можно получить доступ к списку установленных датчиков
print(car.sensors)


# **Запросим значения наблюдений с сенсоров для текущего момента времени:**

# In[7]:


print(car)

# Запрос наблюдений от датчиков (только после присоединения к автомобилю)
gps_observation = gps_sensor.observe()
print('GPS observation = {}'.format(gps_observation))

can_observation = can_sensor.observe()
print('CAN observation = {}'.format(can_observation))


# <a id='program_car_model_check_movement'></a>
# ### Проверка движения модели автомобиля и показаний датчиков<sup>[toc](#toc)</sup>

# In[8]:


# Создание машины
initial_position = [2, 3]
initial_velocity = 10
initial_yaw = 0
initial_omega = 0.1
car = Car(initial_position=initial_position, initial_yaw=initial_yaw,
          initial_velocity=initial_velocity, initial_omega=initial_omega)
print(car)

assert car._position_x == 2
assert car._position_y == 3
assert car._yaw == initial_yaw
assert car._velocity == initial_velocity
assert car._omega == initial_omega
assert car._time == Timestamp.seconds(0)

# Тестирование последовательного движения
movement_model = LinearMovementModel()
car.set_movement_model(movement_model)

# Установка сенсоров
gps_sensor = GpsSensor(noise_variances=[1, 1], random_state=1)
can_sensor = CanSensor(noise_variances=[0.25], random_state=2)
car.add_sensor(gps_sensor)
car.add_sensor(can_sensor)

# Шаг интегрирования
dt = Timestamp.seconds(1)
print('dt = {}'.format(str(dt)))
# Общее время интегрирования
duration = Timestamp.seconds(10)
final_time = car.time + duration
print('')
while car.time < final_time:
    car.move(dt)
    # Смотрим на состояние в момент t + dt
    gps_observation = car.gps_sensor.observe()
    can_observation = car.can_sensor.observe()
    print(car)
    print('GPS observation = {}'.format(gps_observation))
    print('CAN observation = {}'.format(can_observation))
    print('')

assert car._time == Timestamp.seconds(10)


# <a id='program_car_model_create'></a>
# ### Создание автомобиля, настрока движения и установка необходимых датчиков<sup>[toc](#toc)</sup>
# Функция для удобного создания полностью настроенной модели машины:

# In[9]:


def create_car(
        initial_position=[5, 5],
        initial_velocity=5,
        initial_omega=0.0,
        initial_yaw=np.pi / 4,
        can_noise_variances=[0.25],   # Стандартное отклонение - 0.5м
        gps_noise_variances=[1, 1],   # Стандартное отклонение - 1м
        imu_noise_variances=None,     # По умолчанию IMU не используется
        random_state=0,
):
    # Начальное состояние автомобиля
    car = Car(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_yaw=initial_yaw,
        initial_omega=initial_omega)
    # Создание сенсоров
    if can_noise_variances is not None:
        car.add_sensor(CanSensor(noise_variances=can_noise_variances, random_state=random_state))
        random_state += 1
    if gps_noise_variances is not None:
        car.add_sensor(GpsSensor(noise_variances=gps_noise_variances, random_state=random_state))
        random_state += 1
    if imu_noise_variances is not None:
        car.add_sensor(ImuSensor(noise_variances=imu_noise_variances, random_state=random_state))
        random_state += 1
    # Последовательное движение
    movement_model = LinearMovementModel()
    car.set_movement_model(movement_model)
    return car


# <a id='kalman_car_model_architecture'></a>
# ## Программная модель калмановской локализации<sup>[toc](#toc)</sup>
# * [Основные два метода калмановской фильтрации](#kalman_filter)
# * [Создание калмановской модели автомобиля с некоторым начальным состоянием](#program_model_kalman_car)
# * [Создание калмановских моделей сенсоров](#program_model_kalman_sensors)
# * [Создание калмановской модели машины из обычной модели](#program_model_kalman_car_from_car)
# 
# Многие классы в калмановской фильтрации, которые будут рассмотрены ниже, могут показаться похожими и чуть ли не дублирующимися с уже определенными классами для моделирования движения машины. В частности, имеют место следующие аналогии
# * `Car` ->  `KalmanCar`
# * `CanSensor` -> `KalmanCanSensor`
# * `GpsSensor` -> `KalmanGpsSensor`
# * `ImuSensor` -> `KalmanImuSensor`
# * `LinearMovementModel` -> `KalmanMovementModel`
# 
# Тут следует понимать природу такого дублирования и разницу между сущностми, **предназначенными для моделирования показания движения машины и показаний датчиков**, и сущностями, предназначенными для **локалиации машины с помощью калмановской фильтрации на основе уже смоделированных ранее (или полученных из реальности) показаний датчиков**.
# 
# Можно сказать, что все введенные выше классы (`Car`, `CanSensor` и т.д.) в совокупности представляют собой некий симулятор движения машины и служат лишь одной цели &mdash; смоделировать движение машины и показания датчиков.
# И вот теперь мы умеем моделировать показания, и нам нужно понять, сможем ли мы за счет ТОЛЬКО знания этих показаний восстановить траекторию движения автомобилия, пользуясь для этого калмановской фильтрацией. Для начала нужно построить **калмановскую модель движения машины и показаний датчиков**. Используемые для этого сущности с префиксом `Kalman` как раз служат этой цели.
# 
# **Внимание.** И в моделировании движения, и в локализации мы используем один и тот же набор переменных состояния. Т.е. мы как будто-то бы угадали набор переменных, который полностью описывает состояние машины: ($x$, $y$, $\gamma$, $v$, $\omega$). Но стоит понимать, что для моделирования реального движения мы могли бы написать гораздо более физическу точную модель `Car` движения машины, в которой может быть тысяча переменных, описывающих внутреннее состояние машины (скорости колес, углы их поворота, состояние подвески и т.п.). Затем на основе этой модели мы бы получили показания сенсоров, которые использовали бы для калмановской локализации с более "бедным" состояним из 5-и переменных.

# In[10]:


# Калмановская локализация
from sdc.kalman_car import KalmanCar
from sdc.kalman_can_sensor import KalmanCanSensor
from sdc.kalman_gps_sensor import KalmanGpsSensor
from sdc.kalman_imu_sensor import KalmanImuSensor
from sdc.kalman_movement_model import KalmanMovementModel
from sdc.kalman_filter import (
    kalman_transit_covariance,
    kalman_process_observation,
)


# <a id='kalman_filter'></a>
# ### Основные два метода калмановской фильтрации<sup>[toc](#toc)</sup>
# * `kalman_transit_covariance(S, A, R)` &mdash; принимает на вход:
#     * `S` &mdash; текущая матрица ковариации `S`
#     * матрицу перехода `A` или матрицу Якоби `J` (в случае расширенной калмановской фильтрации).
#     * `R` &mdash; матрица ковариации шума. Возвращает матрицу ковариации на следующем шаге.
# * `kalman_process_observation(mu, S, observation, C, Q)` &mdash; принимает на вход
#    * `mu` &mdash; текущая оценка среднего
#    * `S` &mdash; текущая матрица ковариации
#    * `observation` &mdash; вектор-наблюдение
#    * `C` &mdash; матрица наблюдения
#    * `Q` &mdash; матрица ковариации шума в наблюдении
#    
# **Эти два метода вам предстоит написать самим и добавить их в файл `sdc.kalman_filter`. Их написание является первым пунктом домашнего задания. Если методы написаны правильно, модель `KalmanCar` (см. ниже) будет работать корректно.**

# ```
# def kalman_transit_covariance(S, A, R):
#     assert False, 'Not implemented'
# 
# def kalman_process_observation(mu, S, observation, C, Q):
#     assert False, 'Not implemented'
# ```

# Калмановская модель автомобиля `KalmanCar` &mdash; набор переменных состояния автомобиля, сенсоров автомобиля и модели движения:
# * Описание положения машины в фильтре Калмана:
#     * Переменные состояния (`kalman_car.state`) представляют собой текущее значение среднего $\mu(t)$
#     * Матрица ковариаций состония (`kalman_car.covariance_matrix`) представляет собой текущее значение матрицы $\Sigma(t)$
# * Каждая калмановская модель сенсора (наследник класса `KalmanSensorBase`):
#     * знают свою матрицу наблюдения $C$
#     * знают свою матрицу шума $Q$
#     * умеют обрабатывать налюдения `kalman_sensor.process_observation` (метод рабоате, если сенсор установлен на `kalman_car` и ему есть откуда взять текущее состояние; см. примеры далее)
# * Калмановская модель движенеия (`KalmanMovementModel`) описывает модель эволюции.

# <a id='program_model_kalman_car'></a>
# ### Создание калмановской модели автомобиля с некоторым начальным состоянием<sup>[toc](#toc)</sup>

# In[11]:


kalman_car = KalmanCar(
    initial_position=car.initial_position,   # Начальное состояние берем из модели машины
    initial_yaw=car.initial_yaw,
    initial_velocity=car.initial_velocity,
    initial_omega=car.initial_omega)
print(kalman_car)
print('\nInitial kalman state:\n{}\n'.format(kalman_car.state))
print('Initial kalman covariance matrix:\n{}'.format(kalman_car.covariance_matrix))


# <a id='program_model_kalman_sensors'></a>
# ### Создание калмановских моделей сенсоров<sup>[toc](#toc)</sup>

# In[12]:


expected_gps_noise = car.gps_sensor._noise_variances
kalman_gps_sensor = KalmanGpsSensor(noise_variances=expected_gps_noise)

expected_can_noise = car.can_sensor._noise_variances
kalman_can_sensor = KalmanCanSensor(noise_variances=expected_can_noise)

kalman_car.add_sensor(kalman_gps_sensor)
kalman_car.add_sensor(kalman_can_sensor)

print('GPS kalman sensor data:')
print('Q_gps =\n{}'.format(kalman_car.gps_sensor.get_noise_covariance()))
print('C_gps =\n{}\n'.format(kalman_car.gps_sensor.get_observation_matrix()))

print('CAN kalman sensor data:')
print('Q_can =\n{}'.format(kalman_car.can_sensor.get_noise_covariance()))
print('C_can =\n{}\n'.format(kalman_car.can_sensor.get_observation_matrix()))


# <a id='program_model_kalman_car_from_car'></a>
# ### Создание калмановской модели машины из обычной модели<sup>[toc](#toc)</sup>

# In[13]:


def create_kalman_car(car, gps_variances=None, can_variances=None, imu_variances=None):
    """Создает калмановскую модель движения автомобиля на основе уже настроенной модели самого автомобиля"""
    # Скорость нарастания дисперсии в секунду
    noise_covariance_density = np.diag([
        0.1,
        0.1,
        0.1,   # Дисперсия yaw
        0.1,   # Дисперсия скорости
        0.1    # Дисперсия угловой скорости
    ])
    # Формирование состояние калмановской локализации
    kalman_car = KalmanCar(
        initial_position=car.initial_position,
        initial_velocity=car.initial_velocity,
        initial_yaw=car.initial_yaw,
        initial_omega=car.initial_omega)
    # Начальная матрица ковариации
    kalman_car.covariance_matrix = noise_covariance_density

    # Модель движения
    kalman_movement_model = KalmanMovementModel(noise_covariance_density=noise_covariance_density)
    kalman_car.set_movement_model(kalman_movement_model)

    for sensor in car.sensors:
        noise_variances = sensor._noise_variances
        if isinstance(sensor, GpsSensor):
            noise_variances = noise_variances if gps_variances is None else gps_variances
            kalman_sensor = KalmanGpsSensor(noise_variances=noise_variances)
        elif isinstance(sensor, CanSensor):
            noise_variances = noise_variances if can_variances is None else can_variances
            kalman_sensor = KalmanCanSensor(noise_variances=noise_variances)
        elif isinstance(sensor, ImuSensor):
            noise_variances = noise_variances if imu_variances is None else imu_variances
            kalman_sensor = KalmanImuSensor(noise_variances=noise_variances)
        else:
            assert False
        kalman_car.add_sensor(kalman_sensor)
    return kalman_car


# <a id='visualization'></a>
# ## Визуализация<sup>[toc](#toc)</sup>
# * [Визуализатор](#visualizator)
# * [Визуализация траектории автомобиля](#visualization_trajectory)
# * [Визуализация в реальном времени с оценками GPS](#visualizator_real_time)
#         
# Нужно какое-то графическое изображение движения автомобиля. Для этого написан специальный класс `CarPlotter`. В данном задании никаких изменений в него вносить не требуется &mdash; он работает как есть.

# <a id='visualizator'></a>
# ### Визуализатор<sup>[toc](#toc)</sup>

# In[14]:


class CarPlotter(object):
    def __init__(self, car_width=1, car_height=0.5,
                 real_color='g', obs_color='b', pred_color='r',
                 head_width=1):
        """
        :param car_width: Ширина автомобиля
        :param car_height: Длина автомобиля
        :param real_color: Цвет для отрисовки реального положения
        :param obs_color: Цвет для отрисовки наблюдений
        :param pred_color: Цвет для отрисовки калмановского предсказания
        :param head_width: Ширина стрелки при отрисовке скорости автомобиля
        """
        self.car_width = car_width
        self.car_height = car_height
    
        self.real_color = real_color
        self.obs_color = obs_color
        self.pred_color = pred_color

        self.real_vel_ = None
        self.obs_vel_ = None 
        self.head_width = head_width
        
    def plot_car(self, ax, car, marker_size=6):
        """Отрисовывает положение автомобиля и покзания GPS и одометрии.
        :param marker_size: Линейный размер точки положения и GPS-показания
        :param color: Цвет
        """
        assert isinstance(car, Car)
        real_position_x = car._position_x
        real_position_y = car._position_y
        real_velocity_x = car._velocity_x
        real_velocity_y = car._velocity_y

        # Отрисовка реального положения центра автомобиля
        real_position = np.array([real_position_x, real_position_y])
        self._plot_point(ax, real_position, marker='o', marker_color=self.real_color, marker_size=marker_size)
        # Отрисовка реального направления движения
        self.real_vel_ = plt.arrow(real_position_x, real_position_y,
                                   real_velocity_x, real_velocity_y,
                                   color=self.real_color,
                                   head_width=self.head_width) 
        # Отрисовка "прямоугольника" автомобиля
        angle = np.arctan2(real_velocity_y, real_velocity_x)
        y_rec = real_position_y - 0.5 * (self.car_height * np.cos(angle) + self.car_width * np.sin(angle))
        x_rec = real_position_x - 0.5 * (self.car_width * np.cos(angle) - self.car_height * np.sin(angle))
        rec = Rectangle(xy=(x_rec, y_rec), width=self.car_width, height=self.car_height,
                        angle=np.rad2deg(angle))
        rec.set_facecolor('none')
        rec.set_edgecolor('k')
        ax.add_artist(rec)
    
        # Если установлен GPS-датчик, то отрисовать показания GPS
        if car.gps_sensor is not None:
            gps_noise_covariance = car.gps_sensor.get_noise_covariance()
            self._plot_ellipse(ax, car.gps_sensor.observe(), gps_noise_covariance, color=self.obs_color)
            self._plot_point(ax, car.gps_sensor.observe(), marker='*',
                             marker_color=self.obs_color, marker_size=marker_size)

    def plot_kalman_car(self, ax, kalman_car):
        # Извлекаем состояние
        position_x = kalman_car._position_x
        position_y = kalman_car._position_y
        velocity_x = kalman_car._velocity_x
        velocity_y = kalman_car._velocity_y
        covariance = kalman_car.covariance_matrix
        # Отрисовка положения
        mu = np.array([position_x, position_y])
        sigma = covariance[:2, :2]
        self._plot_ellipse(ax, mu, sigma, color=self.pred_color)
        self._plot_point(ax, mu, marker='o', marker_size=6, marker_color=self.pred_color)
        # Отрисовка скорости
        mu = np.array([position_x + velocity_x, position_y + velocity_y])
        plt.arrow(position_x, position_y, velocity_x, velocity_y, color=self.pred_color,
                  head_width=self.head_width)
        
        
    def plot_trajectory(self, ax, car, traj_color='g'):
        """Отрисовывает весь уже проделанный автомобилем путь"""
        ax.plot(car._positions_x, car._positions_y, linestyle='-', color=traj_color)
        
    def plot_observations(self, ax, x, y, color='b'):
        ax.plot(x, y, linestyle='-', color=color)

    def get_limits(self, car):
        """Иногда требуется подогнать размер полотна, чтобы оно вмещало в себя всю траектория.
        Данный метод возвращает диапазоны значений вдоль каждой из осей.
        """
        min_pos_x = np.min(car._positions_x)
        max_pos_x = np.max(car._positions_x)
        min_pos_y = np.min(car._positions_y)
        max_pos_y = np.max(car._positions_y)
        max_vel_x = np.max(np.abs(car._velocities_x))
        max_vel_y = np.max(np.abs(car._velocities_y))
        # Дополнительная граница в полкорпуса
        max_length = max(self.car_width, self.car_height)
        x_limits = (min_pos_x - max_vel_x - 0.5 * max_length, max_pos_x + max_vel_x + 0.5 * max_length)
        y_limits = (min_pos_y - max_vel_y - 0.5 * max_length, max_pos_y + max_vel_y + 0.5 * max_length)
        return x_limits, y_limits
    
    def _plot_point(self, ax, mu, marker='o', marker_size=6, marker_color='b'):
        """Отрисовывает точку"""
        ax.scatter(mu[0], mu[1], marker=marker, color=marker_color, s=marker_size**2, edgecolors='k')

    def _plot_ellipse(self, ax, mu, sigma, color='b'):
        """Отрисовывает эллипс ковариации
        :param ax: полотно
        :param mu: цент нормального распределения
        :param sigma: ковариация нормального распределения
        :param marker: тип маркера для отображения центра
        :parma marker_size: линейный размер маркера для отображения центра
        :param color: цвет маркера и эллипса
        """
        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        lambda_, v = np.linalg.eig(sigma)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=mu, width=lambda_[0] * 2, height=lambda_[1] * 2,
                      angle=np.rad2deg(np.arccos(v[0, 0])), alpha=0.3, zorder=5)
        ell.set_edgecolor('k')
        ell.set_facecolor(color)
        ax.add_artist(ell)
        return ell


# <a id='visualization_trajectory'></a>
# ### Визуализация траектории автомобиля<sup>[toc](#toc)</sup>

# In[15]:


initial_position = [20, 20]
initial_velocity = 10
initial_yaw = 0.5
initial_omega = 0.02

car = Car(initial_position=initial_position, initial_velocity=initial_velocity,
          initial_yaw=initial_yaw, initial_omega=initial_omega)
print(car)

# Тестирование последовательного движения
movement_model = LinearMovementModel()
car.set_movement_model(movement_model)

dt = Timestamp.seconds(0.1)
duration = Timestamp.seconds(40)
final_time = car.time + duration
while car.time < final_time:
    car.move(dt)
print(car)

# Отрисовка траектории
fig = plt.figure(figsize=(15, 15))
ax = plt.subplot(111, aspect='equal')
ax.grid(which='both', linestyle='--', alpha=0.5)

car_plotter = CarPlotter(car_width=3, car_height=1.5)
car_plotter.plot_car(ax, car)
car_plotter.plot_trajectory(ax, car, traj_color='k')

# Установка корректных пределов
x_limits, y_limits = car_plotter.get_limits(car)
ax.set_xlim(x_limits)
ax.set_ylim(y_limits);


# <a id='visualizator_real_time'></a>
# ### Визуализация в реальном времени с оценками GPS<sup>[toc](#toc)</sup>

# In[16]:


# Создаем полотно
fig = plt.figure(figsize=(15, 15))
ax = plt.subplot(111, aspect='equal')
# ax.grid(which='both', linestyle='--', alpha=0.5)
real_color = 'green'
obs_color = 'blue'
est_color = 'red'

legend_lines = {'Measurement': Line2D([0], [0], color=obs_color, lw=4),
                'Real position': Line2D([0], [0], color=real_color, lw=4)}

# Добавляем в автомобиль сенсоры
initial_position= [5, 5]
initial_velocity = 10
initial_yaw = 0.1
initial_omega = 0.0
car = Car(
    initial_position=initial_position,
    initial_velocity=initial_velocity,
    initial_yaw=initial_yaw,
    initial_omega=initial_omega)
car.add_sensor(CanSensor(noise_variances=[0.25], random_state=1))  # std = 1 m/s
car.add_sensor(GpsSensor(noise_variances=[1., 1.], random_state=2))  # std = 3 m

# Тестирование последовательного движения
# movement_model = CycloidMovementModel(x_vel=2, y_vel=0.5, omega=0.4)
movement_model = LinearMovementModel()
car.set_movement_model(movement_model)
 
# Шаг интегрирования
dt = Timestamp.seconds(0.1)
# Длительность проезда
duration = Timestamp.seconds(10)

# Отрисовщик автомобиля и траектории
car_plotter = CarPlotter(car_width=3, car_height=1.5,
                         real_color=real_color, obs_color=obs_color, pred_color=est_color)
car_plotter.plot_car(ax, car)

final_time = car.time + duration

# Отрисовка начальной позиции
car_plotter.plot_car(ax, car)
display.clear_output(wait=True)
time.sleep(0.25)
ax.set_xlim(-5, 150)
ax.set_ylim(-5, 150)

while car.time < final_time:
    ax.clear()
    ax.legend([legend_lines['Measurement'], legend_lines['Real position']], ['Measurement', 'Real position'])
    car.move(dt)
    car_plotter.plot_car(ax, car)
    car_plotter.plot_trajectory(ax, car, traj_color=real_color)
    ax.set_xlim(-20, 75)
    ax.set_ylim(-20, 75)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.25)

display.clear_output(wait=True)


# <a id='test_kalman_localization'></a>
# ## Тестирование калмановской фильтрации<sup>[toc](#toc)</sup>

# In[17]:


car = create_car(initial_omega=0.05)
kalman_car = create_kalman_car(car)

# Создаем полотно
fig = plt.figure(figsize=(15, 15))
ax = plt.subplot(111, aspect='equal')
# ax.grid(which='both', linestyle='--', alpha=0.5)
real_color = 'green'
obs_color = 'blue'
est_color = 'red'

legend_lines = {'Kalman estimate': Line2D([0], [0], color=est_color, lw=4),
                'Measurement': Line2D([0], [0], color=obs_color, lw=4),
                'Real position': Line2D([0], [0], color=real_color, lw=4)}

# Шаг интегрирования
dt = Timestamp.seconds(0.1)
# Длительность проезда
duration = Timestamp.seconds(40)
final_time = car.time + duration

# Отрисовщик автомобиля и траектории
car_plotter = CarPlotter(
    car_width=3, car_height=1.5,
    real_color=real_color, obs_color=obs_color, pred_color=est_color)
car_plotter.plot_car(ax, car)

# Отрисовка начальной позиции
car_plotter.plot_car(ax, car)
display.clear_output(wait=True)
time.sleep(0.25)
ax.set_xlim(-10, 100)
ax.set_ylim(-10, 100)

while car.time < final_time:
    ax.clear()
    ax.legend([legend_lines['Kalman estimate'],
               legend_lines['Measurement'],
               legend_lines['Real position']], ['Kalman estimate', 'Measurement', 'Real position'])
    # Делаем реальный переход к моменту времени t + dt
    car.move(dt)
    # Делаем предсказание на момент времени t + dt
    kalman_car.move(dt)
    # Теперь обработаем наблюдения в момент t + dt
    for sensor, kalman_sensor in zip(car.sensors, kalman_car.sensors):
        observation = sensor.observe()
        kalman_sensor.process_observation(observation)
        
    car_plotter.plot_car(ax, car)
    car_plotter.plot_kalman_car(ax, kalman_car)
    car_plotter.plot_trajectory(ax, car, traj_color=real_color)
    car_plotter.plot_trajectory(ax, kalman_car, traj_color=est_color)
    car_plotter.plot_observations(ax, car.gps_sensor.history[:, 0], car.gps_sensor.history[:, 1], color=obs_color)
    ax.set_xlim(-5, 100)
    ax.set_ylim(-5, 100)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.25)

display.clear_output(wait=True)


# <a id='real_data'></a>
# # Реальные данные<sup>[toc](#toc)</sup>

# In[18]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install ipyleaflet')
get_ipython().system('pip install ipywidgets')


# In[19]:


import pandas as pd
import numpy as np
import os


# In[20]:


DATASETS_DIRECTORY = '/home/siri3us/yandexsdc/workspace/sdc/data'
icp_data = pd.read_csv(os.path.join(DATASETS_DIRECTORY, 'data_icp.csv'))
gps_data = pd.read_csv(os.path.join(DATASETS_DIRECTORY, 'data_gps.csv'))
imu_data = pd.read_csv(os.path.join(DATASETS_DIRECTORY, 'data_imu.csv'))
gt_data = pd.read_csv(os.path.join(DATASETS_DIRECTORY, 'data_gt.csv'))


# In[ ]:


print('{} icp messages'.format(len(icp_data)))
print('{} gps_messages'.format(len(gps_data)))
print('{} imu messages'.format(len(imu_data)))
print('{} gt messages'.format(len(gt_data)))


# In[ ]:


class IcpMessage:
    def __init__(self, secs, nsecs, latitude, longitude, altitude):
        self.stamp = Timestamp(sec=int(secs), nsec=int(nsecs))
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        

class GpsMessage:
    def __init__(self, secs, nsecs, latitude, longitude, altitude, ar):
        self.stamp = Timestamp(sec=int(secs), nsec=int(nsecs))
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.ar = ar


class ImuMessage:
    def __init__(self, secs, nsecs, wx, wy, wz, ax, ay, az):
        self.stamp = Timestamp(sec=int(secs), nsec=int(nsecs))
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.ax = ax
        self.ay = ay
        self.az = az


class GtMessage:
    def __init__(self, secs, nsecs, latitude, longitude, altitude):
        self.stamp = Timestamp(sec=int(secs), nsec=int(nsecs))
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude


def create_messages(data, msg_type):
    messages = []
    for index, row in data.iterrows():
        messages.append(msg_type(**row.to_dict()))
    return messages


# In[ ]:


icp_messages = create_messages(icp_data, IcpMessage)
gps_messages = create_messages(gps_data, GpsMessage)
imu_messages = create_messages(imu_data, ImuMessage)
gt_messages = create_messages(gt_data, GtMessage)


# ### Находим временные границы

# In[ ]:


import itertools

min_timestamp = icp_messages[0].stamp
max_timestamp = icp_messages[0].stamp

for msg in itertools.chain(icp_messages, gps_messages, imu_messages, gt_messages):
    if msg.stamp < min_timestamp:
        min_timestamp = msg.stamp
    elif msg.stamp > max_timestamp:
        max_timestamp = msg.stamp
        
print('min_timestamp={}'.format(min_timestamp))
print('max_timestamp={}'.format(max_timestamp))
print('ride duration={}'.format(max_timestamp.to_seconds()  - min_timestamp.to_seconds()))


# Используем только некоторую часть проезда

# In[ ]:


min_timestamp = Timestamp(sec=1616619000, nsec=0)
max_timestamp = Timestamp(sec=1616620000, nsec=0)


# ### Сэмплирование показаний

# In[ ]:


def create_track_from_messages(messages, period=1.0, min_timestamp=None, max_timestamp=None):
    """
    :messages: list or array of messages sorted in increasing order of their timestamps
    """
    if min_timestamp is None:
        min_timestamp = messages[0].stamp
    if max_timestamp is None:
        max_timestamp = messages[-1].stamp

    track = []
    prev_timestamp = Timestamp(0, 0)
    for msg in messages:
        # Checking if the message is within interval [min_timestamp, max_timestamp]
        if msg.stamp < min_timestamp or msg.stamp > max_timestamp:
            continue

        passed_time = msg.stamp.to_seconds() - prev_timestamp.to_seconds()
        assert passed_time > 0.
        if passed_time < period:
            continue
    
        track.append([msg.latitude, msg.longitude])
        prev_timestamp = msg.stamp

    return track


period = 5.
tracks = {}
tracks['ICP'] = create_track_from_messages(
    icp_messages, period=period,
    min_timestamp=min_timestamp, max_timestamp=max_timestamp)
tracks['GPS'] = create_track_from_messages(
    gps_messages, period=period,
    min_timestamp=min_timestamp, max_timestamp=max_timestamp)
tracks['GT'] = create_track_from_messages(
    gt_messages, period=period,
    min_timestamp=min_timestamp, max_timestamp=max_timestamp)
for track_type, track in tracks.items():
    print('Track of type {} has length {}'.format(track_type, len(track)))


# In[ ]:


import pandas as pd
from ipywidgets import HTML
from ipyleaflet import Map, Marker, Popup
from ipyleaflet import AntPath, WidgetControl
from ipywidgets import IntSlider, jslink

# SHAD coordinates
m = Map(center=(55.733039, 37.589153), zoom=15)

COLOR_BY_TYPE = {
    'GT': '#FA3100',    # RED
    'GPS': '#0F8FEC',   # BLUE
    'ICP': '#00BE34',   # GREEN
}

paths = {}
for track_type, track in tracks.items():
    path = AntPath(
        locations=track,
        dash_array=[1, 10],
        delay=1000,
        weight=3,
        use='polyline',
        color=COLOR_BY_TYPE[track_type],
        pulse_color=COLOR_BY_TYPE[track_type],
    )
    paths[track_type] = path
    m.add_layer(path)

    # Setting tag
    path_popup = HTML()
    path_popup.value = track_type
    path.popup = path_popup

start_marker = Marker(location=tracks['GT'][0])
m.add_layer(start_marker)

finish_marker = Marker(location=tracks['GT'][-1])
m.add_layer(finish_marker)

start = HTML()
finish = HTML()
start.value = "Старт"
finish.value = "Финиш"
start_marker.popup = start
finish_marker.popup = finish



zoom_slider = IntSlider(description='Масштаб:', min=11, max=20, value=14)
jslink((zoom_slider, 'value'), (m, 'zoom'))
widget_control1 = WidgetControl(widget=zoom_slider, position='topright')
m.add_control(widget_control1)

m


# ### Переход из latlong-координат в плоскость

# In[ ]:


get_ipython().system('pip install pyproj')


# <a id='assignment'></a>
# # Домашнее задание<sup>[toc](#toc)</sup>
# * [Реализация методов фильтра Калмана](#implement_kalman_methods)
# * [Оценка качества локализации](#localization_quality_estimation)
# * [Использование IMU](#use_imu)

# <a id='implement_kalman_methods'></a>
# ## Реализация методов фильтра Калмана<sup>[toc](#toc)</sup>

# <a id='localization_quality_estimation'></a>
# ## Оценка качества локализации<sup>[toc](#toc)</sup>

# ### Построение трэков<sup>[toc](#toc)</sup>
# На первом этапе требуется написать функцию, которая моделирует возвращает три трека:
# * Трэк с истинными положениями автомобиля
# * Трэк со значения GPS-сенсора
# * Трэк с калмановскими оценками положения

# In[ ]:


def calculate_tracks(car, kalman_car, duration, dt):
    """Двигает автомобиль вдоль траектории в течение времени duration с шагом dt.

    :param car: инициализированная модель машины
    :param kalman_car: инициализированный калмановская модель машины.
    :param duration: время моделирования
    :param dt: шаг интегрирования
    
    :returns: Возвращает три траектории: истинную, по GPS, из фильтра Калмана.
        Каждая траектория - это np.array размера (N, 2), где N - это количество шагов по времени, сделанных при
        моделировании
    """
    assert isinstance(car, Car)
    assert isinstance(kalman_car, KalmanCar)
    assert isinstance(duration, Timestamp)
    assert isinstance(dt, Timestamp)
    assert car.movement_model is not None
    assert car.gps_sensor is not None
    # Здесь требуется реализовать подсчет траекторий
    real_track = []
    gps_track = []
    kalman_track = []

    #TODO: Some code here

    return np.array(real_track), np.array(gps_track), np.array(kalman_track)


# In[ ]:


# Конфигурация автомобиля: начальное условие, модель движения, датчики
car = create_car(imu_noise_variances=[0.01])
# Конфигурация фильтра Калмана
kalman_car = create_kalman_car(car)
# Конфигурация фильтра Калмана
duration = Timestamp.seconds(40)
dt = Timestamp.seconds(0.1)

real_track, gps_track, kalman_track = calculate_tracks(car, kalman_car, duration, dt)

plt.figure(figsize=(15, 15))
plt.subplot(111, aspect='equal')
plt.plot(gps_track[:, 0], gps_track[:, 1], '.:', c="black", label='GPS measurement')
plt.plot(kalman_track[:, 0], kalman_track[:, 1], 'r.:', label="Kalman estimate")
plt.plot(real_track[:, 0], real_track[:, 1], 'gx', label="Real position")
plt.legend()


# <a id='hw_localization_quality'></a>
# ### Подсчет MSE<sup>[toc](#toc)</sup>
# В качестве оценка качества локализации можно взять $MSE$ между реальным положением автомобиля и оценкой, полученной из фильтра Калмана.
# 
# **Внимание!** В рамках всех экспериментов зафиксируйте одно значение шага `dt`.

# In[ ]:


def calculate_mse(kalman, kalman_car, duration, dt):
    return None

# Для полчения более стабильной оценки может потребоваться запустить calculate_mse несколько раз и затем
# усреднить результат. Для нескольих запусков с разными сидами можно использовать аргумент random_state
# у функции create_car


# In[ ]:


# Конфигурация автомобиля: начальное условие, модель движения, датчики
car = create_car(random_state=0)
# Конфигурация фильтра Калмана
kalman_car = create_kalman_car(car)
# Конфигурация фильтра Калмана
duration = Timestamp.seconds(40)
dt = Timestamp.seconds(0.05)

mse = calculate_mse(car, kalman_car, duration, dt)
print('MSE = {}'.format(mse))


# <a id='use_imu'></a>
# ## Использование IMU<sup>[toc](#toc)</sup>
# 
# Ни в одном из ранее рассмотренных примеров не использовался датчик IMU (гироскоп). Это можно заметить по реализации функции `create_car`. В рамках данного пункта ДЗ вам предлагается реализовать датчик IMU (`ImuSensor` в файле `sdc.imu_sensor`) и его калмановскую модель (`KalmanImuSensor` в файле `sdc.kalman_imu_sensor`). Напомним, что
# датчик IMU измеряет текущую угловую скорость машины, т.е. значение `car._omega`. Для добавления этого датчика достаточно действовать по аналогии с CAN-датчиком (файлы `sdc.can_sensor` и `sdc.kalman_can_sensor`).
# 
# * Реализовать IMU-датчик (`ImuSensor`) и его калмановскую модель (`KalmanImuSensor`)
# * Проверить гипотезу о том, что использование IMU-датчика улучшить качество локализации (уменьшить MSE)

# In[ ]:


# Конфигурация автомобиля: начальное условие, модель движения, датчики
car = create_car(random_state=0, imu_noise_variances=[0.01])
# Конфигурация фильтра Калмана
kalman_car = create_kalman_car(car)
# Конфигурация фильтра Калмана
duration = Timestamp.seconds(40)
dt = Timestamp.seconds(0.05)

mse = calculate_mse(car, kalman_car, duration, dt)
print('MSE = {}'.format(mse))

