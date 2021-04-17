#!/usr/bin/env python
# coding: utf-8

# $\newcommand{\Normal}{\mathcal{N}}
# \newcommand{\Prod}{\prod\limits}
# \newcommand{\Sum}{\sum\limits}
# \newcommand{\Int}{\int\limits}
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

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from IPython import display
import time
get_ipython().run_line_magic('matplotlib', 'inline')

from sdc.timestamp import Timestamp
from sdc.car import Car
from sdc.circle_movement_model import CircleMovementModel
from sdc.car_plotter import CarPlotter


# <a id='toc'></a>
# # Содержание
# * [Фильтр частиц](#particles_filter)
# * [Домашнее задание](#hw)
#     * [Введение](#hw_introduction)
#         * [Геометрия системы](#system_geometry)
#         * [Локальная и глобальная системы координат](#local_and_global_frames)
#         * [LandmarkSensor](#landmark_sensor)
#         * [Модель движения робота по окружности](#circle_movement_model)
#     * [Моделирование и локализация](#modelling)
#         * [Формируем геометрию области](#creating_region_geometry)
#         * [Модель эволюции системы](#particles_evolution)
#         * [Модель наблюдений](#observation_model)
#         * [Запуск моделирования](#running_simulation)

# <a id='particles_filter'></a>
# # Фильтр частиц<sup>[toc](#toc)</sup>

# В байесовской фильтрации мы имеем дело со следующими распределениями вектора состояния $\State_t$:
# * Прогноз $p(\State_t|\ObservHistory_{t - 1}) = p(\State_t|\Observ_1, \dots, \Observ_{t-1})$ &mdash; распределение вектора состояния на момент времени $\State_t$ при условии, что нам даны предшествующие данному моменту времени наблюдения $\Observ_1, \dots, \Observ_{t - 1}$.
# * Апостериорный прогноз:  $p(\State_t|\ObservHistory_{t}) = p(\State_t|\Observ_1, \dots, \Observ_{t})$ &mdash; распределение вектора состояния на момент времени $\State_t$ при условии, что нам даны все наблюдения до момента времени $\State$ включительно.
# 
# В процессе фильтрации эти два распределения пересчитываются друг из друга:
# $$
# p(\State_t|\ObservHistory_{t-1}) = \Int_{\State_{t-1}}p(\State_t|\State_{t-1})p(\State_{t-1}|\ObservHistory_{t-1})d\State_{t-1}
# $$
# $$
# p(\State_t|\ObservHistory_{t-1}) = \frac{1}{C}p(\Observ_t|\State_t)p(\State_t|\ObservHistory_{t-1})
# $$
# 
# В фильтре Калмана распределения $p(\State_t|\ObservHistory_{t-1})$ и $p(\State_t|\ObservHistory_t)$ являются нормальными, а потому параметризованы своими параметрами $\hatStateMean_t$, $\hatStateCov_t$, $\StateMean_t$, $\StateCov_t$, которые персчитываются в процессе поступления новых данных.
# 
# В фильтре частиц распределения $p(\State_t|\ObservHistory_{t-1})$ и $p(\State_t|\ObservHistory_t)$ представлены не своими параметрами, а набором сэмплированных из этих распределний векторов. Т.е. фактически в фильтре частиц не делается предположений о типе распределений $p(\State_t|\ObservHistory_{t-1})$ и $p(\State_t|\ObservHistory_t)$.

# ### Алгоритм работы фильтра частиц
# * В начальный момент времени $t = 0$ некоторым образом сэмплируются $N$ частиц $\{\State_{i, 0}\}_{i=1}^N$ (как именно, определяется исходя из специфики задачи).
# * Переход из состояния в момента $t - 1$ в момент $t$ осуществляется путем сэмплирования нового набора частиц из уже существующего, т.е. новая $i$-ая частица $\State_{i, t}$ получается путем сэмплирования из распределения $p(\State_{t}|\State_{i, t - 1})$:
# $$
# \State_{i, t} \sim p(\State_{t}|\State_{i, t-1})
# $$
# * Обновление состояния при обработке наблюдения $\Observ_t$ происходит следующим образом.
#     * Для каждой частицы определяется ее вес $w_i = p(\Observ_t|\State_{i, t})$
#     * Полученные веса нормируются и используется для (ре)сэмплирования $N$ частиц из множества $\{\State_{i,t}\}_{i=1}^N$

# <a id='hw'></a>
# # Домашнее задание<sup>[toc](#toc)</sup>

# В данном задании требуется реализовать фильтр частиц и с помощью него локализовать положение робота с использованием маяков &mdash; некоторых характерных объектов, положение которых в реальном мире известно.
# Идея локализации на основе маяков состоит в том, робот может понять свое положение в реальном мире за счет наблюдения за маяками.

# <a id='hw_introduction'></a>
# ## Введение<sup>[toc](#toc)</sup>

# <a id='system_geometry'></a>
# ### Геометрия системы<sup>[toc](#toc)</sup>
# Робот движется по окружности внутри квадрата размера 100м x 100м. Радиус окружности 50м. Центр квадрата и окружности совпадают с началом координат. Маяки расположены по углам квадрата. Робота оснащен датчиками, каждый из которых следит одним из 4-х маяков и по запросу возвращает положение маяка относительно робота (в локальной системе координат). Робот начинает движение в нижней точке окружности (см. рис.).

# ![](pics/region_geometry.jpg)

# <a id='local_and_global_frames'></a>
# ### Локальная и глобальная системы координат<sup>[toc](#toc)</sup>

# ![](pics/coordinate_systems.jpg)
# На рис. выше изображены глобальная и локальная система координат, связанная с роботом.
# 
# Для перехода из системы координат робота в глобальную систему координат, получаем
# $$
# \begin{pmatrix}
# x_g\\
# y_g
# \end{pmatrix} = 
# \begin{pmatrix}
# \cos\gamma &-\sin\gamma\\
# \sin\gamma &\cos\gamma
# \end{pmatrix}
# \begin{pmatrix}
# x_l\\
# y_l
# \end{pmatrix} +
# \begin{pmatrix}
# x\\
# y
# \end{pmatrix}
# $$
# Для перехода из глобальной системы координат в локальную, соответственно, получаем
# $$
# \begin{pmatrix}
# x_l\\
# y_l
# \end{pmatrix} = 
# \begin{pmatrix}
# \cos\gamma &\sin\gamma\\
# -\sin\gamma &\cos\gamma
# \end{pmatrix}
# \begin{pmatrix}
# x_g - x\\
# y_g - y
# \end{pmatrix}
# $$

# <a id='landmark_sensor'></a>
# ### LandmarkSensor<sup>[toc](#toc)</sup>
# Данный сенсор следит за положением маяка. Будучи прикрепленным к роботу/машине, данный сенсор по запросу возвращает положение маяка в локальной системе координат.

# In[2]:


from sdc.sensor_landmark import (
    LandmarkSensor,
    get_landmark_position_in_local_frame
)


# In[3]:


# При создании указываем положение маяка в глобальной системе координат
landmark_sensor = LandmarkSensor(x=17, y=45)  
car = Car(initial_position=[0, 0])
car.add_sensor(landmark_sensor)
print('Observed landmark position: {}'.format(landmark_sensor.observe()))

landmark_sensor = LandmarkSensor(x=100, y=100)  
car = Car(initial_position=[100, 0])
car.add_sensor(landmark_sensor)
print('Observed landmark position: {}'.format(landmark_sensor.observe()))


# В первом примере локальная и глобальная системы координат совпадают, а потом наблюдаемое положение в локальной системе координат совпадает с глобальным. Во втором случае же локальная система смещена на уровень X=100, из-за чего положение в локальной системе координат ожидаемо изменяется.
# 
# Далее при решении задачи локализации таких сенсора потребуется установить четыре: по одному на каждый угол квадрата.

# <a id='circle_movement_model'></a>
# ### Модель движения робота по окружности<sup>[toc](#toc)</sup>
# Как и в случае с фильтром Калмана, тут мы также будем искуственно генерировать наблюдения положений маяков. Но для этого нам нужно реализовать движение ровера по окружности. Это сделано в рамках `CircleMovementModel`. При подключении данной модели к модели робота производится вывод параметров движения по окружности на основе начального состояния робота:
# \begin{aligned}
# x(t) &= x_c + r \cos(\omega(t - t_0))\\
# y(t) &= y_c + r \sin(\omega(t - t_0))\\
# v_x(t) &= -r\omega \sin(\omega(t - t_0))\\
# v_y(t) &= r\omega \cos(\omega(t - t_0))
# \end{aligned}
# В начальном состоянии заданы $x(t)$, $y(t)$, $v_x(t) = v \cos\gamma$, $v_y(t) = v \sin\gamma$. Из них выводятся значения $x_c$, $y_c$, $r$, $t_0$.

# In[4]:


# 1. Создаем модель машины/робота
radius = 50.
angular_velocity = 0.2
linear_velocity = abs(angular_velocity * radius)
# Начальное положение и yaw выбраем таким образом, чтобы движение начиналось в нижней точке окружности
initial_position = [0, -radius]
yaw = 0.

car = Car(
   initial_position=initial_position,
   initial_velocity=linear_velocity,
   initial_yaw=yaw,
   initial_omega=angular_velocity)
movement_model = CircleMovementModel()
car.set_movement_model(movement_model)

# 2. Задаем временные рамки моделирования процесса движения
initial_time = car.time
# Рассчитываем время окончания движения так, чтобы робот совершил один полный оборот
final_time = initial_time + Timestamp.seconds(2 * np.pi / angular_velocity)
dt = Timestamp(0, 100000000)

# 3. Проводим моделирование в рамках заданного временного интервала
current_time = initial_time
while current_time < final_time:
   car.move(dt)
   current_time = car.time

# 4. Визуализируем результаты
# Отрисовка траектории
fig = plt.figure(figsize=(15, 15))
ax = plt.subplot(111, aspect='equal')
ax.grid(which='both', linestyle='--', alpha=0.5)

car_plotter = CarPlotter(car_width=3, car_height=1.5)
car_plotter.plot_car(ax, car)
car_plotter.plot_trajectory(ax, car, traj_color='k')

# Установка корректных пределов
x_limits, y_limits = car_plotter.get_limits(car)
ax.set_xlim(x_limits);
ax.set_ylim(y_limits);


# <a id='modelling'></a>
# ## Моделирование и локализация<sup>[toc](#toc)</sup>

# <a id='creating_region_geometry'></a>
# ### Формируем геометрию области<sup>[toc](#toc)</sup>

# * **Рассматриваемая область пространства:** квадрат со стороной 100м и центром в точке (0, 0)
# * **Положения маяков:** по углам квадрата

# In[5]:


SQUARE_REGION_SIDE = 100.
LANDMARKS_REAL_POSITIONS = np.array([
    [SQUARE_REGION_SIDE / 2., SQUARE_REGION_SIDE / 2.],    # Маяк в правом верхнем углу
    [SQUARE_REGION_SIDE / 2., -SQUARE_REGION_SIDE / 2.],   # Маяк в правом нижнем углу
    [-SQUARE_REGION_SIDE / 2., -SQUARE_REGION_SIDE / 2.],  # Маяк в левом нижнем углу
    [-SQUARE_REGION_SIDE / 2., SQUARE_REGION_SIDE / 2.]    # Маяк в левом верхнем углу
], dtype=np.float64)

plt.figure(figsize=(7, 7));
plt.scatter(x=LANDMARKS_REAL_POSITIONS[:, 0], y=LANDMARKS_REAL_POSITIONS[:, 1], color='r');


# ### Создаем модель робота, движущегося по окружности<sup>[toc](#toc)</sup>

# In[46]:


radius = SQUARE_REGION_SIDE / 2.
angular_velocity = 0.2
linear_velocity = abs(angular_velocity * radius)
# Начальное положение и yaw выбраем таким образом, чтобы движение начиналось в нижней точке окружности
initial_position = [0, -radius]
yaw = 0.

car = Car(
    initial_position=initial_position,
    initial_velocity=linear_velocity,
    initial_yaw=yaw,
    initial_omega=angular_velocity)
movement_model = CircleMovementModel()
car.set_movement_model(movement_model)


# #### Добавляем роботу сенсоры для определения положений маяков

# In[47]:


# Дисперсии наблюдений landmark-сенсоров
LANDMARK_SENSOR_VARIANCES = [5., 5.]

for n_landmark, landmark_real_position in enumerate(LANDMARKS_REAL_POSITIONS):
    landmark_sensor = LandmarkSensor(
        x=landmark_real_position[0],
        y=landmark_real_position[1],
        noise_variances=LANDMARK_SENSOR_VARIANCES,
        random_state=n_landmark)
    car.add_sensor(landmark_sensor)
    
for landmark_sensor in car.landmark_sensors:
    print(landmark_sensor.observe())
    
assert len(car.landmark_sensors) == 4


# ### Сэмплирование начального распределения<sup>[toc](#toc)</sup>
# Для задачи локализации будем считать, что вектор состояния имеет вид $(x, y, v, \gamma)^T \in \RR^4$. 
# Требуется каким-нибудь образом насэмплировать начальный набор частиц (не менее 1000).
# 
# Возможный подход к сэмплированию:
# * $x$ из равномерного распределия $Uniform[-50, 50]$
# * $y$ из равномерного распределия $Uniform[-50, 50]$
# * $v$ из равномерного распределия $Uniform[0, 10]$
# * $\gamma$ из равномерного распределия $Uniform[0, 2\pi]$

# In[57]:


# TODO: Implement
PARTICLES_NUMBER = int(1e4)
# np.random.seed(0)
rng = np.random.default_rng(seed=0)
Xs = rng.uniform(-50, 50, size=PARTICLES_NUMBER)
Ys = rng.uniform(-50, 50, size=PARTICLES_NUMBER)
Vs = rng.uniform(0, 10, size=PARTICLES_NUMBER)
Yaws = rng.uniform(0, 2 * np.pi, size=PARTICLES_NUMBER)
particles = np.vstack([Xs, Ys, Vs, Yaws])
assert particles.shape == (4, PARTICLES_NUMBER)

plt.figure(figsize=(7, 7))
plt.scatter(x=LANDMARKS_REAL_POSITIONS[:, 0], y=LANDMARKS_REAL_POSITIONS[:, 1], color='r');
plt.scatter(x=particles[0], y=particles[1]);


# <a id='particles_evolution'></a>
# ### Модель эволюции системы<sup>[toc](#toc)</sup>
# Согласно описанию фильтра частиц процесс перехода из момента времени $t$ в момент $t + \Delta t$ производится путем сэмплирования нового набора точек $\{\State(t + \Delta t)\}$ из $\{\State(t)\}$, т.е.
# $$
# \State(t + \Delta t) \sim p(\State(t + \Delta t)|\State(t)).
# $$
# Для этого нам нужно распредление $p(\State(t + \Delta t)|\State(t))$ (модель эволюции системы). В рамках рассматриваемых частиц можно использовать следующую модель эволюции:
# $$
# \begin{pmatrix}
# x(t + \Delta t)\\
# y(t + \Delta t)\\
# v(t + \Delta t)\\
# \gamma(t + \Delta t)\\
# \end{pmatrix} \approx
# \begin{pmatrix}
# x(t) + v(t) \cos\gamma(t) \cdot \Delta t\\
# y(t) + v(t) \sin\gamma(t) \cdot \Delta t\\
# v(t)\\
# \gamma(t)
# \end{pmatrix} + \boldr,
# $$
# где $\boldr \sim \Normal(\boldzero, R)$. Таким образом
# $$
# p(\State(t + \Delta t)|\State(t)) = \Normal(\State(t + \Delta t)|f(\State(t), \Delta t), R),
# $$
# где
# $$
# f(\State(t), \Delta t) = \begin{pmatrix}
# x(t) + v(t) \cos\gamma(t) \cdot \Delta t\\
# y(t) + v(t) \sin\gamma(t) \cdot \Delta t\\
# v(t)\\
# \gamma(t)
# \end{pmatrix}
# $$
# Реализуйте функцию `move_particles`, которая принимает на вход текущее множество частиц и сэмплирует из них новое множество согласно описанной выше процедуре. Матрицу $R$ выберите на свое усмотрение (она может быть как константной, так и зависящей от $\Delta t$).

# In[58]:


# TODO: implement
def move_particles(particles, dt):
    """
    :param particles: np.array размера (4, N), где N - количество частиц
    :param dt: Timestamp, шаг по времени, на который требуется продвинуться
    """
    dt_in_seconds = dt.to_seconds()
    N = particles.shape[1]
    R = np.ones(4) * dt_in_seconds
    return np.array([particles[0] + particles[2] * np.cos(particles[3]) * dt_in_seconds,
                     particles[1] + particles[2] * np.sin(particles[3]) * dt_in_seconds,
                     particles[2],
                     particles[3]]) + multivariate_normal.rvs(mean=np.zeros(4), cov=R, size=N).T


# #### Проверка модели эволюции

# In[59]:


# Attention: после запуска данной ячейки частицы желательно перегенерировать перед дальшнейшими действиями
initial_time = Timestamp(sec=0, nsec=0)
final_time = Timestamp(sec=1, nsec=0)
dt = Timestamp(0, 100000000)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, aspect='equal')

current_time = initial_time
while current_time < final_time:
    current_time += dt
    particles = move_particles(particles, dt)

    ax.clear()
    ax.scatter(particles[0], particles[1])
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.25)

display.clear_output(wait=True)


# <a id='observation_model'></a>
# ### Модель наблюдений<sup>[toc](#toc)</sup>

# На каждой итерации движения робота мы будем запрашивать наблюдаемые положения маяков, после чего пересчитывать частицы. Для этого нам нужна модель наблюдений $p(\Observ(t)|\State(t), \boldm)$, где
# * $\State(t)$ &mdash; заданное состояние системы (частица)
# * $\boldm \in \RR^2$ &mdash; реальное положение наблюдаемого маяка в глобальной системе координат
# * $\Observ(t) \in \RR^2$ &mdash; наблюдаемое положение маяка в в локальной системе координат
# 
# Будем считать, что погрешности в наблюдениях имеют нормальное распределение, тогда
# $$
# \Observ(t)\sim \Normal(T(\State(t)) \otimes \boldm; Q),
# $$
# где $T(\State(t))$ &mdash; оператор перехода из глобальной системы координат в локальную. Фактически реализован в виде функции `find_landmark_position_in_local_frame`.
# 
# Согласно процедуре байесовской фильтрации требуется определить веса $w_i \propto p(\Observ(t)|\State_i(t))$ и провести ресэмплирование частиц $\{\State_i(t)\}_{i=1}^N$ с данными весами.
# 
# В рамках рассматриваемой задачи мы получаем наблюдения положений сразу 4-х маяков. Будем считать, что мы знаем, какому реальному маяку какое наблюдение соотвествует, т.е. будем работать в режиме известных соотвествий (`matched = True`). Тогда
# $$
# p(\Observ_1(t), \Observ_2(t), \Observ_3(t), \Observ_4(t)|\State(t), \boldM) = \Prod_{i = 1}^4 p(\Observ_i(t)|\State(t),\boldm_i).
# $$
# Тут будет удобнее работать с логарифами:
# $$
# \log w_i = \log p(\Observ_1(t), \Observ_2(t), \Observ_3(t), \Observ_4(t)|\State(t), \boldM) = \Sum_{i = 1}^4 \log p(\Observ_i(t)|\State(t),\boldm_i)
# $$
# (`scipy.stats.multivariate_normal.logpdf` в помощь). Затем провести softmax-преобразование надо полученными логарифмами весов, чтобы получить распредление для сэмплирования частиц.

# In[60]:


from scipy.stats import multivariate_normal
from scipy.special import softmax

from sdc.sensor_landmark import get_landmark_position_in_local_frame

# TODO: Implement


# Вспомогательная функция для подсчета логарифма правдоподобия
def get_landmarks_observations_logprob(
        particle,
        landmarks_observed_positions,
        landmarks_real_positions):
    """Считает логарифм правдоподобия p(observ|state) для одной частицы"""
    local_landmarks = [get_landmark_position_in_local_frame(particle[0], particle[1], particle[3],
                                                            real_pos[0], real_pos[1])
                       for real_pos in landmarks_real_positions]
    lhood = np.sum([multivariate_normal.logpdf(landmarks_observed_positions[lm_idx],
                                               mean=local_landmarks[lm_idx],
                                               cov=np.diag(LANDMARK_SENSOR_VARIANCES)*5
                                               )
                    for lm_idx in range(len(landmarks_real_positions))])
    return lhood

def process_landmarks_observations(
        particles,
        landmarks_observed_positions,
        landmarks_real_positions):
    """
    :param particles: np.array размера (4, N), где N - количество частиц
    :param landmarks_observed_positions: np.array размера (M, 2), где M - количество маяков.
        Содержит результаты наблюдений от сенсоров, т.е. положения маяков в локальной системе координат.
    :param landmarks_real_positions: np.array размера (M, 2), где M - количество маяков.
        Содержит реальные положения маяков в глобальной системе координат.
    """
    particles_number = particles.shape[1]
    observed_np = np.array(landmarks_observed_positions)
    
    lhood = [get_landmarks_observations_logprob(particles[:, p_idx],
                                                landmarks_observed_positions,
                                                landmarks_real_positions)
             for p_idx in range(particles.shape[1])]
    weights = softmax(lhood)

    sampled_particles_indices = np.random.choice(np.arange(particles_number), size=particles_number, p=weights)
    return particles[:, sampled_particles_indices]


# <a id='running_simulation'></a>
# ### Запуск моделирования<sup>[toc](#toc)</sup>

# In[61]:


# Внимание: возможно имеет смысл поместить сюда код создания области, создания робота и создания частиц
# чтобы убедиться, что данные не были случайно изменены в рамках перезапуска каких-либо ячеек
radius = SQUARE_REGION_SIDE / 2.
angular_velocity = 0.2
linear_velocity = abs(angular_velocity * radius)
# Начальное положение и yaw выбраем таким образом, чтобы движение начиналось в нижней точке окружности
initial_position = [0, -radius]
yaw = 0.

car = Car(
    initial_position=initial_position,
    initial_velocity=linear_velocity,
    initial_yaw=yaw,
    initial_omega=angular_velocity)
movement_model = CircleMovementModel()
car.set_movement_model(movement_model)

LANDMARK_SENSOR_VARIANCES = [5., 5.]

for n_landmark, landmark_real_position in enumerate(LANDMARKS_REAL_POSITIONS):
    landmark_sensor = LandmarkSensor(
        x=landmark_real_position[0],
        y=landmark_real_position[1],
        noise_variances=LANDMARK_SENSOR_VARIANCES,
        random_state=n_landmark)
    car.add_sensor(landmark_sensor)
    
for landmark_sensor in car.landmark_sensors:
    print(landmark_sensor.observe())
    
assert len(car.landmark_sensors) == 4

PARTICLES_NUMBER = int(1e3)
# np.random.seed(0)
rng = np.random.default_rng(seed=42)
Xs = rng.uniform(-50, 50, size=PARTICLES_NUMBER)
Ys = rng.uniform(-50, 50, size=PARTICLES_NUMBER)
Vs = rng.uniform(0, 10, size=PARTICLES_NUMBER)
Yaws = rng.uniform(0, 2 * np.pi, size=PARTICLES_NUMBER)
particles = np.vstack([Xs, Ys, Vs, Yaws])
assert particles.shape == (4, PARTICLES_NUMBER)

plt.figure(figsize=(7, 7))
plt.scatter(x=LANDMARKS_REAL_POSITIONS[:, 0], y=LANDMARKS_REAL_POSITIONS[:, 1], color='r');
plt.scatter(x=particles[0], y=particles[1]);


# In[62]:


initial_time = car.time
# Рассчитываем время окончания движения так, чтобы робот совершил один полный оборот
final_time = initial_time + Timestamp.seconds(2 * np.pi / angular_velocity)
# Или же выставляем его вручную на 1 секунду (иначе может быть очень много итераций)
# final_time = Timestamp(1)
dt = Timestamp(0, 100000000)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, aspect='equal')

car_plotter = CarPlotter(car_width=3, car_height=1.5)

current_time = initial_time
while current_time < final_time:
    car.move(dt)
    current_time = car.time

    # Заправшиваем текущие наблюдения маяков у сенсоров
    landmarks_observed_positions = []
    for landmark_sensor in car.landmark_sensors:
        landmarks_observed_positions.append(landmark_sensor.observe())

    # Продвигаем частицы до текущего момента времени
    particles = move_particles(particles, dt)
    # Обрабатываем наблюдения
    particles = process_landmarks_observations(
        particles=particles,
        landmarks_observed_positions=landmarks_observed_positions,
        landmarks_real_positions=LANDMARKS_REAL_POSITIONS)

    ax.clear()
    ax.scatter(x=particles[0], y=particles[1])
    ax.scatter(x=LANDMARKS_REAL_POSITIONS[:, 0], y=LANDMARKS_REAL_POSITIONS[:, 1], color='r')
    car_plotter.plot_car(ax, car)
    car_plotter.plot_trajectory(ax, car, traj_color='k')
    ax.set_xlim([-1.1 * SQUARE_REGION_SIDE / 2.0, 1.1 * SQUARE_REGION_SIDE / 2.0])
    ax.set_ylim([-1.1 * SQUARE_REGION_SIDE / 2.0, 1.1 * SQUARE_REGION_SIDE / 2.0])

    display.clear_output(wait=True)
    display.display(plt.gcf())
#     time.sleep(0.25)
    
display.clear_output(wait=True)


# Проверьте, насколько хорошо полученные с помощью фильтра частиц оценки положения, скорости и угла поворота соответствуют реальным данным.

# In[63]:


mean_particle = np.mean(particles, axis=1)
estimated_x = mean_particle[0]
estimated_y = mean_particle[1]
estimated_v = mean_particle[2]
estimated_yaw = mean_particle[3]

print('X:   real = {}, estimate = {}'.format(car._position_x, estimated_x))
print('Y:   real = {}, estimate = {}'.format(car._position_y, estimated_y))
print('V:   real = {}, estimate = {}'.format(car._linear_velocity, estimated_v))
print('Yaw: real = {}, estimate = {}'.format(car._yaw % (2 * np.pi), estimated_yaw % (2 * np.pi)))


# Если фильтр частиц работает, и визуально видно, что облако точек концентрируется около положения робота, то ДЗ можно считать выполненным.

# In[ ]:




