import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt



# ================================
# ШАГ 1. СОЗДАНИЕ ЛИНГВИСТИЧЕСКОЙ ПЕРЕМЕННОЙ
# ================================

# Универсум температуры от -10 до 40 градусов
temperature = ctrl.Antecedent(np.arange(-10, 41, 1), 'temperature')


# ================================
# ШАГ 2. ТРАПЕЦИЕВИДНЫЕ ФУНКЦИИ ПРИНАДЛЕЖНОСТИ
# ================================

temperature['cold'] = fuzz.trapmf(temperature.universe, [-10, -10, 0, 10])
temperature['cool'] = fuzz.trapmf(temperature.universe, [5, 10, 15, 20])
temperature['warm'] = fuzz.trapmf(temperature.universe, [15, 20, 25, 30])
temperature['hot']  = fuzz.trapmf(temperature.universe, [25, 30, 40, 40])


# ================================
# ШАГ 3. ВЫБОР ДВУХ МНОЖЕСТВ ДЛЯ ОБЪЕДИНЕНИЯ
# ================================

print("Доступные термы температуры:")
print("cold, cool, warm, hot")

term1 = input("Введите первый терм: ")
term2 = input("Введите второй терм: ")

mf1 = temperature[term1].mf
mf2 = temperature[term2].mf


# ================================
# ШАГ 4. ОПЕРАЦИЯ ОБЪЕДИНЕНИЯ (НЕЧЕТКОЕ ИЛИ)
# ================================

union_mf = np.fmax(mf1, mf2)


# ================================
# ШАГ 5. ВВОД ЧЕТКОГО ЗНАЧЕНИЯ
# ================================

value = float(input("Введите четкое значение температуры: "))

mu1 = fuzz.interp_membership(temperature.universe, mf1, value)
mu2 = fuzz.interp_membership(temperature.universe, mf2, value)
mu_union = max(mu1, mu2)

print("\nРЕЗУЛЬТАТ:")
print(f"μ({term1}) = {mu1:.3f}")
print(f"μ({term2}) = {mu2:.3f}")
print(f"μ(объединение) = {mu_union:.3f}")


# ================================
# ШАГ 6. ВИЗУАЛИЗАЦИЯ
# ================================

plt.figure()
plt.plot(temperature.universe, mf1, label=term1)
plt.plot(temperature.universe, mf2, label=term2)
plt.plot(temperature.universe, union_mf, '--', label="Union (max)")

plt.title("Объединение нечетких множеств (Температура)")
plt.xlabel("Температура (°C)")
plt.ylabel("Степень принадлежности μ(x)")
plt.legend()
plt.grid()
plt.show()