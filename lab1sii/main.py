import random
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from functools import partial

# =========================
# 1. ДАННЫЕ
# =========================
products = [
    ("Куриная грудка", 250, 165, 31, 3.6, 0),
    ("Говядина",       400, 250, 26, 15,  0),
    ("Рис",             80, 360,  7, 0.6, 78),
    ("Гречка",          90, 343, 13, 3.4, 72),
    ("Молоко",          70,  42, 3.4, 1,   5),
    ("Яйца",           120, 155, 13, 11,   1.1),
    ("Яблоко",          60,  52, 0.3, 0.2, 14),
    ("Банан",           65,  96, 1.3, 0.3, 27),
    ("Сыр",            500, 402, 25, 33,   1.3),
    ("Йогурт",          90,  59, 10, 0.4,  3.6),
    ("Картофель",       50,  77,  2, 0.1, 17),
    ("Лосось",         600, 208, 20, 13,   0),
]

N = len(products)
K = 5

NORMS = {
    "kcal":   (2000, 2500),
    "protein": (75,  150),
    "fat":    (50,  100),
    "carbs":  (250, 400)
}

# =========================
# 2. ПОЛНЫЙ ПЕРЕБОР (для маленьких N)
# =========================
def brute_force():
    best_cost = float("inf")
    best_combo = None
    for combo in itertools.combinations(range(N), K):
        cost = sum(products[i][1] for i in combo)
        kcal   = sum(products[i][2] for i in combo)
        prot   = sum(products[i][3] for i in combo)
        fat    = sum(products[i][4] for i in combo)
        carbs  = sum(products[i][5] for i in combo)

        if (NORMS["kcal"][0]   <= kcal   <= NORMS["kcal"][1]   and
            NORMS["protein"][0] <= prot   <= NORMS["protein"][1] and
            NORMS["fat"][0]     <= fat    <= NORMS["fat"][1]     and
            NORMS["carbs"][0]   <= carbs  <= NORMS["carbs"][1]):
            if cost < best_cost:
                best_cost = cost
                best_combo = combo
    return best_cost, best_combo


# =========================
# 3. ФУНКЦИЯ ОЦЕНКИ
# =========================
def evaluate(individual):
    cost = kcal = prot = fat = carbs = 0
    selected_count = sum(individual)

    for i, selected in enumerate(individual):
        if selected:
            p = products[i]
            cost  += p[1]
            kcal  += p[2]
            prot  += p[3]
            fat   += p[4]
            carbs += p[5]

    penalty = 0

    # Штраф за количество продуктов (самый строгий)
    if selected_count != K:
        penalty += 20000 * abs(selected_count - K)

    # Штрафы за выход за диапазон
    for value, (low, high), w in [
        (kcal,   NORMS["kcal"],   5),
        (prot,   NORMS["protein"], 8),
        (fat,    NORMS["fat"],    6),
        (carbs,  NORMS["carbs"],  4)
    ]:
        if value < low:
            penalty += (low - value) * w
        elif value > high:
            penalty += (value - high) * w

    return cost + penalty,


# =========================
# 4. СОЗДАНИЕ ОСОБИ (ровно K единиц)
# =========================
def create_individual():
    ind = [0] * N
    indices = random.sample(range(N), K)
    for i in indices:
        ind[i] = 1
    return ind


# =========================
# 5. КАСТОМНАЯ МУТАЦИЯ (обмен 0↔1)
# =========================
def custom_mutation(individual):
    ones  = [i for i, g in enumerate(individual) if g == 1]
    zeros = [i for i, g in enumerate(individual) if g == 0]
    if ones and zeros:
        i = random.choice(ones)
        j = random.choice(zeros)
        individual[i] = 0
        individual[j] = 1
    return individual,


# =========================
# 6. НАСТРОЙКА DEAP
# =========================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("select",   tools.selTournament, tournsize=3)

toolbox.register("mate",   tools.cxTwoPoint)           # будет переопределяться
toolbox.register("mutate", tools.mutFlipBit, indpb=0.08)


# =========================
# 7. ЗАПУСК ГА
# =========================
def run_ga(cx_operator, mut_operator, cxpb=0.75, mutpb=0.25, ngen=120, popsize=180):
    toolbox.register("mate",   cx_operator)
    toolbox.register("mutate", mut_operator)

    pop = toolbox.population(n=popsize)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cxpb, mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    return logbook.select("min"), hof[0]


# =========================
# 8. ВЫВОД РЕЗУЛЬТАТОВ ОСОБИ
# =========================
def print_solution(ind, label=""):
    cost, = evaluate(ind)
    kcal = prot = fat = carbs = 0
    selected = []

    for i, g in enumerate(ind):
        if g:
            p = products[i]
            selected.append(p[0])
            kcal  += p[2]
            prot  += p[3]
            fat   += p[4]
            carbs += p[5]

    print(f"\n{label}")
    print(f"Стоимость + штраф = {cost:.0f}")
    print(f"Продуктов выбрано: {sum(ind)} из {K}")
    print("Выбранные продукты:", ", ".join(selected))
    print(f"kcal: {kcal}   protein: {prot}   fat: {fat}   carbs: {carbs}")


# =========================
# 9. ОСНОВНАЯ ПРОГРАММА
# =========================
if __name__ == "__main__":
    random.seed(42)

    print("=== ПОЛНЫЙ ПЕРЕБОР ===")
    start = time.time()
    bf_cost, bf_indices = brute_force()
    bf_time = time.time() - start

    if bf_indices:
        print("Лучшая стоимость:", bf_cost)
        print("Продукты:", ", ".join(products[i][0] for i in bf_indices))
    else:
        print("Решение в жёстких рамках не найдено")
    print(f"Время: {bf_time:.4f} сек\n")

    # ─── Эксперименты ГА ──────────────────────────────────────
    crossovers = {
        "OnePoint": tools.cxOnePoint,
        "TwoPoint": tools.cxTwoPoint,
        "Uniform":  partial(tools.cxUniform, indpb=0.5),
    }

    mutations = {
        "FlipBit":  partial(tools.mutFlipBit, indpb=0.07),
        "Shuffle":  partial(tools.mutShuffleIndexes, indpb=0.18),
        "Custom":   custom_mutation,
    }

    results = {}
    print("=== ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ===\n")

    for cx_name, cx in crossovers.items():
        for mut_name, mut in mutations.items():
            combo_name = f"{cx_name} + {mut_name}"
            print(f"→ {combo_name}")

            start = time.time()
            fitness_history, best_ind = run_ga(cx, mut)
            ga_time = time.time() - start

            print_solution(best_ind, f"Лучшее решение ({combo_name})")
            print(f"Время: {ga_time:.3f} сек")
            print("-" * 60)

            results[combo_name] = fitness_history

    # ─── График ───────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    for label, hist in results.items():
        plt.plot(hist, label=label, linewidth=1.4, alpha=0.9)
    plt.xlabel("Поколение")
    plt.ylabel("Значение функции (цена + штраф)")
    plt.title("Сравнение комбинаций скрещивания и мутации")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()