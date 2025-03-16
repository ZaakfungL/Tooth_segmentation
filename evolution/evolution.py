import random
import numpy as np
from typing import List, Tuple
from .individual import Individual


class Evolution:
    def __init__(self, population_size: int, generations: int, search_space,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.search_space = search_space
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[Individual] = []
        self.best_individual = None

    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            architecture = self.search_space.sample_architecture()
            individual = Individual(architecture)
            self.population.append(individual)

    def select_parents(self) -> Tuple[Individual, Individual]:
        """使用锦标赛选择法选择父代"""
        tournament_size = 3
        parents = []
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
            parents.append(winner)
        return tuple(parents)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """对两个父代进行交叉操作"""
        if random.random() < self.crossover_rate:
            child_arch = self.search_space.crossover_architecture(
                parent1.architecture,
                parent2.architecture
            )
        else:
            child_arch = parent1.architecture.copy()
        return Individual(child_arch)

    def mutate(self, individual: Individual) -> Individual:
        """对个体进行变异操作"""
        if random.random() < self.mutation_rate:
            mutated_arch = self.search_space.mutate_architecture(
                individual.architecture,
                self.mutation_rate
            )
            return Individual(mutated_arch)
        return individual

    def update_population(self, offspring: List[Individual]):
        """更新种群"""
        # 将子代和父代合并
        combined = self.population + offspring
        # 按适应度排序
        combined.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
        # 选择最优的个体组成新种群
        self.population = combined[:self.population_size]
        # 更新最佳个体
        self.best_individual = self.population[0]

    def run(self, train_loader, val_loader, device) -> Individual:
        """运行进化算法"""
        print("Initializing population...")
        self.initialize_population()

        # 评估初始种群
        for individual in self.population:
            individual.evaluate(train_loader, val_loader, device)

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")

            # 生成子代
            offspring = []
            while len(offspring) < self.population_size:
                # 选择父代
                parent1, parent2 = self.select_parents()

                # 交叉
                child = self.crossover(parent1, parent2)

                # 变异
                child = self.mutate(child)

                # 评估子代
                child.evaluate(train_loader, val_loader, device)
                offspring.append(child)

            # 更新种群
            self.update_population(offspring)

            # 打印当前最佳结果
            best_fitness = self.best_individual.fitness
            print(f"Best fitness: {best_fitness:.4f}")

        return self.best_individual