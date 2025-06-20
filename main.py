import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from math import sin, cos, pi
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from io import BytesIO
import asyncio
from typing import List, Callable, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
bot = Bot(token="")
dp = Dispatcher()


# Состояния бота
class FunctionStates(StatesGroup):
    waiting_for_function = State()
    algorithm_running = State()


# Доступные тестовые функции (адаптированные для работы с numpy)
TEST_FUNCTIONS = {
    "Schwefel": lambda coords: 418.9829 * len(coords) - np.sum(coords * np.sin(np.sqrt(np.abs(coords)))),
    "Rastrigin": lambda coords: 10 * len(coords) + np.sum(coords ** 2 - 10 * np.cos(2 * np.pi * coords)),
    "Himmelblau": lambda coords: (coords[0] ** 2 + coords[1] - 11) ** 2 + (coords[0] + coords[1] ** 2 - 7) ** 2,
    "Sphere": lambda coords: np.sum(coords ** 2),
    "Rosenbrock": lambda coords: np.sum(100 * (coords[1:] - coords[:-1] ** 2) ** 2 + (1 - coords[:-1]) ** 2)
}

# Параметры для каждой функции
FUNCTION_PARAMS = {
    "Schwefel": {"min": -500, "max": 500, "dim": 2},
    "Rastrigin": {"min": -5.12, "max": 5.12, "dim": 2},
    "Himmelblau": {"min": -5, "max": 5, "dim": 2},
    "Sphere": {"min": -100, "max": 100, "dim": 2},
    "Rosenbrock": {"min": -2.048, "max": 2.048, "dim": 2}
}


# Клавиатура для выбора функции
def get_functions_keyboard():
    buttons = [KeyboardButton(text=name) for name in TEST_FUNCTIONS.keys()]
    return ReplyKeyboardMarkup(keyboard=[buttons], resize_keyboard=True)


# Класс частицы
class Particle:
    def __init__(self, coords: List[float], alpha_p: float = 0.1, alpha_b: float = 0.1):
        self.coords = np.array(coords, dtype=np.float64)
        self.speed = np.zeros_like(self.coords)
        self.personal_best = float('inf')
        self.personal_best_coords = self.coords.copy()
        self.alpha_p = alpha_p
        self.r_p = 0.0
        self.alpha_b = alpha_b
        self.r_b = 0.0

    def update_best(self, target_function: Callable):
        current = target_function(self.coords)
        if current < self.personal_best:
            self.personal_best = current
            self.personal_best_coords = self.coords.copy()

    def move(self, global_best_coords: np.ndarray, inertion: float):
        self.r_p = uniform(0, 1)
        self.r_b = uniform(0, 1)
        cognitive = self.alpha_p * self.r_p * (self.personal_best_coords - self.coords)
        social = self.alpha_b * self.r_b * (global_best_coords - self.coords)
        self.speed = self.speed * inertion + cognitive + social
        self.coords += self.speed


# Класс роя
class Swarm:
    def __init__(self, num_particles: int, dim: int, min_coord: float, max_coord: float):
        self.particles = [Particle(np.random.uniform(min_coord, max_coord, dim))
                          for _ in range(num_particles)]
        self.global_best_coords = self.particles[0].coords.copy()
        self.global_best_value = float('inf')

    def update(self, target_function: Callable):
        for particle in self.particles:
            particle.update_best(target_function)
            if particle.personal_best < self.global_best_value:
                self.global_best_value = particle.personal_best
                self.global_best_coords = particle.personal_best_coords.copy()

    def move(self, inertion: float):
        for particle in self.particles:
            particle.move(self.global_best_coords, inertion)


async def create_plot(swarm: Swarm, func_name: str, iteration: int) -> BytesIO:
    plt.figure(figsize=(10, 8))
    params = FUNCTION_PARAMS[func_name]

    if params["dim"] == 2:
        x = np.linspace(params["min"], params["max"], 100)
        y = np.linspace(params["min"], params["max"], 100)
        X, Y = np.meshgrid(x, y)

        # Векторизованный расчет значений функции
        if func_name == "Himmelblau":
            Z = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2
        elif func_name == "Schwefel":
            Z = 418.9829 * 2 - (X * np.sin(np.sqrt(np.abs(X))) + Y * np.sin(np.sqrt(np.abs(Y))))
        elif func_name == "Rastrigin":
            Z = 10 * 2 + (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y))
        elif func_name == "Sphere":
            Z = X ** 2 + Y ** 2
        elif func_name == "Rosenbrock":
            Z = 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2

        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar()

    coords = np.array([p.coords for p in swarm.particles])
    if params["dim"] == 2:
        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=20)
        plt.scatter(swarm.global_best_coords[0], swarm.global_best_coords[1],
                    c='yellow', s=100, marker='*', edgecolors='black')

    plt.title(f"{func_name} (Итерация {iteration})")
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.set_state(FunctionStates.waiting_for_function)
    await message.answer(
        "🔍 Выберите тестовую функцию для оптимизации:",
        reply_markup=get_functions_keyboard()
    )


@dp.message(FunctionStates.waiting_for_function, F.text.in_(TEST_FUNCTIONS.keys()))
async def function_selected(message: types.Message, state: FSMContext):
    func_name = message.text
    await state.update_data(selected_function=func_name)
    await state.set_state(FunctionStates.algorithm_running)
    await message.answer(
        f"Выбрана функция: {func_name}\n"
        f"Теперь используйте:\n"
        f"/run_new - новые сообщения\n"
        f"/run_edit - редактирование",
        reply_markup=types.ReplyKeyboardRemove()
    )


@dp.message(Command("run_new"))
async def run_new_messages(message: types.Message, state: FSMContext):
    data = await state.get_data()
    if "selected_function" not in data:
        await message.answer("Сначала выберите функцию через /start")
        return

    func_name = data["selected_function"]
    params = {
        "particles": 30,
        "iterations": 50,
        "update_interval": 5,
        **FUNCTION_PARAMS[func_name]
    }

    swarm = Swarm(params["particles"], params["dim"],
                  params["min"], params["max"])

    status_msg = await message.answer(f"🚀 Запуск {func_name}...")

    for i in range(params["iterations"]):
        inertion = 0.9 - (0.5 * i / params["iterations"])
        swarm.update(TEST_FUNCTIONS[func_name])
        swarm.move(inertion)

        if i % params["update_interval"] == 0 or i == params["iterations"] - 1:
            buf = await create_plot(swarm, func_name, i + 1)
            await message.answer_photo(
                BufferedInputFile(buf.getvalue(), filename="plot.png"),
                caption=f"{func_name} - Итерация {i + 1}/{params['iterations']}"
            )
            buf.close()
            await asyncio.sleep(0.3)

    await status_msg.edit_text(
        f"✅ {func_name} - Результат:\n"
        f"Лучшее значение: {swarm.global_best_value:.4f}\n"
        f"Координаты: {np.round(swarm.global_best_coords, 4)}"
    )
    await state.clear()


@dp.message(Command("run_edit"))
async def run_edit_message(message: types.Message, state: FSMContext):
    data = await state.get_data()
    if "selected_function" not in data:
        await message.answer("Сначала выберите функцию через /start")
        return

    func_name = data["selected_function"]
    params = {
        "particles": 30,
        "iterations": 50,
        "update_interval": 5,
        **FUNCTION_PARAMS[func_name]
    }

    swarm = Swarm(params["particles"], params["dim"],
                  params["min"], params["max"])
    status_msg = await message.answer(f"✏️ Запуск {func_name} (режим редактирования)...")
    plot_msg = None

    for i in range(params["iterations"]):
        inertion = 0.9 - (0.5 * i / params["iterations"])
        swarm.update(TEST_FUNCTIONS[func_name])
        swarm.move(inertion)

        if i % params["update_interval"] == 0 or i == params["iterations"] - 1:
            buf = await create_plot(swarm, func_name, i + 1)
            input_file = BufferedInputFile(buf.getvalue(), filename="plot.png")

            if plot_msg is None:
                plot_msg = await message.answer_photo(
                    input_file,
                    caption=f"{func_name} - Итерация {i + 1}/{params['iterations']}"
                )
            else:
                try:
                    await bot.edit_message_media(
                        chat_id=message.chat.id,
                        message_id=plot_msg.message_id,
                        media=types.InputMediaPhoto(
                            media=input_file,
                            caption=f"{func_name} - Итерация {i + 1}/{params['iterations']}"
                        )
                    )
                except Exception as e:
                    logger.error(f"Ошибка редактирования: {e}")
                    plot_msg = await message.answer_photo(input_file)

            buf.close()
            await asyncio.sleep(0.3)

    await status_msg.edit_text(
        f"✅ {func_name} - Результат:\n"
        f"Лучшее значение: {swarm.global_best_value:.4f}\n"
        f"Координаты: {np.round(swarm.global_best_coords, 4)}"
    )
    await state.clear()


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())