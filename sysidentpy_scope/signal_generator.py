# -*- coding: utf-8 -*-
from time import time, sleep
from random import randint, choice, seed
from math import log2, ceil

from typing import Optional, Iterator

from bitarray import bitarray
from tqdm import tqdm

import pigpio

try:
    from gpiozero import Button as DigitalInput
    STOP_BUTTON = DigitalInput(2)
    DEVICE = pigpio.pi()

except ImportError:
    #Debug
    from dataclasses import dataclass
    @dataclass
    class DigitalInput:
        pin: int
        is_pressed: bool=False
        is_held: bool=False
        is_active: bool=False
        when_activated: object=None
        when_deactivated: object=None
        when_held: object=None
        when_pressed: object=None
        when_released: object=None

    @dataclass
    class RaspberryPI:
        id: int=0
        def pi(self): return self
        def set_mode(self, *args, **kwargs): return 0
        def wave_add_generic(self, *args, **kwargs): return 0
        def wave_create(self, *args, **kwargs): return 0
        def wave_send_once(self, *args, **kwargs): return 0
        def wave_tx_busy(self, *args, **kwargs): return 1
        def wave_tx_stop(self, *args, **kwargs): return 0
        def wave_tx_at(self, *args, **kwargs): return 0
        def wave_delete(self, *args, **kwargs): return 0
        def stop(self, *args, **kwargs): return 0

    STOP_BUTTON = DigitalInput(2)
    DEVICE = RaspberryPI().pi()

def force_loop_break():
    """Interrompe a execução de uma onda gerada
    """
    if DEVICE.wave_tx_busy():
        DEVICE.wave_tx_stop()
        wave_id = DEVICE.wave_tx_at()
        DEVICE.wave_delete(wave_id)
        DEVICE.stop()

OUTPUT_PORT = 4
STOP_BUTTON.when_pressed = force_loop_break

def prbs_sequence(prbs_bits:int, rng_seed:int) -> bitarray:
    """Gera uma sequência de int do tipo PRBS

    Args:
        prbs_bits (int): Quantidade de bits do gerador PRBS
        rng_seed (int): Valor inicial do gerador PRBS

    Returns:
        bitarray: Sinal PRBS
    """
    prbs_types = {
        3: {'bit_1':2 , 'bit_2':1 }, #size = 7
        4: {'bit_1':3 , 'bit_2':2 }, #size = 15
        5: {'bit_1':4 , 'bit_2':2 }, #size = 31
        6: {'bit_1':5 , 'bit_2':4 }, #size = 63
        7: {'bit_1':6 , 'bit_2':5 }, #size = 127
        9: {'bit_1':8 , 'bit_2':4 }, #size = 511
       10: {'bit_1':9 , 'bit_2':6 }, #size = 1_023
       11: {'bit_1':10, 'bit_2':8 }, #size = 2_047
       15: {'bit_1':14, 'bit_2':13}, #size = 32_767
       17: {'bit_1':16, 'bit_2':13}, #size = 131_071
       18: {'bit_1':17, 'bit_2':10}, #size = 262_143
       20: {'bit_1':19, 'bit_2':16}, #size = 1_048_575
       21: {'bit_1':20, 'bit_2':18}, #size = 2_097_151
       22: {'bit_1':21, 'bit_2':20}, #size = 4_194_303
       23: {'bit_1':22, 'bit_2':17}, #size = 8_388_607
    #  25: {'bit_1':24, 'bit_2':21}, #size = 33_554_431
    #  28: {'bit_1':27, 'bit_2':24}, #size = 268_435_455
    #  29: {'bit_1':28, 'bit_2':26}, #size = 536_870_911
    #  31: {'bit_1':30, 'bit_2':27}, #size = 2_147_483_647
    }
    if prbs_bits >= max(prbs_types.keys()):
        prbs_bits = max(prbs_types.keys())
    else:
        prbs_bits = min(b for b in prbs_types.keys() if b >= prbs_bits)
    size = (2**prbs_bits) - 1
    bit_1 = prbs_types[prbs_bits]['bit_1']
    bit_2 = prbs_types[prbs_bits]['bit_2']
    start_value = randint(0,size-1) if rng_seed is None else rng_seed
    start_value = int(min(max(start_value, 0), size-1))

    bit_sequence = bitarray([start_value & 0x1])
    new_value = start_value
    for _ in tqdm(range(size-1), desc=f'Gerando sinal (PRBS{prbs_bits:d})'):
        new_bit = ~((new_value>>bit_1) ^ (new_value>>bit_2)) & 0x1
        new_value = ((new_value<<1) + new_bit) & size
        #Fechou um período ou atingiu estado proibido: retorna o resultado
        if (new_value == start_value) or (new_value == size):
            return bit_sequence
        bit_sequence.append(bool(new_bit))
    return bit_sequence


def infinite_prbs_loop(prbs_bits:int=15, rng_seed:Optional[int]=None) -> Iterator[bool]:
    """Gerador do loop de sinal em PRBS

    Args:
        prbs_bits (int): Quantidade de bits do gerador PRBS
        rng_seed (Optional[int]): Valor inicial do gerador PRBS

    Yields:
        Iterator[bool]: Estado atual
    """
    sequence = prbs_sequence(prbs_bits=prbs_bits, rng_seed=rng_seed)
    len_sequence = len(sequence)
    i = 0
    while True:
        i = i+1 if i+1 <= len_sequence else 1
        yield bool(sequence[i-1])


def infinite_random_loop(rng_seed:Optional[int]=None, **kwargs) -> Iterator[bool]:
    """Gerador do loop em Random

    Yields:
        Iterator[bool]: Estado atual
    """
    if rng_seed is not None:
        seed(rng_seed)
    outputs = bitarray([True, False])
    while True:
        yield bool(choice(outputs))


def infinite_square_loop(**kwargs) -> Iterator[bool]:
    """Gerador do loop de sinal quadrado

    Yields:
        Iterator[bool]: Estado atual
    """
    output = False
    while True:
        output = not output
        yield output


def generate_output_signal(
    generator_type:str,
    frequency:float,
    time_interval:Optional[float],
    auto_adjust_prbs: bool=False,
    **kwargs
    ):
    """Modifica o estado da porta digital de saída de acordo com o sinal escolhido

    Args:
        generator_type (str): Padrão do sinal gerado: 'prbs', 'random', 'square'
        frequency (float): Frequência do sinal [Hz]
        time_interval (float): Tempo que o sinal ficará sendo gerado [s].
        auto_adjust_prbs (bool, optional): Ajusta a quantidade de bits do gerador PRBS automaticamente? Defaults to False.
    """
    generators_dict = {
        'prbs': infinite_prbs_loop,
        'random': infinite_random_loop,
        'square': infinite_square_loop
    }
    samples = ceil((2 * frequency) * time_interval) + 1
    if (generator_type == 'prbs') and auto_adjust_prbs:
        kwargs.update({'prbs_bits': ceil(log2(samples))})
    loop_generator = generators_dict[generator_type](**kwargs)
    output_signal = [e for e,_ in zip(loop_generator,range(samples))]
    return output_signal


def send_signal_to_port(input_signal: bitarray, frequency: float):
    delay_us = round((1 / frequency) * 0.5 * 1e6)
    pulse_train = list()
    for state in input_signal:
        if state:
            pulse_train.append(pigpio.pulse(1<<OUTPUT_PORT, 0, delay_us))
            continue
        pulse_train.append(pigpio.pulse(0, 1<<OUTPUT_PORT, delay_us))
    DEVICE.set_mode(OUTPUT_PORT, pigpio.OUTPUT)
    DEVICE.wave_add_generic(pulse_train)
    wave_id = DEVICE.wave_create()
    if wave_id > 0:
        DEVICE.wave_send_once(wave_id)


if __name__ == '__main__':
    debug_time_interval = 15
    debug_frequency = 1000
    signal = generate_output_signal(
        generator_type='prbs',#'prbs',
        frequency=debug_frequency,
        time_interval=debug_time_interval,
        rng_seed=1,
        # prbs_bits=5,
        auto_adjust_prbs=True,
    )

    # send_signal_to_port(
    #     input_signal=signal,
    #     frequency=debug_frequency
    # )

    # from sysidentpy_scope.tools.statstools import correlation, plot_correlation
    # import numpy as np
    # import matplotlib.pyplot as plt
    # plot_time = np.arange(0, time_interval+(1/(2*frequency)), 1/(2*frequency))
    # plt.step(plot_time, signal, where='post')
    # plt.grid()
    # plt.show()
    # auto_correlation, k, limits = correlation(y=signal)
    # plot_correlation(auto_correlation, k, limits)

    print('fim')
