# -*- coding: utf-8 -*-
from random import randint, choice, seed
from math import log2, ceil
from itertools import zip_longest

from typing import Optional, Iterator

from bitarray import bitarray
from tqdm import tqdm

SERIAL_MESSAGE_LENGTH = 10

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


def generate_signal(
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
    samples = ceil((2 * frequency) * time_interval) + 1 #Número mínimo de amostras para garantir o sinal completo
    samples = samples - (samples % 4) + (4 if samples % 4 else 0) #Garante que o número de amostras seja múltiplo de 4
    if (generator_type == 'prbs') and auto_adjust_prbs:
        kwargs.update({'prbs_bits': ceil(log2(samples))})
    loop_generator = generators_dict[generator_type](**kwargs)
    output_signal = bitarray([e for e,_ in zip(loop_generator,range(samples))])
    return output_signal


def encode_signal(input_signal: bitarray) -> list[str]:
    sliced_signal = [bitarray(bit_sequence) for bit_sequence in zip(*(iter(input_signal),) * 4)]
    hexed_signal = [hex(int(bit_sequence.to01(), 2))[-1] for bit_sequence in sliced_signal]
    sliced_hex = [str_sequence for str_sequence in zip_longest(*(iter(hexed_signal),) * SERIAL_MESSAGE_LENGTH)]
    encoded_signal = [''.join([s for s in signal_slice if s is not None]) for signal_slice in sliced_hex]
    return encoded_signal


# def generate_encoded_signal() -> list[str]:

if __name__ == '__main__':
    debug_time_interval = 1
    debug_frequency = 15
    signal = generate_signal(
        generator_type='prbs',
        frequency=debug_frequency,
        time_interval=debug_time_interval,
        rng_seed=1,
        # prbs_bits=5,
        auto_adjust_prbs=True,
    )
    encoded_signal = encode_signal(signal)

    # from sysidentpy_scope.tools.statstools import correlation, plot_correlation
    # import numpy as np
    # import matplotlib.pyplot as plt
    # plot_time = np.arange(0, debug_time_interval+(1/(2*debug_frequency)), 1/(2*debug_frequency))
    # plt.step(plot_time, [s for s in signal], where='post')
    # plt.grid()
    # plt.show()
    # auto_correlation, k, limits = correlation(y=[s for s in signal])
    # plot_correlation(auto_correlation, k, limits)

    print('fim')
