# -*- coding: utf-8 -*-
from random import randint, choice, seed
from math import log2, ceil
from itertools import zip_longest
from time import sleep
import sys 
from contextlib import suppress

from typing import Optional, Iterator

from serial import Serial
from serial.tools import list_ports
from bitarray import bitarray
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

UC_DRIVER_BITS = 8 #Nescessita ser multiplo de 4

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
    generator_type: str,
    samples: int,
    auto_adjust_prbs: bool=False,
    **kwargs
    ):
    """Modifica o estado da porta digital de saída de acordo com o sinal escolhido

    Args:
        generator_type (str): Padrão do sinal gerado: 'prbs', 'random', 'square'
        auto_adjust_prbs (bool, optional): Ajusta a quantidade de bits do gerador PRBS automaticamente? Defaults to False.
    """
    generators_dict = {
        'prbs': infinite_prbs_loop,
        'random': infinite_random_loop,
        'square': infinite_square_loop
    }
    samples += 1 #Número mínimo de amostras para garantir o sinal completo
    samples = samples - (samples % UC_DRIVER_BITS) + (UC_DRIVER_BITS if samples % UC_DRIVER_BITS else 0)
    if (generator_type == 'prbs') and auto_adjust_prbs:
        kwargs.update({'prbs_bits': ceil(log2(samples))})
    loop_generator = generators_dict[generator_type](**kwargs)
    output_signal = bitarray([e for e,_ in zip(loop_generator,range(samples))])
    return output_signal


def encode_signal_slice(signal_slice: bitarray) -> str:
    sub_slice = [bitarray(s) for s in zip(*(iter(signal_slice),) * 4)]
    hex_str_slice = [hex(int(b.to01(), 2))[-1] for b in sub_slice]
    return ''.join(hex_str_slice)


def encode_signal(input_signal: bitarray) -> str:
    sliced_signal = [bitarray(bit_sequence) for bit_sequence in zip(*(iter(input_signal),) * UC_DRIVER_BITS)]
    hexed_signal = [encode_signal_slice(bit_sequence) for bit_sequence in sliced_signal]
    encoded_signal = ''.join(hexed_signal)
    return encoded_signal


def generate_encoded_signal(
    generator_type:str,
    samples:int,
    time_interval:float,
    auto_adjust_prbs: bool=False,
    **kwargs
    ) -> list[str]:
    raw_signal = generate_signal(
            generator_type=generator_type,
            samples=samples,
            auto_adjust_prbs=auto_adjust_prbs,
            **kwargs
        )
    encoded_signal = encode_signal(input_signal=raw_signal)
    mili_half_period = round(time_interval * 1000)
    serial_output = f'T{mili_half_period:04d}S{encoded_signal}X'
    return bytes(serial_output, 'utf-8'), raw_signal.tolist()


def get_driver_port():
    def is_valid_port(port_description):
        valid_descriptions_list = ['CH340', 'USB2.0-Serial', 'Arduino']
        return len([valid_device in port_description for valid_device in valid_descriptions_list]) > 0
    ports = [p.device for p in list_ports.comports() if is_valid_port(p.description)]
    return ports[0]


def send_signal_to_driver(encoded_signal: bytes) -> str:
    def serial_driver():
        return Serial(
                    port=get_driver_port(),
                    baudrate=9600,
                    timeout=30,
                )
    with serial_driver() as driver:
        sleep(3)
        driver.write(encoded_signal)
        while True:
            cmd = driver.read(1)
            if cmd != b'T':
                sleep(1)
                continue
            serial_message = driver.read_until(expected=b'X')
            break
    return serial_message.decode('utf-8')[:-1]


def decode_sampled_data(sampled_data: str) -> list:
    hex_sampled_data = [''.join(d) for d in zip(*(iter(sampled_data),) * 3)]
    converted_data = [float(int(x, 16)/1024) for x in hex_sampled_data]
    return converted_data


def is_debugging() -> bool:
    with suppress(AttributeError):
        return bool(sys.gettrace())


if __name__ == '__main__':

    # debug_time_interval = 1e-3
    # debug_samples = 300

    # """ RLC """
    # debug_time_interval = 15e-3
    # debug_samples = 300
    
    # """ RC """
    debug_time_interval = 3e-3
    debug_samples = 300

    encoded_signal, u = generate_encoded_signal(
        generator_type='prbs',
        samples=debug_samples,
        time_interval=debug_time_interval,
        rng_seed=1,
        auto_adjust_prbs=True,
    )

    encoded_output = send_signal_to_driver(encoded_signal)
    y = decode_sampled_data(encoded_output)
    k = np.arange(len(y))

    train_ratio = 0.8
    n_train = ceil(train_ratio * len(u))
    u, y = np.asarray(u, dtype=float)*5, np.asarray(y, dtype=float)*5
    u_train, y_train = u[:n_train].reshape(-1,1), y[:n_train].reshape(-1,1)
    u_eval, y_eval = u[n_train:].reshape(-1,1), y[n_train:].reshape(-1,1)

    model = FROLS(
        order_selection=True,
        n_info_values=15,
        extended_least_squares=False,
        ylag=5, xlag=5,
        info_criteria='bic',
        estimator='least_squares',
        basis_function=Polynomial(degree=3)
    )
    model.fit(X=u_train, y=y_train)

    plt.figure()
    plt.step(k, u.flatten(), where='post', marker='o', markersize=3, markerfacecolor='black', label='u')
    plt.plot(k, y.flatten(), 'ro--', markersize=3, markerfacecolor='black', label='y')
    plt.legend()
    plt.grid()
    plt.show(block=False)

    plt.figure()
    xaxis = np.arange(1, model.n_info_values + 1)
    plt.plot(xaxis, model.info_values, 'bo--')
    plt.grid()
    plt.xlabel('n_terms')
    plt.ylabel('Information Criteria')

    if is_debugging():
        plt.show(block=True)
        model.n_terms = 3
    else:
        plt.show(block=False)
        model.n_terms = int(input('\nNúmero de regressores do modelo: '))

    y_predicted = model.predict(X=u_eval, y=y_eval)
    rrse = root_relative_squared_error(y_eval, y_predicted)
    result_model = pd.DataFrame(
        results(
            model.final_model, model.theta, model.err,
            model.n_terms, err_precision=8, dtype='sci'
        ),
        columns=['Regressors', 'Parameters', 'ERR'],
    )
    print(f'\nRRSE: {rrse}\n')
    print(f'\n{result_model.to_string()}\n')

    plt.ion()
    plot_results(y=y_eval, yhat=y_predicted, n=1000)
    ee = compute_residues_autocorrelation(y_eval, y_predicted)
    plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
    x1e = compute_cross_correlation(y_eval, y_predicted, u_eval)
    plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
    plt.ioff()
    plt.show(block=True)

    print('fim')
