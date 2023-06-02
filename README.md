# Sieve of Eratosthenes

Sieve of Eratosthenes is algorithm for finding all prime numbers up to any given.

For more details on the Sieve of Eratosthenes algorithm, you can refer to **[Wikipedia](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)**.

## Usage

```
    1. Make sure you have Python installed on your system.
    2. Clone or download the script.
    3. Open a terminal or command prompt.
    4. Navigate to the directory containing the script.
    5. Run the script by executing the following command
```

## Command
```
python eratosthenes.py
```

## Requirements
```
- Python 3.10
- NumPy 1.24
```

## Performance

|Number         |Primes     |Duration (Basic / Odd Number)  |Peak memory (Basic / Odd Number)   |
|---            |---        |---                            |---                                |
|10             |4          |0.000076 / 0.000169            |0.005 / 0.010                      |
|100            |25         |0.000113 / 0.000270            |0.001 / 0.002                      |
|1,000          |168        |0.000066 / 0.000191            |0.003 / 0.004                      |
|10,000         |1,229      |0.000143 / 0.000268            |0.020 / 0.025                      |
|100,000        |9,592      |0.000437 / 0.000409            |0.169 / 0.195                      |
|1,000,000      |78,498     |0.002943 / 0.002182            |1.553 / 1.676                      |
|10,000,000     |664,579    |0.071657 / 0.029560            |14.608 / 14.910                    |
|100,000,000    |5,761,455  |0.936934 / 0.438990            |139.325 / 135.598                  |
|1,000,000,000  |50,847,534 |10.159814 / 5.117700           |1341.611 / 1252.710                |

* **Basic** is eratosthenes() run 2 3 4 5 ... to √n
* **Odd Number** is eratosthenes_odd() run 3 5 7 9 ... to √n