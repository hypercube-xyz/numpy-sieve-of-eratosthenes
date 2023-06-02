import gc
import numpy as np
import time
import tracemalloc

# basic algorithm
def eratosthenes(n: int):
    number = n + 1
    is_prime = np.ones(number, dtype=bool)

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i**2::i] = False

    primes = np.flatnonzero(is_prime)[2:]

    return primes

# odd numbers algorithm
def eratosthenes_odd(n: int):
    number = (n - 1) // 2 + 1
    is_prime = np.ones(number, dtype=bool)

    for i in range(3, int(n**0.5) + 1, 2):
        if (is_prime[i//2]):
            is_prime[i*i//2::i] = False

    primes = np.insert(2 * np.flatnonzero(is_prime)[1:] + 1, 0, 2)

    return primes

def main():
    while True:
        print("\nChoose a mode of operation")
        print("1. Basic (recommend if less than 1,000,000)")
        print("2. Odd Numbers (recommend if more than 1,000,000)")

        mode = input("\nEnter mode (or 'q' to quit): ")
        if mode.lower() == 'q':
            print("Exit Program")
            exit()

        try:
            mode = int(mode)
            if mode == 1 or mode == 2:
                break
            else:
                print("Error: Value Invalid")
                continue
        except:
            print("Error: Value Invalid")
            continue

    while True:
        number = input("\nEnter number (or 'q' to quit): ")
        if number.lower() == 'q':
            print("Exit Program")
            break

        try:
            number = int(number)
            if number < 2 or number > 1000000000:
                print("Error: Value must be 2 - 1000000000\n")
                continue
        except Exception:
            print("Error: Value Invalid")
            continue

        try:
            tracemalloc.start()
            start = time.perf_counter()

            if mode == 2:
                primes = eratosthenes_odd(number)
            else:
                primes = eratosthenes(number)

            end = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()

            print("Primes: ", primes)
            print("Found prime numbers: ", len(primes))
            
            print("\nDuration: %.6f s." % (end - start))
            print("Current memory: %.3f MB" % (current / (1024 * 1024)))
            print("Peak memory: %.3f MB" % (peak / (1024 * 1024)))
            tracemalloc.stop() 
        finally:
            # clean up memory
            del primes
            gc.collect()

if __name__ == '__main__':
    main()