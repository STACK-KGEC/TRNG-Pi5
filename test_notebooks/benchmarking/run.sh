python spso_trng_experiment.py --rng prng --func rastrigin --dim 10 --runs 30
python spso_trng_experiment.py --rng prng --func ackley --dim 10 --runs 30
python spso_trng_experiment.py --rng prng --func rosenbrock --dim 10 --runs 30
python spso_trng_experiment.py --rng prng --func sphere --dim 10 --runs 30

python spso_trng_experiment.py --rng trng --trng-file readings_1.txt --func rastrigin --dim 10 --runs 30
python spso_trng_experiment.py --rng trng --trng-file readings_1.txt --func ackley --dim 10 --runs 30
python spso_trng_experiment.py --rng trng --trng-file readings_1.txt --func rosenbrock --dim 10 --runs 30
python spso_trng_experiment.py --rng trng --trng-file readings_1.txt --func sphere --dim 10 --runs 30
