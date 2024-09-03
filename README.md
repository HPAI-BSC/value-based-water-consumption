How to run:
 
```bash
mkdir -p results/daily
mkdir -p results/monthly
docker build -t water .
docker run -v $(pwd)/results:/code/results run water
```

To set number of repetitions (e.g. 5):

```
docker run -v $(pwd)/results:/code/results run water 5
```

To set random seed (e.g. 42):

```
docker run -v $(pwd)/results:/code/results run water 5 42
```
## Citation
Oliva-Felipe, L., Lobo, I., McKinlay, J., Dignum, F., De Vos, M., Cortés, U., Cortés, A. (2024). Context Matters: Contextual Value-Based Deliberation in Water Consumption Scenarios. In: XXX, Y., et al. Artificial Intelligence. ECAI 2024 International Workshops. ECAI 2024. Communications in Computer and Information Science, vol XXXX. Springer, Cham. https://doi.org/XXXX/YYYYY (accepted, to be printed) 
