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
