This work is the result of the microproject “AI ethics and responsible AI” belonging to HumaneAI-net. This project seeks to formalize how AI agents can interpret and act according to socio-ethical values within complex, layered social contexts. The goal is to integrate stakeholders' values, norms, and conventions into AI decision-making processes, enabling ethics-by-design approaches where agents align their behavior with the broader social values relevant to their interactions. While existing literature on value systems often focuses on either abstract or specific contexts, this project addresses the challenge of multiple overlapping contexts where value preferences may conflict or shift depending on the situation.

An exploratory exercise of values and preferences was carried out which would guide this formalization. This resulted in different outcomes: 

1. An agent-based model (ABM) with values and value preferences as part of agents’ deliberation as well as contexts expressed as value preferences. Contexts affected agents in such a way that it may result in them temporary changing their value preferences according to an effort function (we assumed such effort would be proportional to how much importance each agent gave to their values)
2. A software implementing the above-mentioned model
3. A workshop paper presenting the results of experimenting with such ABM. The highlights of these results are:
	- Adding value preferences and contexts show more realistic results as one would expect
	- Given our grounding in the Schwartz’s circumflex model of values and a value preference, some value orders are more prone to shift than others, that is, they are more flexible in terms of changing their preferences
These outputs are intended to guide the subsequent formalization and architecture design which is the goal of the micro project.
The code of the ABM can be found in this repository: https://github.com/HPAI-BSC/value-based-water-consumption/
The paper can be cited as: Oliva-Felipe, L., Lobo, I., McKinlay, J., Dignum, F., De Vos, M., Cortés, U., Cortés, A. (2024). Context Matters: Contextual Value-Based Deliberation in Water Consumption Scenarios. In: XXX, Y., et al. Artificial Intelligence. ECAI 2024 International Workshops. ECAI 2024. Communications in Computer and Information Science, vol XXXX. Springer, Cham. https://doi.org/XXXX/YYYYY (accepted, to be printed)

## How to run:
 
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
