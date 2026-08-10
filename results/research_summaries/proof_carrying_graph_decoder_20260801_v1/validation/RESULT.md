# Proof-carrying graph decoder — validation confirmation

Baseline F1: **0.518450**
Proof-graph F1: **0.520729**
Delta: **+0.002279**

| relation | baseline | proof graph | delta |
|---|---:|---:|---:|
| awardWonBy | 0.114372 | 0.114372 | +0.000000 |
| companyTradesAtStockExchange | 0.742905 | 0.749905 | +0.007000 |
| countryLandBordersCountry | 0.961540 | 0.967265 | +0.005725 |
| hasArea | 0.420000 | 0.420000 | +0.000000 |
| hasCapacity | 0.180000 | 0.180000 | +0.000000 |
| personHasCityOfDeath | 0.470000 | 0.470000 | +0.000000 |

The strict two-family-margin rule is a development-informed refinement of a looser rule whose failure ledger was inspected on this validation split. It is therefore validation-tuned evidence, not a fresh blind-test confirmation; its real generalization test is the competition test.
